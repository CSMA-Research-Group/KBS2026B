"""Shared policy engine. Only scheduling and the declared ablations differ.

All policies can use the same compact families, geometry cache, incremental
concrete frontiers, abstract certificate, independent checks and early stops.
Neither the generator's feasibility stratum nor any reference is an argument.
"""
from collections import Counter, deque
import heapq
import time

from e3_cpu_correctness.core import PROOF, verify_certificate
from innovation_stage3.storage import stable
from .cases import bounds, member, member_at
from .proofs import check_closure, check_witness

METHODS = {
    "CATALOG_SHARED_GBFS": {"schedule": "interleaved", "eager": True, "certificate": True},
    "LAZY_SHARED_GBFS": {"schedule": "interleaved", "eager": False, "certificate": True},
    "LAZY_CERT_FIRST": {"schedule": "cert_first", "eager": False, "certificate": True},
    "DECISION_SCHEDULER": {"schedule": "decision", "eager": False, "certificate": True},
    "ABLATION_EAGER": {"schedule": "decision", "eager": True, "certificate": True},
    "ABLATION_NO_CERT": {"schedule": "decision", "eager": False, "certificate": False},
    "ABLATION_FIXED_ORDER": {"schedule": "fixed", "eager": False, "certificate": True},
}
COST_KEYS = ("candidate_materializations", "family_queries", "heuristic_queries", "geometry_queries",
             "geometry_cache_hits", "geometry_cell_queries", "concrete_expansions", "abstract_expansions",
             "certificate_builds", "certificate_checks", "certificate_check_cell_queries",
             "concrete_closure_checks", "concrete_check_cell_queries", "witness_checks", "witness_check_cell_queries")


class BudgetExceeded(Exception):
    pass


class Geometry:
    def __init__(self, owner):
        self.owner = owner
        self.world = owner.public["world"]
        self.obstacles = set(map(tuple, self.world["blocked"]))
        self.cache = {}

    def clear(self, xy, width, height):
        self.owner.check_budget()
        self.owner.cost["geometry_queries"] += 1
        key = tuple(xy), width, height
        if key in self.cache:
            self.owner.cost["geometry_cache_hits"] += 1
            return self.cache[key]
        x, y = xy
        good = True
        for dx in range(-(width // 2), width // 2 + 1):
            for dy in range(-(height // 2), height // 2 + 1):
                self.owner.cost["geometry_cell_queries"] += 1
                px, py = x + dx, y + dy
                if not (0 <= px < self.world["width"] and 0 <= py < self.world["height"]) or (px, py) in self.obstacles:
                    good = False
        self.cache[key] = good
        return good


class ConcreteGraph:
    def __init__(self, owner, config):
        self.owner, self.config = owner, config
        self.parents, self.frontier = {}, []
        self.expanded = 0
        self.closed = False
        source = tuple(config["start"])
        if self.clear(source):
            self.parents[source] = None
            heapq.heappush(self.frontier, (owner.distance(config["id"], source[:2]), source))

    def clear(self, pose, swept=False):
        a, b = self.config["footprint"]
        if swept:
            a = b = max(a, b)
        elif pose[2]:
            a, b = b, a
        return self.owner.geometry.clear(pose[:2], a, b)

    def refresh(self):
        self.frontier = [(self.owner.distance(self.config["id"], p[:2]), p) for _, p in self.frontier]
        heapq.heapify(self.frontier)

    def path(self, target):
        result = []
        while target is not None:
            result.append(list(target))
            target = self.parents[target]
        return list(reversed(result))

    def proof(self):
        return {"configuration_id": self.config["id"], "world_sha256": self.owner.world_hash,
                "parents": [[list(n), list(p) if p is not None else None] for n, p in sorted(self.parents.items())]}

    def step(self):
        self.owner.check_budget()
        if not self.frontier:
            self.owner.cost["concrete_closure_checks"] += 1
            proof = self.proof()
            def charge():
                self.owner.cost["concrete_check_cell_queries"] += 1
            result = check_closure(self.owner.public["world"], self.config, proof, self.owner.check_budget, charge)
            self.owner.check_budget()
            if not result["valid"]:
                raise ValueError("Concrete closure checker rejected the policy proof")
            self.closed = True
            self.owner.closures.append(proof)
            self.owner.resolved.update(f["id"] for f in self.owner.public["families"] if f["configuration_id"] == self.config["id"])
            return
        _, node = heapq.heappop(self.frontier)
        self.expanded += 1
        self.owner.cost["concrete_expansions"] += 1
        for f in self.owner.public["families"]:
            if f["configuration_id"] != self.config["id"] or f["id"] in self.owner.resolved:
                continue
            self.owner.cost["family_queries"] += 1
            i = member_at(f, node[:2])
            if i is not None:
                goal = self.owner.materialize(f, i)
                if goal["goal_id"] not in self.owner.witnesses:
                    path = self.path(node)
                    self.owner.cost["witness_checks"] += 1
                    checked = check_witness(self.owner.public, goal, path)
                    self.owner.cost["witness_check_cell_queries"] += checked["cell_queries"]
                    self.owner.check_budget()
                    if not checked["valid"]:
                        raise ValueError("Positive path failed replay")
                    self.owner.witnesses[goal["goal_id"]] = {"goal": goal, "path": path}
                    self.owner.family_positives[f["id"]] += 1
                    if self.owner.family_positives[f["id"]] == f["count"]:
                        self.owner.resolved.add(f["id"])
                    self.owner.refresh()
        # Do not claim closure after an early stop at a positive witness.
        if self.owner.ready():
            return
        x, y, o = node
        following = [(x + 1, y, o), (x, y + 1, o), (x - 1, y, o), (x, y - 1, o)]
        if self.owner.public["world"]["rotation"] == "square_sweep" and self.clear(node, True):
            following.append((x, y, 1 - o))
        for nxt in following:
            if nxt not in self.parents and self.clear(nxt):
                self.parents[nxt] = node
                heapq.heappush(self.frontier, (self.owner.distance(self.config["id"], nxt[:2]), nxt))


class AbstractGraph:
    def __init__(self, owner):
        self.owner = owner
        self.side = min(min(c["footprint"]) for c in owner.public["world"]["configurations"])
        self.nodes = {tuple(c["start"][:2]) for c in owner.public["world"]["configurations"]}
        self.queue = deque(sorted(self.nodes))
        self.complete = False
        if not all(owner.geometry.clear(p, self.side, self.side) for p in self.nodes):
            raise ValueError("Invalid relaxed initial state in cost design")

    def step(self):
        self.owner.check_budget()
        if not self.queue:
            proof = {"schema": PROOF, "world_sha256": self.owner.world_hash, "side": self.side,
                     "scope_ids": sorted(c["id"] for c in self.owner.public["world"]["configurations"]),
                     "nodes": [list(p) for p in sorted(self.nodes)]}
            self.owner.cost["certificate_builds"] += 1
            self.owner.cost["certificate_checks"] += 1
            tick = time.process_time()
            checked = verify_certificate(self.owner.public["world"], proof)
            self.owner.certificate_check_seconds += time.process_time() - tick
            self.owner.cost["certificate_check_cell_queries"] += checked["cost"]["cell_queries"]
            self.owner.check_budget()
            if not checked["valid"]:
                raise ValueError("Abstract closure checker rejected certificate")
            self.owner.certificate = proof
            self.complete = True
            # Sound, deliberately conservative bounding-box disjointness. No
            # enumerating the members of an excluded family is needed here.
            xs, ys = zip(*self.nodes)
            lo_x, hi_x, lo_y, hi_y = min(xs), max(xs), min(ys), max(ys)
            for f in self.owner.public["families"]:
                self.owner.cost["family_queries"] += 1
                a, b, c, d = bounds(f)
                if c < lo_x or a > hi_x or d < lo_y or b > hi_y:
                    self.owner.resolved.add(f["id"])
                    self.owner.abstract_exclusions.append(f["id"])
            self.owner.refresh()
            return
        x, y = self.queue.popleft()
        self.owner.cost["abstract_expansions"] += 1
        for p in ((x + 1, y), (x, y + 1), (x - 1, y), (x, y - 1)):
            if p not in self.nodes and self.owner.geometry.clear(p, self.side, self.side):
                self.nodes.add(p)
                self.queue.append(p)


class Policy:
    def __init__(self, public, method, budget_seconds, settings, start_cpu=None):
        self.started = time.process_time() if start_cpu is None else start_cpu
        self.budget = budget_seconds
        self.public, self.method, self.settings = public, METHODS[method], settings
        self.cost = Counter({k: 0 for k in COST_KEYS})
        self.world_hash = stable(public["world"])
        self.family_bounds = {f["id"]: bounds(f) for f in public["families"]}
        self.materialized, self.witnesses = {}, {}
        self.family_positives = Counter()
        self.resolved = set()
        self.closures, self.abstract_exclusions, self.actions = [], [], []
        self.graphs = {}
        self.abstract = None
        self.certificate = None
        self.certificate_check_seconds = 0.
        self.geometry = Geometry(self)

    def check_budget(self):
        if self.budget is not None and time.process_time() - self.started >= self.budget:
            raise BudgetExceeded()

    def materialize(self, family, i):
        self.check_budget()
        key = family["id"], i
        if key not in self.materialized:
            self.cost["candidate_materializations"] += 1
            self.materialized[key] = member(family, i)
        return self.materialized[key]

    def distance(self, config_id, xy):
        best = float("inf")
        for f in self.public["families"]:
            if f["configuration_id"] == config_id and f["id"] not in self.resolved:
                self.cost["heuristic_queries"] += 1
                a, b, c, d = self.family_bounds[f["id"]]
                best = min(best, max(a - xy[0], 0, xy[0] - c) + max(b - xy[1], 0, xy[1] - d))
        return best

    def refresh(self):
        for graph in self.graphs.values():
            graph.refresh()

    def ready(self):
        return len(self.witnesses) >= 2 or len(self.resolved) == len(self.public["families"])

    def pick_config(self):
        choices = []
        for c in self.public["world"]["configurations"]:
            if not any(f["configuration_id"] == c["id"] and f["id"] not in self.resolved for f in self.public["families"]):
                continue
            graph = self.graphs.get(c["id"])
            if graph and graph.closed:
                continue
            # Empty open list still needs paid closure verification.
            priority = graph.frontier[0][0] if graph and graph.frontier else -1 if graph else self.distance(c["id"], c["start"][:2])
            choices.append((c["id"] if self.method["schedule"] == "fixed" else priority, c["id"], c))
        if not choices:
            return None, float("inf")
        choice = min(choices, key=lambda r: r[:2])
        c = choice[2]
        return c, (0 if self.method["schedule"] == "fixed" else choice[0])

    def operation(self, round_index, nearest):
        if not self.method["certificate"] or (self.abstract is not None and self.abstract.complete):
            return "concrete"
        schedule = self.method["schedule"]
        if schedule in ("cert_first", "fixed"):
            return "abstract"
        if schedule == "interleaved":
            return "concrete" if round_index % 2 == 0 else "abstract"
        if self.witnesses:
            return "concrete" if nearest <= self.settings["near_goal_distance"] else "abstract"
        return "concrete" if self.cost["concrete_expansions"] < self.settings["initial_witness_expansions"] else "abstract"

    def run(self):
        decision, termination = "ABSTAIN", "budget_exhausted"
        try:
            self.check_budget()
            if self.method["eager"]:
                for f in self.public["families"]:
                    for i in range(f["count"]):
                        self.materialize(f, i)
            round_index = 0
            while not self.ready():
                self.check_budget()
                config, nearest = self.pick_config()
                if config is None:
                    raise ValueError("No remaining operation but unresolved families")
                op = self.operation(round_index, nearest)
                before = dict(self.cost)
                positives_before = len(self.witnesses)
                if op == "abstract":
                    if self.abstract is None:
                        self.abstract = AbstractGraph(self)
                    for _ in range(self.settings["quantum_expansions"]):
                        self.abstract.step()
                        if self.abstract.complete or self.ready():
                            break
                else:
                    if config["id"] not in self.graphs:
                        self.graphs[config["id"]] = ConcreteGraph(self, config)
                    graph = self.graphs[config["id"]]
                    for _ in range(self.settings["quantum_expansions"]):
                        graph.step()
                        if graph.closed or self.ready() or len(self.witnesses) != positives_before:
                            break
                self.actions.append({"operation": op, "configuration_id": config["id"] if op == "concrete" else None,
                                     "concrete_expansions": self.cost["concrete_expansions"] - before["concrete_expansions"],
                                     "abstract_expansions": self.cost["abstract_expansions"] - before["abstract_expansions"],
                                     "witnesses": len(self.witnesses), "resolved_families": sorted(self.resolved)})
                round_index += 1
            self.check_budget()
            decision = "CLARIFY" if len(self.witnesses) >= 2 else "EXECUTE" if self.witnesses else "REJECT"
            termination = "checked_evidence_complete"
        except BudgetExceeded:
            pass
        elapsed = time.process_time() - self.started
        if self.budget is not None and elapsed > self.budget:
            decision, termination = "ABSTAIN", "budget_exhausted"
        return {"decision": decision, "termination": termination,
                "selected_goal": next(iter(self.witnesses)) if decision == "EXECUTE" else None,
                "policy_cpu_seconds": elapsed, "budget_seconds": self.budget,
                "budget_overrun_seconds": max(0., elapsed - self.budget) if self.budget is not None else 0.,
                "cost": dict(self.cost), "actions": self.actions,
                "certificate_check_cpu_seconds": self.certificate_check_seconds,
                "witnesses": list(self.witnesses.values()), "certificate": self.certificate,
                "abstract_exclusions": self.abstract_exclusions, "closures": self.closures,
                "resolved_families": sorted(self.resolved),
                "partial_searches": [{"configuration_id": cid, "expanded": g.expanded, "closed": g.closed,
                                      "open_nodes": len(g.frontier), "discovered_nodes": len(g.parents)}
                                     for cid, g in sorted(self.graphs.items())],
                "partial_abstract": {"expanded": self.cost["abstract_expansions"],
                                     "complete": bool(self.abstract and self.abstract.complete),
                                     "discovered_nodes": len(self.abstract.nodes) if self.abstract else 0}}


def solve(public, method, budget_seconds, settings, start_cpu=None):
    return Policy(public, method, budget_seconds, settings, start_cpu).run()
