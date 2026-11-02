"""Shared, lazy configuration-graph evidence for the finite Stage 2 domain.

All five deterministic policies use the same source checker, empty navigation
graph and footprint-specific loaded graphs. Each node's successors are queried
at most once per policy invocation. Union-find components share discovered
connections between different sources with the same original footprint.

Scheduling is a fixed development heuristic, not a learned or novel algorithm.
Public candidate families stay factored during checking; the complete pair-state
map is materialized for decision auditing and is not a claim of implicit language
enumeration. Query units exclude cached-graph bookkeeping and witness routing;
those computations are included in wall time and auxiliary counters.
"""
from __future__ import annotations

from collections import deque
import copy
import hashlib
import json
import time
from typing import Any

from .domain import configuration_valid, goal_id, motion_neighbors, parse_families, validate_witness


METHODS = (
    "FULL_SHARED_CONFIG", "CHEAP_FIRST_SHARED_CONFIG", "ROUND_ROBIN_SHARED_CONFIG",
    "SIMPLE_HYBRID_SHARED_CONFIG", "FAMILY_PRIORITY_SHARED_CONFIG",
)


class _Graph:
    def __init__(self, engine, footprint=None):
        self.engine = engine
        self.footprint = tuple(footprint) if footprint is not None else None
        self.scope = "EMPTY" if footprint is None else "LOADED:" + "x".join(map(str, footprint))
        self.parents = {}
        self.members = {}
        self.order = {}
        self.adjacency = {}
        self.expanded = set()

    def add(self, node):
        node = tuple(node)
        if node in self.parents:
            self.engine.aux["cache_hits"] += 1
            return
        self.parents[node] = node
        self.members[node] = {node}
        self.order[node] = len(self.order)
        self.adjacency[node] = set()

    def root(self, node):
        parent = self.parents[node]
        if parent != node:
            self.parents[node] = self.root(parent)
        return self.parents[node]

    def union(self, a, b):
        left, right = self.root(a), self.root(b)
        if left == right:
            return
        left, right = min(left, right), max(left, right)
        self.parents[right] = left
        self.members[left].update(self.members.pop(right))

    def connected(self, a, b):
        if a not in self.parents or b not in self.parents:
            return False
        self.engine.aux["cache_hits"] += 1
        return self.root(a) == self.root(b)

    def frontier(self, node):
        if node not in self.parents:
            return None
        pending = self.members[self.root(node)] - self.expanded
        return min(pending, key=self.order.get) if pending else None

    def closed(self, node):
        return node in self.parents and self.frontier(node) is None

    def component_id(self, node):
        return self.scope + ":" + ",".join(map(str, self.root(node)))

    def expand(self, node):
        node = tuple(node)
        if node not in self.parents or node in self.expanded:
            raise RuntimeError("Configuration expansion must query a known, unexpanded node")
        if self.footprint is None:
            neighbors = []
            x, y, _ = node
            for nx, ny in ((x - 1, y), (x, y - 1), (x, y + 1), (x + 1, y)):
                self.engine.aux["transition_checks"] += 1
                following = (nx, ny, 0)
                if configuration_valid(self.engine.runtime, (1, 1), following, self.engine.aux):
                    neighbors.append((following, {"action": "move", "to": [nx, ny]}))
            operation = "expand_empty"
            self.engine.cost["empty_node_expansions"] += 1
        else:
            neighbors = motion_neighbors(self.engine.runtime, self.footprint, node, self.engine.aux)
            operation = "expand_loaded"
            self.engine.cost["loaded_node_expansions"] += 1
        discovered = []
        for following, _ in neighbors:
            following = tuple(following)
            if following not in self.parents:
                discovered.append(list(following))
                self.add(following)
            self.adjacency[node].add(following)
            self.adjacency[following].add(node)
            self.union(node, following)
        self.expanded.add(node)
        self.engine.trace.append({
            "operation": operation, "cost_units": 1, "scope": self.scope,
            "footprint": list(self.footprint) if self.footprint else None,
            "world_sha256": self.engine.world_hash, "node": list(node),
            "neighbors": [list(following) for following, _ in neighbors],
            "discovered": discovered, "component_id": self.component_id(node),
            "component_closed": self.closed(node),
        })

    def route(self, start, target):
        if not self.connected(start, target):
            raise RuntimeError("A routing witness requires an already discovered connection")
        queue, parent = deque([start]), {start: None}
        while queue:
            current = queue.popleft()
            self.engine.aux["graph_traversals"] += 1
            if current == target:
                path = [current]
                while parent[path[-1]] is not None:
                    path.append(parent[path[-1]])
                return list(reversed(path))
            self.engine.aux["cache_hits"] += 1
            for following in sorted(self.adjacency[current]):
                if following not in parent:
                    parent[following] = current
                    queue.append(following)
        raise RuntimeError("Union-find connection disagrees with cached graph edges")

    def closed_certificates(self):
        result = []
        for root in sorted(self.members):
            if self.closed(root):
                nodes = sorted(self.members[root])
                result.append({"component_id": self.component_id(root), "scope": self.scope,
                               "footprint": list(self.footprint) if self.footprint else None,
                               "world_sha256": self.engine.world_hash, "closed": True,
                               "nodes": [list(node) for node in nodes], "expanded_nodes": len(nodes)})
        return result


class _Engine:
    def __init__(self, runtime, families):
        self.runtime = runtime
        self.world_hash = hashlib.sha256(json.dumps(runtime, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()).hexdigest()
        self.objects = {obj["id"]: obj for obj in runtime["objects"]}
        self.targets = {target["id"]: target for target in runtime["targets"]}
        self.families = {family["source_id"]: tuple(sorted(set(family["target_ids"])))
                         for family in families if family["target_ids"]}
        self.sources = sorted(self.families)
        self.certificates = {}
        self.cost = {"source_checks": 0, "empty_node_expansions": 0, "loaded_node_expansions": 0}
        self.aux = {"collision_cell_checks": 0, "transition_checks": 0, "cache_hits": 0, "graph_traversals": 0}
        self.trace = []
        self.empty = _Graph(self)
        self.start = (*runtime["robot"]["start"], 0)
        self.empty.add(self.start)
        self.loaded = {}

    def source_node(self, source):
        obj = self.objects[source]
        return (*obj["cell"], obj["orientation"])

    def empty_node(self, source):
        return (*self.objects[source]["cell"], 0)

    def graph(self, source):
        key = tuple(self.objects[source]["footprint"])
        if key not in self.loaded:
            self.loaded[key] = _Graph(self, key)
        return self.loaded[key]

    def source_check(self, source):
        if source in self.certificates:
            raise RuntimeError("A source certificate was queried twice")
        obj, failures = self.objects[source], []
        if not obj["movable"]:
            failures.append("immovable_source")
        if obj["mass"] > self.runtime["robot"]["payload"]:
            failures.append("payload_exceeded")
        node = self.source_node(source)
        if not failures:
            graph = self.graph(source)
            if node in graph.parents:
                self.aux["cache_hits"] += 1
            elif not configuration_valid(self.runtime, tuple(obj["footprint"]), node, self.aux):
                failures.append("invalid_initial_loaded_configuration")
            if not failures:
                graph.add(node)
        certificate = {"source_id": source, "passed": not failures, "failed_conditions": failures,
                       "scope": "all_semantically_admissible_targets_of_source", "world_sha256": self.world_hash,
                       "footprint": list(obj["footprint"]), "initial_node": list(node)}
        self.certificates[source] = certificate
        self.cost["source_checks"] += 1
        self.trace.append({"operation": "source_check", "cost_units": 1, **copy.deepcopy(certificate)})

    def source_reached(self, source):
        return self.empty_node(source) in self.empty.parents

    def states(self):
        states, reasons = {}, {}
        empty_closed = self.empty.closed(self.start)
        for source in self.sources:
            certificate = self.certificates.get(source)
            for target in self.families[source]:
                candidate = {"action": "place", "source_id": source, "target_id": target}
                identifier = goal_id(candidate)
                state, reason = "UNKNOWN", {"kind": "unresolved"}
                if certificate is not None and not certificate["passed"]:
                    state, reason = "INFEASIBLE", {"kind": "source_necessary_condition", "source_id": source}
                elif empty_closed and not self.source_reached(source):
                    state, reason = "INFEASIBLE", {"kind": "closed_empty_component", "component_id": self.empty.component_id(self.start)}
                elif certificate is not None:
                    graph, node = self.graph(source), self.source_node(source)
                    target_nodes = [(*self.targets[target]["cell"], orientation)
                                    for orientation in self.targets[target]["allowed_orientations"]]
                    connected = next((end for end in target_nodes if graph.connected(node, end)), None)
                    if connected is not None and self.source_reached(source):
                        state, reason = "FEASIBLE", {"kind": "known_empty_and_loaded_paths", "loaded_target_node": list(connected),
                                                     "loaded_component_id": graph.component_id(node)}
                    elif connected is None and graph.closed(node):
                        state, reason = "INFEASIBLE", {"kind": "closed_loaded_component", "component_id": graph.component_id(node)}
                states[identifier], reasons[identifier] = state, reason
        return states, reasons

    def unknown_counts(self, states):
        return {source: sum(states[goal_id({"action": "place", "source_id": source, "target_id": target})] == "UNKNOWN"
                            for target in self.families[source]) for source in self.sources}

    def search_jobs(self, counts, include_unchecked=False, full=False):
        jobs = {}
        needed_empty = 0
        for source in self.sources:
            certificate = self.certificates.get(source)
            eligible = certificate is not None and certificate["passed"]
            needed = full or counts[source] > 0
            if not needed or not (eligible or include_unchecked and certificate is None):
                continue
            if not self.source_reached(source):
                if not self.empty.closed(self.start):
                    needed_empty += len(self.families[source]) if full else counts[source]
                continue
            if not eligible:
                continue
            graph, node = self.graph(source), self.source_node(source)
            pending = graph.frontier(node)
            if pending is not None:
                key = (graph.scope, graph.root(node))
                if key not in jobs:
                    jobs[key] = [graph, pending, 0]
                jobs[key][2] += len(self.families[source]) if full else counts[source]
        if needed_empty:
            jobs[(self.empty.scope, self.empty.root(self.start))] = [self.empty, self.empty.frontier(self.start), needed_empty]
        return [jobs[key] for key in sorted(jobs)]

    def witness(self, candidate, reason):
        source = candidate["source_id"]
        empty_path = self.empty.route(self.start, self.empty_node(source))
        actions = [{"action": "move", "to": list(node[:2])} for node in empty_path[1:]]
        actions.append({"action": "pick", "source_id": source})
        graph = self.graph(source)
        path = graph.route(self.source_node(source), tuple(reason["loaded_target_node"]))
        for previous, following in zip(path, path[1:]):
            if previous[:2] == following[:2]:
                actions.append({"action": "rotate", "orientation": following[2]})
            else:
                actions.append({"action": "move", "to": list(following[:2])})
        actions.append({"action": "place", "target_id": candidate["target_id"]})
        return actions


def _decision(states):
    feasible = sorted(identifier for identifier, state in states.items() if state == "FEASIBLE")
    if len(feasible) >= 2:
        return "CLARIFY", None
    if len(feasible) == 1 and "UNKNOWN" not in states.values():
        return "EXECUTE", feasible[0]
    if states and all(state == "INFEASIBLE" for state in states.values()):
        return "REJECT", None
    return "ABSTAIN", None


def run_policy(runtime: dict[str, Any], method: str, budget: int | None, seed: int = 101) -> dict[str, Any]:
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError("Unknown Stage 2 policy: " + str(method))
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be a nonnegative integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    base = {"method": method, "budget": budget, "seed": seed,
            "candidate_catalog_materialized": False, "audit_goal_states_materialized": True,
            "runtime_candidate_representation": "source_target_families_with_explicit_target_ids",
            "cost_semantics": "one source check or previously unexpanded configuration-node successor query per unit; collision/transition counters cover those paid queries only; cached-graph bookkeeping, witness routing and final witness validation are included in wall time; graph_traversals counts witness-route node visits only",
            "policy_kind": "fixed_development_configuration_search_heuristic; no model or novelty claim"}
    parse_started = time.perf_counter()
    try:
        families = parse_families(runtime)
    except ValueError as exc:
        return {**base, "supported": False, "reason": "UNSUPPORTED: " + str(exc), "decision": "ABSTAIN",
                "selected_goal": None, "selected_candidate": None, "witness": [], "goal_states": {},
                "trace": [], "source_certificates": [], "closed_components": [], "candidate_count": 0,
                "cost": {"source_checks": 0, "empty_node_expansions": 0, "loaded_node_expansions": 0,
                         "total_units": 0, "collision_cell_checks": 0, "transition_checks": 0, "cache_hits": 0, "graph_traversals": 0},
                "parse_seconds": time.perf_counter() - parse_started, "wall_seconds": time.perf_counter() - started}
    parse_seconds = time.perf_counter() - parse_started
    engine = _Engine(runtime, families)
    source_turn, stop_reason = True, "budget_exhausted"
    while True:
        states, reasons = engine.states()
        decision, _ = _decision(states)
        if not states:
            stop_reason = "EMPTY_SEMANTICS"
            break
        if method != "FULL_SHARED_CONFIG" and decision != "ABSTAIN":
            stop_reason = "decision_certified"
            break
        if budget is not None and sum(engine.cost.values()) >= budget:
            break
        counts = engine.unknown_counts(states)
        pending = [source for source in engine.sources if source not in engine.certificates
                   and (method in {"FULL_SHARED_CONFIG", "CHEAP_FIRST_SHARED_CONFIG"} or counts[source] > 0)]
        source_query, job = None, None
        if method in {"FULL_SHARED_CONFIG", "CHEAP_FIRST_SHARED_CONFIG"}:
            if pending:
                source_query = pending[0]
            elif method == "FULL_SHARED_CONFIG" and not engine.empty.closed(engine.start):
                job = [engine.empty, engine.empty.frontier(engine.start), 0]
            else:
                jobs = engine.search_jobs(counts, full=method == "FULL_SHARED_CONFIG")
                job = jobs[0] if jobs else None
        elif method == "ROUND_ROBIN_SHARED_CONFIG":
            jobs = engine.search_jobs(counts, include_unchecked=True)
            if pending and (source_turn or not jobs):
                source_query = pending[0]
            elif jobs:
                job = jobs[0]
        else:
            prioritized = method == "FAMILY_PRIORITY_SHARED_CONFIG"
            rank = (lambda source: (-counts[source], source)) if prioritized else (lambda source: source)
            ready = sorted((source for source in pending if engine.source_reached(source)), key=rank)
            if ready:
                source_query = ready[0]
            else:
                jobs = engine.search_jobs(counts)
                if jobs:
                    job = min(jobs, key=lambda item: (-item[2], item[0].scope, item[1])) if prioritized else jobs[0]
                elif pending:
                    source_query = min(pending, key=rank)
        if source_query is not None:
            engine.source_check(source_query)
            source_turn = False
        elif job is not None:
            job[0].expand(job[1])
            source_turn = True
        else:
            if decision == "ABSTAIN":
                raise RuntimeError("A nonempty finite task has no evidence query but remains unresolved")
            stop_reason = "all_required_shared_components_complete"
            break
    states, reasons = engine.states()
    decision, selected_goal = _decision(states)
    selected = None
    if selected_goal is not None:
        selected = next({"action": "place", "source_id": source, "target_id": target}
                        for source in engine.sources for target in engine.families[source]
                        if goal_id({"action": "place", "source_id": source, "target_id": target}) == selected_goal)
    witness = engine.witness(selected, reasons[selected_goal]) if selected is not None else []
    if selected is not None and not validate_witness(runtime, selected, witness):
        raise RuntimeError("Configuration-graph evidence produced an invalid execution witness")
    graphs = [engine.empty] + [engine.loaded[key] for key in sorted(engine.loaded)]
    closed = [certificate for graph in graphs for certificate in graph.closed_certificates()]
    return {**base, "supported": True, "reason": stop_reason, "decision": decision,
            "selected_goal": selected_goal, "selected_candidate": selected, "witness": witness,
            "goal_states": states, "goal_evidence": reasons, "families": copy.deepcopy(families),
            "candidate_count": len(states), "family_count": len(engine.families),
            "trace": engine.trace, "source_certificates": list(engine.certificates.values()),
            "world_sha256": engine.world_hash, "closed_components": closed,
            "graph_statistics": {"footprint_graphs": len(engine.loaded), "known_nodes": sum(len(g.parents) for g in graphs),
                                 "expanded_nodes": sum(len(g.expanded) for g in graphs), "closed_components": len(closed),
                                 "auxiliary_count_scope": "collision_cell_checks and transition_checks cover paid source/successor queries, excluding final validate_witness replay; cache_hits counts tracked known-node/connection/route lookups; graph_traversals counts witness-route node visits only, not all graph bookkeeping"},
            "cost": {**engine.cost, "total_units": sum(engine.cost.values()), **engine.aux},
            "parse_seconds": parse_seconds, "wall_seconds": time.perf_counter() - started}
