"""Independent reference, closure check, and direct witness replay.

This module never imports the evaluated search engine. A checked rooted tree
establishes reachability and checking all outgoing moves establishes closure.
"""
from collections import deque

from innovation_stage3.storage import stable
from .cases import member, member_at


class Geometry:
    def __init__(self, world, config, meter=None):
        self.world, self.config = world, config
        self.obstacles = frozenset(tuple(p) for p in world["blocked"])
        self.cache = {}
        self.cell_queries = 0
        self.meter = meter

    def clear(self, pose, swept=False):
        if len(pose) != 3 or any(type(v) is not int for v in pose) or pose[2] not in (0, 1):
            return False
        key = tuple(pose), swept
        if key not in self.cache:
            a, b = self.config["footprint"]
            if swept:
                a = b = max(a, b)
            elif pose[2]:
                a, b = b, a
            x, y, _ = pose
            valid = True
            for cy in range(y - b // 2, y + b // 2 + 1):
                for cx in range(x - a // 2, x + a // 2 + 1):
                    self.cell_queries += 1
                    if self.meter:
                        self.meter()
                    if cx < 0 or cy < 0 or cx >= self.world["width"] or cy >= self.world["height"] or (cx, cy) in self.obstacles:
                        valid = False
            self.cache[key] = valid
        return self.cache[key]

    def adjacent(self, pose):
        x, y, o = pose
        for q in ((x, y - 1, o), (x - 1, y, o), (x, y + 1, o), (x + 1, y, o)):
            if self.clear(pose) and self.clear(q):
                yield q
        if self.world["rotation"] == "square_sweep" and self.clear(pose, True):
            q = x, y, 1 - o
            if self.clear(q):
                yield q


def reference(public):
    """BFS reference independent of all policy heuristics and candidate ordering."""
    records = []
    for c in public["world"]["configurations"]:
        geom = Geometry(public["world"], c)
        source = tuple(c["start"])
        parents = {source: None} if geom.clear(source) else {}
        queue = deque(parents)
        while queue:
            node = queue.popleft()
            for nxt in geom.adjacent(node):
                if nxt not in parents:
                    parents[nxt] = node
                    queue.append(nxt)
        records.append({"configuration_id": c["id"], "world_sha256": stable(public["world"]),
                        "parents": [[list(n), list(p) if p is not None else None] for n, p in sorted(parents.items())]})
    feasible = []
    by_id = {r["configuration_id"]: {tuple(n[:2]) for n, _ in r["parents"]} for r in records}
    for f in public["families"]:
        for i in range(f["count"]):
            goal = member(f, i)
            if tuple(goal["target"]) in by_id[f["configuration_id"]]:
                feasible.append(goal["goal_id"])
    return {"public_sha256": stable(public), "closures": records, "feasible_goal_ids": sorted(feasible),
            "expected_decision": "REJECT" if not feasible else "EXECUTE" if len(feasible) == 1 else "CLARIFY"}


def check_closure(world, config, proof, check_budget=lambda: None, meter=None):
    """Check a complete concrete proof; reject omitted or disconnected states."""
    geom = Geometry(world, config, meter)
    try:
        if set(proof) != {"configuration_id", "world_sha256", "parents"} or proof["world_sha256"] != stable(world) or proof["configuration_id"] != config["id"]:
            return {"valid": False, "cell_queries": geom.cell_queries}
        parents = {tuple(n): tuple(p) if p is not None else None for n, p in proof["parents"]}
        if len(parents) != len(proof["parents"]):
            return {"valid": False, "cell_queries": geom.cell_queries}
        source = tuple(config["start"])
        if not geom.clear(source):
            return {"valid": not parents, "cell_queries": geom.cell_queries}
        if source not in parents or parents[source] is not None:
            return {"valid": False, "cell_queries": geom.cell_queries}
        children = {n: [] for n in parents}
        for node, parent in parents.items():
            check_budget()
            if not geom.clear(node):
                return {"valid": False, "cell_queries": geom.cell_queries}
            if node != source:
                if parent not in parents or node not in set(geom.adjacent(parent)):
                    return {"valid": False, "cell_queries": geom.cell_queries}
                children[parent].append(node)
            if any(q not in parents for q in geom.adjacent(node)):
                return {"valid": False, "cell_queries": geom.cell_queries}
        visited, todo = set(), [source]
        while todo:
            node = todo.pop()
            if node in visited:
                return {"valid": False, "cell_queries": geom.cell_queries}
            visited.add(node)
            todo.extend(children[node])
        return {"valid": visited == set(parents), "cell_queries": geom.cell_queries}
    except (KeyError, TypeError, ValueError, IndexError):
        return {"valid": False, "cell_queries": geom.cell_queries}


def check_witness(public, goal, path):
    config = next((c for c in public["world"]["configurations"] if c["id"] == goal["configuration_id"]), None)
    if config is None or not path or path[0] != config["start"] or path[-1][:2] != goal["target"]:
        return {"valid": False, "cell_queries": 0}
    # Membership is checked using the public family declaration, not an ID alone.
    matches = [f for f in public["families"] if f["configuration_id"] == config["id"] and member_at(f, goal["target"]) is not None]
    if not any(member(f, member_at(f, goal["target"])) == goal for f in matches):
        return {"valid": False, "cell_queries": 0}
    geom = Geometry(public["world"], config)
    good = geom.clear(tuple(path[0])) and all(tuple(b) in set(geom.adjacent(tuple(a))) for a, b in zip(path, path[1:]))
    return {"valid": good, "cell_queries": geom.cell_queries}


def audit_reference(public, value):
    if value["public_sha256"] != stable(public):
        raise ValueError("Reference public input changed")
    configs = {c["id"]: c for c in public["world"]["configurations"]}
    closures = value["closures"]
    if len(closures) != len(configs) or {p["configuration_id"] for p in closures} != set(configs):
        raise ValueError("Reference scopes missing or duplicated")
    if not all(check_closure(public["world"], configs[p["configuration_id"]], p)["valid"] for p in closures):
        raise ValueError("Reference closure/tree failed independent check")
    nodes = {p["configuration_id"]: {tuple(n[:2]) for n, _ in p["parents"]} for p in closures}
    expected = sorted(member(f, i)["goal_id"] for f in public["families"] for i in range(f["count"])
                      if tuple(member(f, i)["target"]) in nodes[f["configuration_id"]])
    decision = "REJECT" if not expected else "EXECUTE" if len(expected) == 1 else "CLARIFY"
    if expected != value["feasible_goal_ids"] or decision != value["expected_decision"]:
        raise ValueError("Reference candidate mapping changed")
    return True
