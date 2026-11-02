"""Orientation-free inner-square relaxation and independently checked closure.

Every concrete odd rectangle contains a centred square whose side is the
minimum dimension in the entire declared configuration family. Translation
projects to an edge in this relaxed graph; rotation projects to a self-loop.
A closed set containing every source therefore overapproximates every loaded
reachable component. Only exclusion from the closed set is negative evidence.
"""
from collections import deque

from innovation_stage3.storage import stable

SCHEMA = "e3a-loaded-motion-v1"
PROOF = "e3a-inner-square-closure-v1"


def validate_world(world):
    if set(world) != {"schema", "width", "height", "blocked", "configurations", "rotation"} or world["schema"] != SCHEMA:
        raise ValueError("Unsupported loaded-motion world schema")
    if any(type(world[k]) is not int or world[k] < 1 for k in ("width", "height")):
        raise ValueError("Invalid dimensions")
    if world["rotation"] not in ("square_sweep", "disabled"):
        raise ValueError("Undeclared rotation rule")
    def cell(value):
        return (isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value)
                and 0 <= value[0] < world["width"] and 0 <= value[1] < world["height"])
    if not isinstance(world["blocked"], list) or not all(cell(c) for c in world["blocked"]):
        raise ValueError("Invalid obstacle cells")
    if len({tuple(c) for c in world["blocked"]}) != len(world["blocked"]):
        raise ValueError("Repeated obstacle")
    if not isinstance(world["configurations"], list) or not world["configurations"]:
        raise ValueError("Empty configuration scope")
    ids = []
    for config in world["configurations"]:
        if set(config) != {"id", "footprint", "start"} or not isinstance(config["id"], str):
            raise ValueError("Invalid source scope")
        fp, start = config["footprint"], config["start"]
        if not isinstance(fp, list) or len(fp) != 2 or not all(type(v) is int and v > 0 and v % 2 for v in fp):
            raise ValueError("Footprints must be positive odd rectangles")
        if not isinstance(start, list) or len(start) != 3 or not cell(start[:2]) or type(start[2]) is not int or start[2] not in (0, 1):
            raise ValueError("Invalid initial loaded pose")
        ids.append(config["id"])
    if len(set(ids)) != len(ids):
        raise ValueError("Repeated configuration ID")


def abstract_component(world, max_expansions=None):
    validate_world(world)
    if max_expansions is not None and (type(max_expansions) is not int or max_expansions < 0):
        raise ValueError("Invalid expansion budget")
    side = min(min(c["footprint"]) for c in world["configurations"])
    half = side // 2
    blocked = {tuple(c) for c in world["blocked"]}
    cost = {"expanded_nodes": 0, "cell_queries": 0, "centres_tested": 0}
    cache = {}
    def free(position):
        if position not in cache:
            x, y = position
            cost["centres_tested"] += 1
            good = True
            for dx in range(-half, half+1):
                for dy in range(-half, half+1):
                    cost["cell_queries"] += 1
                    cx, cy = x+dx, y+dy
                    if not (0 <= cx < world["width"] and 0 <= cy < world["height"]) or (cx,cy) in blocked:
                        good = False
            cache[position] = good
        return cache[position]
    sources = sorted({tuple(c["start"][:2]) for c in world["configurations"]})
    if not all(free(p) for p in sources):
        return {"status": "UNKNOWN", "reason": "invalid_relaxed_start", "cost": cost}
    seen, queue = set(sources), deque(sources)
    while queue:
        if max_expansions is not None and cost["expanded_nodes"] >= max_expansions:
            return {"status": "UNKNOWN", "reason": "budget_exhausted", "cost": cost}
        x, y = queue.popleft()
        cost["expanded_nodes"] += 1
        for position in ((x+1,y),(x,y+1),(x-1,y),(x,y-1)):
            if position not in seen and free(position):
                seen.add(position)
                queue.append(position)
    return {"status": "COMPLETE", "cost": cost, "certificate": {
        "schema": PROOF, "world_sha256": stable(world), "side": side,
        "scope_ids": sorted(c["id"] for c in world["configurations"]),
        "nodes": [list(p) for p in sorted(seen)]}}


def verify_certificate(world, proof):
    """Check inclusion and closure; do not trust a saved reachable/not-reachable label.

    This checker uses direct rectangle bounds/obstacle membership, not the
    builder's cache, graph traversal, costs, or claimed negative targets.
    A nonminimal closed superset is valid but can only weaken exclusions.
    """
    cost = {"cell_queries": 0, "centres_tested": 0}
    def reject(reason):
        return {"valid": False, "reason": reason, "cost": cost}
    try:
        validate_world(world)
        if not isinstance(proof, dict) or set(proof) != {"schema", "world_sha256", "side", "scope_ids", "nodes"}:
            return reject("proof_schema")
        if proof["schema"] != PROOF or proof["world_sha256"] != stable(world):
            return reject("world_binding")
        required = min(v for c in world["configurations"] for v in c["footprint"])
        if type(proof["side"]) is not int or proof["side"] != required:
            return reject("configuration_lower_bound")
        if proof["scope_ids"] != sorted(c["id"] for c in world["configurations"]):
            return reject("source_scope")
        if not isinstance(proof["nodes"], list) or any(not isinstance(n,list) or len(n)!=2 or any(type(v) is not int for v in n) for n in proof["nodes"]):
            return reject("node_schema")
        nodes = {tuple(n) for n in proof["nodes"]}
        if len(nodes) != len(proof["nodes"]):
            return reject("duplicate_node")
        obstacles = {tuple(b) for b in world["blocked"]}
        radius = required//2
        def clear(x,y):
            cost["centres_tested"] += 1
            valid = True
            for cy in range(y-radius,y+radius+1):
                for cx in range(x-radius,x+radius+1):
                    cost["cell_queries"] += 1
                    if cx < 0 or cy < 0 or cx >= world["width"] or cy >= world["height"] or (cx,cy) in obstacles:
                        valid = False
            return valid
        if any(tuple(c["start"][:2]) not in nodes for c in world["configurations"]):
            return reject("missing_source")
        for x,y in nodes:
            if not clear(x,y):
                return reject("invalid_node")
            for dx,dy in ((0,-1),(-1,0),(0,1),(1,0)):
                if clear(x+dx,y+dy) and (x+dx,y+dy) not in nodes:
                    return reject("open_boundary")
        return {"valid": True, "reason": "closed_overapproximation", "cost": cost}
    except (ValueError, TypeError, KeyError, IndexError):
        return reject("malformed_input")
