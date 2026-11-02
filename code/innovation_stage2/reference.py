"""Independent augmented-state reference search for the discrete Stage 2 world.

Occupancy and transitions are implemented here, independently of the online
configuration/motion helpers. Only public validation and the hidden semantic AST
interpreter are imported. Place is a terminal action, so no object relocation
or second job occurs inside a witness.
"""

from __future__ import annotations

from collections import deque

from .domain import evaluate_ast, goal_id, validate_runtime


def reference_oracle(runtime, ast, max_nodes=100000):
    validate_runtime(runtime)
    if type(max_nodes) is not int or max_nodes < 0:
        raise ValueError("max_nodes must be a nonnegative integer")
    candidates = evaluate_ast(ast, runtime)
    goal_keys = {goal_id(candidate) for candidate in candidates}
    if not goal_keys:
        return {"complete": True, "semantic_candidates": [], "feasible_goals": [], "witnesses": {},
                "expanded_nodes": 0, "label": "EMPTY_SEMANTICS"}
    grid = runtime["grid"]
    obstacles = {tuple(cell) for cell in grid["blocked"]}
    objects = {obj["id"]: obj for obj in runtime["objects"]}

    def occupied(footprint, x, y, orientation):
        # Independent occupancy implementation; no online helper invocation.
        extent_x = footprint[orientation] // 2
        extent_y = footprint[1 - orientation] // 2
        cells = set()
        for offset_y in range(-extent_y, extent_y + 1):
            for offset_x in range(-extent_x, extent_x + 1):
                cells.add((x + offset_x, y + offset_y))
        return cells

    def clear(cells):
        for px, py in cells:
            if px < 0 or py < 0 or px >= grid["width"] or py >= grid["height"] or (px, py) in obstacles:
                return False
        return True

    start = (*runtime["robot"]["start"], None, 0)
    queue = deque([start])
    previous = {start: None}
    witnesses = {}
    expanded = 0

    def trace(state):
        result = []
        while previous[state] is not None:
            state, action = previous[state]
            result.append(action)
        result.reverse()
        return result

    while queue and expanded < max_nodes:
        state = queue.popleft()
        expanded += 1
        x, y, held, orientation = state
        next_states = []
        if held is None:
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if clear({(x, y), (nx, ny)}):
                    next_states.append(((nx, ny, None, 0), {"action": "move", "to": [nx, ny]}))
            for source in runtime["objects"]:
                if (source["cell"] == [x, y] and source["movable"] and source["mass"] <= runtime["robot"]["payload"]
                        and clear(occupied(source["footprint"], x, y, source["orientation"]))):
                    next_states.append(((x, y, source["id"], source["orientation"]), {"action": "pick", "source_id": source["id"]}))
        else:
            source = objects[held]
            footprint = source["footprint"]
            for target in runtime["targets"]:
                key = f"place:{held}->{target['id']}"
                if (key in goal_keys and key not in witnesses and target["cell"] == [x, y]
                        and orientation in target["allowed_orientations"]):
                    witnesses[key] = trace(state) + [{"action": "place", "target_id": target["id"]}]
            occupied_now = occupied(footprint, x, y, orientation)
            for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if clear(occupied_now | occupied(footprint, nx, ny, orientation)):
                    next_states.append(((nx, ny, held, orientation), {"action": "move", "to": [nx, ny]}))
            radius = max(footprint) // 2
            swept = {(px, py) for px in range(x - radius, x + radius + 1)
                     for py in range(y - radius, y + radius + 1)}
            if clear(swept):
                next_states.append(((x, y, held, 1 - orientation), {"action": "rotate", "orientation": 1 - orientation}))
        for following, action in next_states:
            if following not in previous:
                previous[following] = (state, action)
                queue.append(following)
    complete = not queue
    feasible = sorted(witnesses)
    label = None if not complete else "INFEASIBLE" if not feasible else "SUFFICIENT" if len(feasible) == 1 else "INSUFFICIENT"
    return {"complete": complete, "semantic_candidates": candidates, "feasible_goals": feasible,
            "witnesses": witnesses, "expanded_nodes": expanded, "label": label}
