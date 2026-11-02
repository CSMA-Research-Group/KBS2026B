"""Public controlled semantics and footprint-dependent discrete motion rules.

Objects and targets are floor markers. Empty-handed motion is a point robot;
carrying uses the selected object's centred odd rectangular footprint. A move
checks the union of initial/final footprints. A quarter-turn checks the square
of side max(width,height), a declared discrete square-sweep rule. This is not
a continuous swept-volume bound, dynamics model, or collision guarantee.
"""

from __future__ import annotations

import math
import re
from typing import Any

from innovation_stage1.domain import _parse_predicate, _validate_ast
from innovation_stage1.domain import render_instruction as _render_base


SCHEMA_VERSION = "innovation-stage2-footprint-runtime-v1"
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _require(value: bool, message: str) -> None:
    if not value:
        raise ValueError(message)


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def validate_runtime(runtime: dict[str, Any]) -> None:
    """Validate public fields, retaining initially invalid cargo footprints."""
    _require(isinstance(runtime, dict) and set(runtime) == {"instruction", "grid", "robot", "objects", "targets", "task_contract"}, "Invalid public runtime fields")
    _require(isinstance(runtime["instruction"], str) and bool(runtime["instruction"]), "instruction must contain text")
    _require(runtime["task_contract"] == {"mode": "referential"}, "Unsupported task contract")
    grid = runtime["grid"]
    _require(isinstance(grid, dict) and set(grid) == {"width", "height", "blocked"}, "Invalid grid fields")
    _require(all(type(grid[key]) is int and grid[key] > 0 for key in ("width", "height")), "Grid dimensions must be positive integers")

    def cell(value: Any) -> tuple[int, int]:
        _require(isinstance(value, list) and len(value) == 2 and all(type(v) is int for v in value), "Cell requires integer [x,y]")
        _require(0 <= value[0] < grid["width"] and 0 <= value[1] < grid["height"], "Cell outside grid")
        return tuple(value)

    _require(isinstance(grid["blocked"], list), "blocked must be a list")
    obstacles = [cell(value) for value in grid["blocked"]]
    _require(len(obstacles) == len(set(obstacles)), "Duplicate blocked cells")
    blocked = set(obstacles)
    robot = runtime["robot"]
    _require(isinstance(robot, dict) and set(robot) == {"start", "payload"}, "Invalid robot fields")
    _require(cell(robot["start"]) not in blocked and _number(robot["payload"]), "Invalid robot start/payload")
    seen = set()
    for collection in ("objects", "targets"):
        _require(isinstance(runtime[collection], list), "Entity collections must be lists")
        fields = {"id", "cell", "color", "shape"} | ({"mass", "movable", "footprint", "orientation"} if collection == "objects" else {"allowed_orientations", "accepts_colors"})
        for entity in runtime[collection]:
            _require(isinstance(entity, dict) and set(entity) == fields, "Invalid entity fields")
            _require(_identifier(entity["id"]) and entity["id"] not in seen, "Entity IDs must be unique identifiers")
            seen.add(entity["id"])
            _require(cell(entity["cell"]) not in blocked, "Entity anchor is blocked")
            _require(_identifier(entity["color"]) and _identifier(entity["shape"]), "Invalid entity attributes")
            if collection == "objects":
                footprint = entity["footprint"]
                _require(isinstance(footprint, list) and len(footprint) == 2 and all(type(v) is int and v > 0 and v % 2 == 1 for v in footprint), "Footprint dimensions must be positive odd integers")
                _require(type(entity["orientation"]) is int and entity["orientation"] in (0, 1), "Invalid object orientation")
                _require(_number(entity["mass"]) and type(entity["movable"]) is bool, "Invalid source conditions")
            else:
                allowed = entity["allowed_orientations"]
                _require(isinstance(allowed, list) and len(allowed) > 0 and all(type(v) is int and v in (0, 1) for v in allowed) and len(set(allowed)) == len(allowed), "Invalid allowed target orientations")
                accepts = entity["accepts_colors"]
                _require(isinstance(accepts, list) and all(_identifier(v) for v in accepts) and len(set(accepts)) == len(accepts), "Invalid accepted-color list")


def _validate_program(ast: dict[str, Any]) -> None:
    _require(isinstance(ast, dict) and set(ast) == {"action", "source", "target", "pairing"}, "AST requires action/source/target/pairing")
    _require(ast["action"] == "place" and ast["pairing"] in ("any", "compatible"), "Unsupported AST action/pairing")
    _validate_ast({key: ast[key] for key in ("action", "source", "target")})


def render_instruction(ast: dict[str, Any]) -> str:
    _validate_program(ast)
    return _render_base({key: ast[key] for key in ("action", "source", "target")}) + f" Pairing: {ast['pairing']}."


def goal_id(candidate: dict[str, Any]) -> str:
    _require(isinstance(candidate, dict) and set(candidate) == {"action", "source_id", "target_id"}, "Invalid candidate fields")
    _require(candidate["action"] == "place" and _identifier(candidate["source_id"]) and _identifier(candidate["target_id"]), "Invalid place candidate")
    return f"place:{candidate['source_id']}->{candidate['target_id']}"


def parse_families(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Public semantic groups; no candidate dict cross-product or physical filter."""
    validate_runtime(runtime)
    match = re.fullmatch(r"Place an object satisfying (.+) into a target satisfying (.+)\. Pairing: (any|compatible)\.", runtime["instruction"])
    _require(match is not None, "Instruction outside the whole-text Stage 2 grammar")
    source_predicate = _parse_predicate(match.group(1))
    target_predicate = _parse_predicate(match.group(2))
    pairing = match.group(3)
    targets = [target for target in runtime["targets"] if target_predicate(target)]
    families = []
    for source in sorted(runtime["objects"], key=lambda value: value["id"]):
        if not source_predicate(source):
            continue
        target_ids = sorted(target["id"] for target in targets if pairing == "any" or source["color"] in target["accepts_colors"])
        if target_ids:
            families.append({"source_id": source["id"], "target_ids": target_ids})
    return families


def evaluate_ast(ast: dict[str, Any], runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Benchmark denotation through an independent recursive AST interpreter."""
    _validate_program(ast)
    validate_runtime(runtime)

    def evaluate(node, entity):
        if node["op"] == "eq":
            return entity[node["field"]] == node["value"]
        results = [evaluate(child, entity) for child in node["args"]]
        return all(results) if node["op"] == "and" else any(results)

    candidates = []
    for source in runtime["objects"]:
        if evaluate(ast["source"], source):
            for target in runtime["targets"]:
                if evaluate(ast["target"], target) and (ast["pairing"] == "any" or source["color"] in target["accepts_colors"]):
                    candidates.append({"action": "place", "source_id": source["id"], "target_id": target["id"]})
    return sorted(candidates, key=goal_id)


def footprint_cells(footprint, node) -> set[tuple[int, int]]:
    """Centred occupancy cells for an odd rectangle and orientation 0 or 1."""
    _require(isinstance(footprint, (list, tuple)) and len(footprint) == 2 and all(type(v) is int and v > 0 and v % 2 for v in footprint), "Invalid footprint")
    _require(isinstance(node, (list, tuple)) and len(node) == 3 and all(type(v) is int for v in node) and node[2] in (0, 1), "Invalid configuration node")
    x, y, orientation = node
    width, height = footprint if orientation == 0 else reversed(footprint)
    return {(cx, cy) for cx in range(x - width // 2, x + width // 2 + 1)
            for cy in range(y - height // 2, y + height // 2 + 1)}


def _clear(runtime, cells, stats=None) -> bool:
    blocked = {tuple(cell) for cell in runtime["grid"]["blocked"]}
    valid = True
    # Count every visited cell explicitly; do not infer hardware cost from it.
    for x, y in sorted(cells):
        if stats is not None:
            stats["collision_cell_checks"] = stats.get("collision_cell_checks", 0) + 1
        if not (0 <= x < runtime["grid"]["width"] and 0 <= y < runtime["grid"]["height"]) or (x, y) in blocked:
            valid = False
    return valid


def configuration_valid(runtime, footprint, node, stats=None) -> bool:
    return _clear(runtime, footprint_cells(footprint, node), stats)


def motion_neighbors(runtime, footprint, node, stats=None) -> list[tuple[tuple[int, int, int], dict[str, Any]]]:
    """Legal carrying successors, counting proposed transitions and collision cells."""
    current_cells = footprint_cells(footprint, node)
    if not _clear(runtime, current_cells, stats):
        return []
    x, y, orientation = node
    neighbors = []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        following = (x + dx, y + dy, orientation)
        if stats is not None:
            stats["transition_checks"] = stats.get("transition_checks", 0) + 1
        if _clear(runtime, current_cells | footprint_cells(footprint, following), stats):
            neighbors.append((following, {"action": "move", "to": [following[0], following[1]]}))
    side = max(footprint)
    if stats is not None:
        stats["transition_checks"] = stats.get("transition_checks", 0) + 1
    if _clear(runtime, footprint_cells((side, side), (x, y, 0)), stats):
        neighbors.append(((x, y, 1 - orientation), {"action": "rotate", "orientation": 1 - orientation}))
    return neighbors


def validate_witness(runtime, candidate, actions) -> bool:
    """Direct physical replay; semantic membership is checked by the evaluator."""
    try:
        validate_runtime(runtime)
        goal_id(candidate)
        if not isinstance(actions, list):
            return False
        objects = {obj["id"]: obj for obj in runtime["objects"]}
        targets = {target["id"]: target for target in runtime["targets"]}
        if candidate["source_id"] not in objects or candidate["target_id"] not in targets:
            return False
        x, y = runtime["robot"]["start"]
        held = None
        orientation = 0
        picked = False
        placed = None
        for action in actions:
            if not isinstance(action, dict) or placed is not None:
                return False
            operation = action.get("action")
            if operation == "move" and set(action) == {"action", "to"}:
                to = action["to"]
                if not isinstance(to, list) or len(to) != 2 or any(type(v) is not int for v in to) or abs(to[0] - x) + abs(to[1] - y) != 1:
                    return False
                if held is None:
                    cells = {(x, y), tuple(to)}
                else:
                    fp = objects[held]["footprint"]
                    cells = footprint_cells(fp, (x, y, orientation)) | footprint_cells(fp, (*to, orientation))
                if not _clear(runtime, cells):
                    return False
                x, y = to
            elif operation == "pick" and set(action) == {"action", "source_id"}:
                source = objects.get(action["source_id"])
                if source is None or picked or source["id"] != candidate["source_id"] or source["cell"] != [x, y] or not source["movable"] or source["mass"] > runtime["robot"]["payload"]:
                    return False
                orientation = source["orientation"]
                if not _clear(runtime, footprint_cells(source["footprint"], (x, y, orientation))):
                    return False
                held, picked = source["id"], True
            elif operation == "rotate" and set(action) == {"action", "orientation"}:
                new_orientation = action["orientation"]
                if held is None or type(new_orientation) is not int or new_orientation != 1 - orientation:
                    return False
                side = max(objects[held]["footprint"])
                if not _clear(runtime, footprint_cells((side, side), (x, y, 0))):
                    return False
                orientation = new_orientation
            elif operation == "place" and set(action) == {"action", "target_id"}:
                target = targets.get(action["target_id"])
                if target is None or held != candidate["source_id"] or target["id"] != candidate["target_id"] or target["cell"] != [x, y] or orientation not in target["allowed_orientations"]:
                    return False
                held, placed = None, target["id"]
            else:
                return False
        return picked and held is None and placed == candidate["target_id"]
    except (KeyError, TypeError, ValueError):
        return False
