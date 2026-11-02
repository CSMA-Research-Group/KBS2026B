"""Controlled language and an independent finite-state planning reference.

The public world is a point robot on a four-neighbour grid with fixed obstacles.
Objects/targets are labelled locations, not additional collision obstacles. A
job picks one movable object within the payload limit, then optionally carries
it to one target and terminates. No other object can be moved. These semantics
are a discrete abstraction, not a robot dynamics or manipulation simulator.

The hidden AST interpreter and public text parser use separate implementations.
The reference explores (robot position, held object) states and persists action
witnesses. It does not call a runtime connected-component or budgeted checker.
"""

from __future__ import annotations

from collections import deque
import math
import random
import re
from typing import Any, Callable


SCHEMA_VERSION = "innovation-stage1-grid-v1"
GRAMMAR_VERSION = "controlled-composition-v1"
PATTERNS = ("uniform", "mostly_ineligible", "one_eligible", "all_ineligible", "split_wall")
CANDIDATE_SCALES = (4, 8, 16, 32)
GRAMMARS = ("atom", "and", "or")
_IDENTIFIER = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _identifier(value: Any) -> bool:
    return isinstance(value, str) and bool(_IDENTIFIER.fullmatch(value))


def validate_runtime(runtime: dict[str, Any]) -> None:
    """Validate the entire allowed runtime payload; hidden labels are forbidden."""
    required = {"instruction", "grid", "robot", "objects", "targets", "task_contract"}
    _require(isinstance(runtime, dict) and set(runtime) == required, "Runtime must contain exactly the public schema fields")
    _require(isinstance(runtime["instruction"], str) and bool(runtime["instruction"].strip()), "instruction must be nonempty text")
    _require(runtime["task_contract"] == {"mode": "referential"}, "Only the explicit referential contract is supported")
    grid = runtime["grid"]
    _require(isinstance(grid, dict) and set(grid) == {"width", "height", "blocked"}, "Invalid grid fields")
    for key in ("width", "height"):
        _require(type(grid[key]) is int and grid[key] > 0, "Grid dimensions must be positive integers")

    def cell(value: Any) -> tuple[int, int]:
        _require(isinstance(value, list) and len(value) == 2 and all(type(c) is int for c in value), "A cell must be an integer [x,y] list")
        _require(0 <= value[0] < grid["width"] and 0 <= value[1] < grid["height"], "Cell lies outside the grid")
        return tuple(value)

    _require(isinstance(grid["blocked"], list), "blocked must be a list")
    blocked = [cell(value) for value in grid["blocked"]]
    _require(len(set(blocked)) == len(blocked), "Duplicate obstacle cells")
    blocked_set = set(blocked)
    robot = runtime["robot"]
    _require(isinstance(robot, dict) and set(robot) == {"start", "payload"}, "Invalid robot fields")
    _require(cell(robot["start"]) not in blocked_set, "Robot starts in an obstacle")
    _require(_number(robot["payload"]), "payload must be finite and nonnegative")
    identifiers: set[str] = set()
    for collection in ("objects", "targets"):
        _require(isinstance(runtime[collection], list), f"{collection} must be a list")
        fields = {"id", "cell", "color", "shape"}
        if collection == "objects":
            fields |= {"mass", "movable"}
        for item in runtime[collection]:
            _require(isinstance(item, dict) and set(item) == fields, f"Invalid {collection} item fields")
            _require(_identifier(item["id"]) and item["id"] not in identifiers, "Entity IDs must be unique identifiers")
            identifiers.add(item["id"])
            _require(cell(item["cell"]) not in blocked_set, "An entity lies in an obstacle")
            _require(_identifier(item["color"]) and _identifier(item["shape"]), "Attributes must be identifier tokens")
            if collection == "objects":
                _require(_number(item["mass"]), "mass must be finite and nonnegative")
                _require(type(item["movable"]) is bool, "movable must be boolean")


def goal_id(candidate: dict[str, Any]) -> str:
    _require(isinstance(candidate, dict) and set(candidate) == {"action", "source_id", "target_id"}, "Invalid candidate fields")
    _require(_identifier(candidate["source_id"]), "Invalid source ID")
    if candidate["action"] == "pick":
        _require(candidate["target_id"] is None, "A pick candidate cannot contain a target")
        return f"pick:{candidate['source_id']}"
    _require(candidate["action"] == "place" and _identifier(candidate["target_id"]), "Invalid place candidate")
    return f"place:{candidate['source_id']}->{candidate['target_id']}"


def _validate_ast(ast: dict[str, Any]) -> None:
    _require(isinstance(ast, dict) and set(ast) == {"action", "source", "target"}, "AST requires action/source/target")
    _require(ast["action"] in {"pick", "place"}, "Unsupported AST action")

    def predicate(node: Any, depth: int = 0) -> None:
        _require(depth <= 64 and isinstance(node, dict), "Invalid or excessively nested predicate")
        if node.get("op") == "eq":
            _require(set(node) == {"op", "field", "value"}, "Invalid atom fields")
            _require(node["field"] in {"color", "shape"} and _identifier(node["value"]) and node["value"] not in {"AND", "OR"}, "Unsupported atom")
        else:
            _require(set(node) == {"op", "args"} and node["op"] in {"and", "or"}, "Unsupported predicate operator")
            _require(isinstance(node["args"], list) and len(node["args"]) >= 2, "A conjunction/disjunction needs at least two arguments")
            for child in node["args"]:
                predicate(child, depth + 1)

    predicate(ast["source"])
    if ast["action"] == "pick":
        _require(ast["target"] is None, "A pick AST must have target=None")
    else:
        predicate(ast["target"])


def render_instruction(ast: dict[str, Any]) -> str:
    """Compile only the declared controlled grammar; no free-form paraphrasing."""
    _validate_ast(ast)

    def render(node: dict[str, Any]) -> str:
        if node["op"] == "eq":
            return f"{node['field']}={node['value']}"
        return "(" + f" {node['op'].upper()} ".join(render(child) for child in node["args"]) + ")"

    source = render(ast["source"])
    if ast["action"] == "pick":
        return f"Pick an object satisfying {source}."
    return f"Place an object satisfying {source} into a target satisfying {render(ast['target'])}."


def evaluate_ast(ast: dict[str, Any], runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Benchmark-only denotation of all legal goals, independent of feasibility."""
    _validate_ast(ast)
    validate_runtime(runtime)

    def interpret(node: dict[str, Any], entity: dict[str, Any]) -> bool:
        if node["op"] == "eq":
            return entity[node["field"]] == node["value"]
        values = [interpret(child, entity) for child in node["args"]]
        if node["op"] == "and":
            return all(values)
        return any(values)

    candidates = []
    for source in runtime["objects"]:
        if not interpret(ast["source"], source):
            continue
        if ast["action"] == "pick":
            candidates.append({"action": "pick", "source_id": source["id"], "target_id": None})
        else:
            for target in runtime["targets"]:
                if interpret(ast["target"], target):
                    candidates.append({"action": "place", "source_id": source["id"], "target_id": target["id"]})
    return sorted(candidates, key=goal_id)


def _parse_predicate(text: str) -> Callable[[dict[str, Any]], bool]:
    """Parse public text into executable predicates, without creating/using ASTs."""
    tokens: list[str] = []
    position = 0
    token = re.compile(r"\s*([A-Za-z][A-Za-z0-9_]*|[()=])")
    while position < len(text):
        match = token.match(text, position)
        if match is None:
            if not text[position:].strip():
                break
            raise ValueError(f"Unsupported predicate token at position {position}")
        tokens.append(match.group(1))
        position = match.end()
    cursor = 0

    def expression(depth: int = 0) -> Callable[[dict[str, Any]], bool]:
        nonlocal cursor
        _require(depth <= 64 and cursor < len(tokens), "Incomplete/excessively nested predicate")
        if tokens[cursor] != "(":
            _require(cursor + 2 < len(tokens), "Incomplete atom")
            field, equals, value = tokens[cursor:cursor + 3]
            _require(field in {"color", "shape"} and equals == "=" and _identifier(value), "Unsupported atom syntax")
            _require(value not in {"AND", "OR"}, "Reserved operator used as a value")
            cursor += 3
            return lambda entity: entity[field] == value
        cursor += 1
        conditions = [expression(depth + 1)]
        _require(cursor < len(tokens) and tokens[cursor] in {"AND", "OR"}, "Parentheses must contain an explicit conjunction/disjunction")
        operator = tokens[cursor]
        while cursor < len(tokens) and tokens[cursor] == operator:
            cursor += 1
            conditions.append(expression(depth + 1))
        _require(cursor < len(tokens) and tokens[cursor] == ")", "Missing parenthesis or mixed unparenthesized operators")
        cursor += 1
        if operator == "AND":
            return lambda entity: all(condition(entity) for condition in conditions)
        return lambda entity: any(condition(entity) for condition in conditions)

    result = expression()
    _require(cursor == len(tokens), "Trailing or unparenthesized predicate tokens")
    return result


def parse_candidates(runtime: dict[str, Any]) -> list[dict[str, Any]]:
    """Recover every denoted candidate using the public, whole-text grammar.

    Syntax failure raises ValueError; it must never be treated as an empty set.
    This deterministic catalog is also available to the strongest baselines.
    """
    validate_runtime(runtime)
    instruction = runtime["instruction"]
    pick = re.fullmatch(r"Pick an object satisfying (.+)\.", instruction)
    place = re.fullmatch(r"Place an object satisfying (.+) into a target satisfying (.+)\.", instruction)
    _require(pick is not None or place is not None, "Instruction is outside the controlled whole-text grammar")
    if pick:
        source_matches = _parse_predicate(pick.group(1))
        return sorted(({"action": "pick", "source_id": item["id"], "target_id": None}
                       for item in runtime["objects"] if source_matches(item)), key=goal_id)
    assert place is not None
    source_matches = _parse_predicate(place.group(1))
    target_matches = _parse_predicate(place.group(2))
    return sorted(({"action": "place", "source_id": source["id"], "target_id": target["id"]}
                   for source in runtime["objects"] if source_matches(source)
                   for target in runtime["targets"] if target_matches(target)), key=goal_id)


def validate_witness(runtime: dict[str, Any], candidate: dict[str, Any], actions: list[dict[str, Any]]) -> bool:
    """Replay a complete witness directly against the declared transition rules."""
    try:
        validate_runtime(runtime)
        goal_id(candidate)
        if not isinstance(actions, list):
            return False
        objects = {item["id"]: item for item in runtime["objects"]}
        targets = {item["id"]: item for item in runtime["targets"]}
        if candidate["source_id"] not in objects:
            return False
        if candidate["action"] == "place" and candidate["target_id"] not in targets:
            return False
        x, y = runtime["robot"]["start"]
        held: str | None = None
        picked = False
        placed: str | None = None
        blocked = {tuple(cell) for cell in runtime["grid"]["blocked"]}
        for action in actions:
            if not isinstance(action, dict) or placed is not None:
                return False
            if action.get("action") == "move" and set(action) == {"action", "to"}:
                destination = action["to"]
                if not isinstance(destination, list) or len(destination) != 2 or any(type(value) is not int for value in destination):
                    return False
                nx, ny = destination
                if abs(nx - x) + abs(ny - y) != 1 or not (0 <= nx < runtime["grid"]["width"] and 0 <= ny < runtime["grid"]["height"]) or (nx, ny) in blocked:
                    return False
                x, y = nx, ny
            elif action.get("action") == "pick" and set(action) == {"action", "source_id"}:
                source = objects.get(action["source_id"])
                if source is None or picked or held is not None or source["cell"] != [x, y] or not source["movable"] or source["mass"] > runtime["robot"]["payload"]:
                    return False
                held, picked = source["id"], True
                if held != candidate["source_id"]:
                    return False
            elif action.get("action") == "place" and set(action) == {"action", "target_id"}:
                target = targets.get(action["target_id"])
                if target is None or target["cell"] != [x, y] or held != candidate["source_id"] or candidate["action"] != "place":
                    return False
                held, placed = None, target["id"]
            else:
                return False
        if candidate["action"] == "pick":
            return picked and held == candidate["source_id"] and placed is None
        return picked and held is None and placed == candidate["target_id"]
    except (KeyError, TypeError, ValueError):
        return False


def reference_oracle(runtime: dict[str, Any], ast: dict[str, Any], max_nodes: int = 100000) -> dict[str, Any]:
    """Exhaust an augmented-state BFS; incomplete searches never label negatives.

    ``max_nodes`` counts popped/expanded states, not edges or wall-clock time.
    Physical witnesses from a partial search remain valid, but label=None and
    complete=False prevent treating unvisited goals as impossible.
    """
    validate_runtime(runtime)
    _require(type(max_nodes) is int and max_nodes >= 0, "max_nodes must be a nonnegative integer")
    semantic_candidates = evaluate_ast(ast, runtime)
    goals = {goal_id(candidate): candidate for candidate in semantic_candidates}
    width, height = runtime["grid"]["width"], runtime["grid"]["height"]
    forbidden = {tuple(cell) for cell in runtime["grid"]["blocked"]}
    initial = (*runtime["robot"]["start"], None)
    # States have no post-place continuation: a place action terminates the job.
    frontier = deque([initial])
    parents: dict[tuple[int, int, str | None], tuple[tuple[int, int, str | None], dict[str, Any]] | None] = {initial: None}
    witnesses: dict[str, list[dict[str, Any]]] = {}
    expanded = 0

    def trace(state: tuple[int, int, str | None]) -> list[dict[str, Any]]:
        result = []
        edge = parents[state]
        while edge is not None:
            previous, action = edge
            result.append(action)
            edge = parents[previous]
        return list(reversed(result))

    while frontier and expanded < max_nodes:
        current = frontier.popleft()
        expanded += 1
        x, y, held = current
        if held is not None:
            pick_id = f"pick:{held}"
            if pick_id in goals and pick_id not in witnesses:
                witnesses[pick_id] = trace(current)
            for target in runtime["targets"]:
                place_id = f"place:{held}->{target['id']}"
                if target["cell"] == [x, y] and place_id in goals and place_id not in witnesses:
                    witnesses[place_id] = trace(current) + [{"action": "place", "target_id": target["id"]}]
        successors = []
        for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            nx, ny = x + dx, y + dy
            if 0 <= nx < width and 0 <= ny < height and (nx, ny) not in forbidden:
                successors.append(((nx, ny, held), {"action": "move", "to": [nx, ny]}))
        if held is None:
            for source in runtime["objects"]:
                if source["cell"] == [x, y] and source["movable"] and source["mass"] <= runtime["robot"]["payload"]:
                    successors.append(((x, y, source["id"]), {"action": "pick", "source_id": source["id"]}))
        for following, action in successors:
            if following not in parents:
                parents[following] = (current, action)
                frontier.append(following)
    complete = not frontier
    feasible_goals = sorted(witnesses)
    label = None
    if complete:
        label = "INFEASIBLE" if not feasible_goals else "SUFFICIENT" if len(feasible_goals) == 1 else "INSUFFICIENT"
    for identifier, witness in witnesses.items():
        if not validate_witness(runtime, goals[identifier], witness):
            raise RuntimeError(f"Reference emitted an invalid witness for {identifier}")
    return {"complete": complete, "semantic_candidates": semantic_candidates,
            "feasible_goals": feasible_goals, "witnesses": witnesses,
            "expanded_nodes": expanded, "label": label}


def _generation_predicate(grammar: str, *, target: bool = False) -> dict[str, Any]:
    first_shape, second_shape = ("tray", "bin") if target else ("cube", "sphere")
    red = {"op": "eq", "field": "color", "value": "red"}
    blue = {"op": "eq", "field": "color", "value": "blue"}
    first = {"op": "eq", "field": "shape", "value": first_shape}
    second = {"op": "eq", "field": "shape", "value": second_shape}
    if grammar == "atom":
        return red
    if grammar == "and":
        return {"op": "and", "args": [red, {"op": "or", "args": [first, second]}]}
    return {"op": "or", "args": [{"op": "and", "args": [red, first]}, {"op": "and", "args": [blue, second]}]}


def generate_cases(n_cases: int = 120, seed: int = 20260920) -> list[dict[str, Any]]:
    """Generate a declared development recipe, without solving/filtering cases.

    The first 120 cases balance family, scale, source-eligibility pattern and
    grammar. Place sharing layouts rotate prospectively; this pilot does not
    cover every Cartesian stratum combination. A generated sample is one base
    scene cluster, never a new natural-language annotation or confirmation set.
    """
    _require(type(n_cases) is int and n_cases >= 0, "n_cases must be a nonnegative integer")
    _require(type(seed) is int and seed >= 0, "seed must be a nonnegative integer")
    cases = []
    for index in range(n_cases):
        position = index // 2
        family = ("pick", "place")[index % 2]
        scale = CANDIDATE_SCALES[position % len(CANDIDATE_SCALES)]
        pattern = PATTERNS[(position // 4) % len(PATTERNS)]
        grammar = GRAMMARS[(position // 20) % len(GRAMMARS)]
        shared = family == "place" and ((position // 4) + (position // 20)) % 2 == 0
        sharing_layout = "source_shared" if shared else "low"
        target_count = 4 if shared else 1 if family == "place" else 0
        source_count = scale // target_count if target_count else scale
        rng = random.Random(f"{SCHEMA_VERSION}:{seed}:{index}")
        source_ids = list(range(source_count))
        target_ids = list(range(target_count))
        random.Random(f"{SCHEMA_VERSION}:{seed}:{index}:source_ids").shuffle(source_ids)
        random.Random(f"{SCHEMA_VERSION}:{seed}:{index}:target_ids").shuffle(target_ids)
        width, height = 14, 10
        blocked = {(7, y) for y in range(height)} if pattern == "split_wall" else set()
        for y in range(height):
            for x in range(width):
                if (x, y) != (0, 0) and rng.random() < 0.06:
                    blocked.add((x, y))
        free = [(x, y) for y in range(height) for x in range(width)
                if (x, y) not in blocked and (x, y) != (0, 0)]
        rng.shuffle(free)
        # With this fixed grid/density there is ample capacity. Do not resample
        # according to any oracle or method outcome if a recipe is changed.
        _require(len(free) >= source_count + target_count + 2, "Generator recipe exceeds available distinct cells")
        objects = []
        for item_index in range(source_count):
            eligible = (pattern in {"uniform", "split_wall"}
                        or pattern == "mostly_ineligible" and item_index % 8 == 0
                        or pattern == "one_eligible" and item_index == 0)
            color = "blue" if grammar == "or" and item_index % 2 else "red"
            shape = "sphere" if item_index % 2 else "cube"
            mass = round(rng.uniform(0.5, 4.0), 4) if eligible else 6.0 + item_index % 3
            movable = eligible or item_index % 2 == 0
            objects.append({"id": f"source_{source_ids[item_index]:03d}", "cell": list(free.pop()),
                            "color": color, "shape": shape, "mass": mass, "movable": movable})
        targets = []
        for item_index in range(target_count):
            targets.append({"id": f"target_{target_ids[item_index]:03d}", "cell": list(free.pop()),
                            "color": "blue" if grammar == "or" and item_index % 2 else "red",
                            "shape": "bin" if item_index % 2 else "tray"})
        objects.append({"id": "distractor_object", "cell": list(free.pop()), "color": "purple",
                        "shape": "cone", "mass": 1.0, "movable": True})
        targets.append({"id": "distractor_target", "cell": list(free.pop()), "color": "purple", "shape": "shelf"})
        # Independent, prospective permutations prevent a fixed candidate/list
        # order from exploiting the eligibility recipe's construction index.
        random.Random(f"{SCHEMA_VERSION}:{seed}:{index}:object_order").shuffle(objects)
        random.Random(f"{SCHEMA_VERSION}:{seed}:{index}:target_order").shuffle(targets)
        ast = {"action": family, "source": _generation_predicate(grammar),
               "target": _generation_predicate(grammar, target=True) if family == "place" else None}
        runtime = {"instruction": render_instruction(ast),
                   "grid": {"width": width, "height": height, "blocked": [list(cell) for cell in sorted(blocked)]},
                   "robot": {"start": [0, 0], "payload": 5.0}, "objects": objects, "targets": targets,
                   "task_contract": {"mode": "referential"}}
        validate_runtime(runtime)
        cases.append({"instance_id": f"stage1_dev_{seed}_{index:05d}", "cluster_id": f"stage1_scene_{seed}_{index:05d}",
                      "split": "dev", "stratum": {"family": family, "candidate_scale": scale,
                      "pattern": pattern, "grammar": grammar, "sharing_layout": sharing_layout},
                      "runtime": runtime, "ast": ast})
    return cases
