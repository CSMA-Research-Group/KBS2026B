"""Predeclared sparse-graph development data with shared physical-world clusters.

Core: 48 independently generated worlds, each rendered with two four-edge
semantic graphs and two equivalent controlled-language forms (192 observations).
Stress: 24 additional worlds with 0/1/2 semantic edges (24 observations). The
stress singleton is an explicitly specified goal, not the physical-uniqueness
test in the core panel. Neither panel is a held-out confirmation distribution.

Construction consults only declared initial-footprint geometry. It never calls
a reachability oracle, policy, or label-based rejection sampler. All prescribed
outcomes, including initialization fallbacks, are retained.
"""

from __future__ import annotations

import hashlib
import json
import random

from innovation_stage2.domain import footprint_cells, render_instruction, validate_runtime


DEFAULT_N_CASES = 216
DATA_VERSION = "innovation-stage2b-sparse-clustered-development-v1"
SCHEMA_VERSION = "innovation-stage2b-case-v1"
MAPS = ("open", "narrow_gate", "wide_gate", "split_wall")
FOOTPRINT_PAIRS = {
    "point_square": ((1, 1), (3, 3)),
    "bar_square": ((3, 1), (3, 3)),
    "bar_bar": ((3, 1), (3, 1)),
}
GRAMMARS = ("atom", "compound")
CORE_GRAPHS = ("balanced_2_2", "uneven_3_1")
PLACEMENT_RULE = "initial_footprint_valid_if_available"
_COLORS = ("red", "blue", "green", "yellow", "purple", "white", "orange", "cyan")
_SOURCE_SHAPES = ("cube", "sphere", "prism", "cylinder", "ring", "capsule", "cone", "pyramid")
_TARGET_SHAPES = ("tray", "bin", "pad", "slot", "basket", "shelf", "dish", "dock")


def _cell(panel, semantic_graph, eligibility, footprint_pair, grammar, replicate, map_name):
    return {"panel": panel, "semantic_graph": semantic_graph, "eligibility": eligibility,
            "footprint_pair": footprint_pair, "grammar": grammar, "replicate": replicate,
            "map": map_name, "placement_rule": PLACEMENT_RULE, "initial_placement_fallback_count": 0}


def design_cells():
    """All 216 metadata cells; every five-row early block includes a stress row.

    A short engineering prefix can contain incomplete four-variant core clusters;
    it must not be reported as the complete development design.
    """
    core_worlds = [(map_name, eligibility, footprint_pair, replicate)
                   for replicate in range(2) for footprint_pair in FOOTPRINT_PAIRS
                   for eligibility in ("one_eligible", "all_eligible") for map_name in MAPS]
    stress = [_cell("stress", f"stress_edges_{edges}", eligibility, "point_square", "atom", 0, map_name)
              for edges in (0, 1, 2) for eligibility in ("all_eligible", "all_ineligible") for map_name in MAPS]
    cells = []
    for world_index, (map_name, eligibility, footprint_pair, replicate) in enumerate(core_worlds):
        cells.extend(_cell("core", graph, eligibility, footprint_pair, grammar, replicate, map_name)
                     for graph in CORE_GRAPHS for grammar in GRAMMARS)
        if world_index < len(stress):
            cells.append(stress[world_index])
    return cells


def _world_key(seed, cell):
    factors = {key: cell[key] for key in ("panel", "map", "eligibility", "footprint_pair", "replicate")}
    # Core graph/language variants share a world; every stress recipe owns one.
    if cell["panel"] == "stress":
        factors["semantic_graph"] = cell["semantic_graph"]
    return json.dumps({"version": DATA_VERSION, "seed": seed, "factors": factors}, sort_keys=True, separators=(",", ":"))


def _physical_attributes(eligible, rng):
    if eligible:
        return {"mass": round(rng.uniform(0.5, 4.0), 4), "movable": True}
    return {"mass": float(rng.choice((6, 7, 8))), "movable": bool(rng.randrange(2))}


def _initial_clear(cell, footprint, orientation, width, height, blocked):
    """Initial-state validity, without searching robot or goal reachability."""
    return all(0 <= x < width and 0 <= y < height and (x, y) not in blocked
               for x, y in footprint_cells(footprint, (*cell, orientation)))


def _make_world(seed, cell):
    key = _world_key(seed, cell)
    world_rng = random.Random(key + ":world")
    attribute_rng = random.Random(key + ":attributes")
    semantic_rng = random.Random(key + ":semantic_vocabulary")
    source_colors = semantic_rng.sample(_COLORS, 2)
    target_colors = semantic_rng.sample(_COLORS, 3)
    source_shapes = semantic_rng.sample(_SOURCE_SHAPES, 2)
    target_shapes = semantic_rng.sample(_TARGET_SHAPES, 2)
    width, height, start = 17, 11, [1, 5]
    if cell["map"] == "open":
        blocked, gate = set(), set()
    else:
        gate = ({(8, 5)} if cell["map"] == "narrow_gate" else
                {(8, 4), (8, 5), (8, 6)} if cell["map"] == "wide_gate" else set())
        blocked = {(8, y) for y in range(height)} - gate
    for y in range(height):
        for x in range(width):
            if [x, y] != start and (x, y) not in gate and world_rng.random() < 0.018:
                blocked.add((x, y))
    free = [(x, y) for y in range(height) for x in range(width) if (x, y) not in blocked and [x, y] != start]
    world_rng.shuffle(free)
    if len(free) < 9:
        raise ValueError("The predefined physical world lacks distinct anchors; no map redraw is allowed")
    fallback_count = 0

    def source_cell(footprint, orientation):
        nonlocal fallback_count
        for index in range(len(free) - 1, -1, -1):
            if _initial_clear(free[index], footprint, orientation, width, height, blocked):
                return list(free.pop(index))
        fallback_count += 1
        return list(free.pop())

    identifiers = iter(f"entity_{value:012x}" for value in random.Random(key + ":ids").sample(range(1 << 48), 9))
    eligible_count = {"one_eligible": 1, "all_eligible": 2, "all_ineligible": 0}[cell["eligibility"]]
    eligible_indices = set(random.Random(key + ":eligible_identity").sample(range(2), eligible_count))
    # Independent namespace: changing the chosen eligible source is not a rule
    # for choosing the high-degree source, or vice versa.
    high_degree_source = random.Random(key + ":high_degree_identity").randrange(2)
    footprints = FOOTPRINT_PAIRS[cell["footprint_pair"]]
    objects, source_ids = [], []
    for index in range(2):
        identity = next(identifiers)
        source_ids.append(identity)
        physical = _physical_attributes(index in eligible_indices, attribute_rng)
        orientation = attribute_rng.randrange(2)
        footprint = list(footprints[index])
        objects.append({"id": identity, "cell": source_cell(footprint, orientation), "color": source_colors[index],
                        "shape": source_shapes[0], **physical, "footprint": footprint, "orientation": orientation})
    # Both forms must denote the same objects in this same physical world.
    # Near matches satisfy a color atom of the compound form but not its shape.
    for index in range(2):
        physical = _physical_attributes(attribute_rng.random() < eligible_count / 2, attribute_rng)
        footprint = list(attribute_rng.choice(footprints))
        orientation = attribute_rng.randrange(2)
        objects.append({"id": next(identifiers), "cell": source_cell(footprint, orientation), "color": source_colors[index],
                        "shape": source_shapes[1], **physical, "footprint": footprint, "orientation": orientation})
    orientations = [[0], [1], [0, 1]]
    random.Random(key + ":target_orientations").shuffle(orientations)
    target_ids, targets = [], []
    for index in range(3):
        identity = next(identifiers)
        target_ids.append(identity)
        targets.append({"id": identity, "cell": list(free.pop()), "color": target_colors[index], "shape": target_shapes[0],
                        "allowed_orientations": list(orientations[index]), "accepts_colors": []})
    for index in range(2):
        targets.append({"id": next(identifiers), "cell": list(free.pop()), "color": target_colors[index], "shape": target_shapes[1],
                        "allowed_orientations": list(attribute_rng.choice(([0], [1], [0, 1]))), "accepts_colors": list(source_colors)})
    random.Random(key + ":object_order").shuffle(objects)
    random.Random(key + ":target_order").shuffle(targets)
    target_roles = list(target_ids)
    random.Random(key + ":target_roles").shuffle(target_roles)
    runtime = {"instruction": "Unrendered controlled instruction.",
               "grid": {"width": width, "height": height, "blocked": [list(item) for item in sorted(blocked)]},
               "robot": {"start": start, "payload": 5.0}, "objects": objects, "targets": targets,
               "task_contract": {"mode": "referential"}}
    return {"runtime": runtime, "source_ids": source_ids, "target_ids": target_ids, "target_roles": target_roles,
            "source_colors": source_colors, "target_colors": target_colors, "source_shape": source_shapes[0],
            "target_shape": target_shapes[0], "high_degree_source": high_degree_source,
            "fallback_count": fallback_count, "world_key": key}


def _predicate(grammar, shape, colors):
    shape_atom = {"op": "eq", "field": "shape", "value": shape}
    if grammar == "atom":
        return shape_atom
    return {"op": "or", "args": [{"op": "and", "args": [dict(shape_atom), {"op": "eq", "field": "color", "value": color}]}
                                   for color in colors]}


def _generate_case(index, seed, cell):
    world = _make_world(seed, cell)
    runtime = world["runtime"]
    source_ids, roles = world["source_ids"], world["target_roles"]
    graph = cell["semantic_graph"]
    if graph == "balanced_2_2":
        edges = [(0, roles[0]), (0, roles[1]), (1, roles[1]), (1, roles[2])]
    elif graph == "uneven_3_1":
        high = world["high_degree_source"]
        edges = [(high, target) for target in roles] + [(1 - high, roles[1])]
    elif graph == "stress_edges_0":
        edges = []
    elif graph == "stress_edges_1":
        source = random.Random(world["world_key"] + ":singleton_source").randrange(2)
        edges = [(source, roles[0])]
    elif graph == "stress_edges_2":
        edges = [(0, roles[0]), (1, roles[1])]
    else:
        raise ValueError("Unknown predeclared semantic graph")
    targets = {target["id"]: target for target in runtime["targets"]}
    for source_index, target_id in edges:
        targets[target_id]["accepts_colors"].append(world["source_colors"][source_index])
    for target in runtime["targets"]:
        target["accepts_colors"].sort()
    ast = {"action": "place", "source": _predicate(cell["grammar"], world["source_shape"], world["source_colors"]),
           "target": _predicate(cell["grammar"], world["target_shape"], world["target_colors"]), "pairing": "compatible"}
    runtime["instruction"] = render_instruction(ast)
    validate_runtime(runtime)
    cluster_hash = hashlib.sha256(world["world_key"].encode()).hexdigest()[:20]
    return {"instance_id": f"stage2b_dev_{seed}_{index:04d}", "cluster_id": f"stage2b_{cell['panel']}_{cluster_hash}",
            "split": "dev", "data_version": DATA_VERSION, "schema_version": SCHEMA_VERSION,
            "stratum": {**cell, "initial_placement_fallback_count": world["fallback_count"]}, "runtime": runtime, "ast": ast}


def generate_cases(n_cases=DEFAULT_N_CASES, seed=20260925):
    """Return a fixed engineering prefix, never selecting by reference labels."""
    if type(n_cases) is not int or not 0 <= n_cases <= DEFAULT_N_CASES:
        raise ValueError("n_cases must be a fixed prefix from 0 through 216")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    return [_generate_case(index, seed, cell) for index, cell in enumerate(design_cells()[:n_cases])]
