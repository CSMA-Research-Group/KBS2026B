"""Prospectively crossed, unfiltered Stage 2 engineering case generator.

The 96 cells cross map, semantic pairing, source eligibility and replication.
All source/target positions are sampled once. Disconnected worlds, invalid
initial cargo footprints and empty semantic outcomes are retained; neither
reference labels nor policy outputs are consulted by the generator.
"""

from __future__ import annotations

import random

from .domain import footprint_cells, render_instruction, validate_runtime


DATA_VERSION = "innovation-stage2-footprint-engineering-v1"
SCHEMA_VERSION = "innovation-stage2-case-v1"
MAPS = ("open", "narrow_gate", "wide_gate", "split_wall")
PAIRINGS = ("any", "compatible")
ELIGIBILITY = ("all_eligible", "mixed", "all_ineligible")
_COLORS = ("red", "blue", "green", "yellow", "purple", "white", "orange", "cyan")
_SHAPES = ("cube", "sphere", "prism", "cylinder", "ring", "capsule", "cone", "pyramid")
_TARGET_SHAPES = ("tray", "bin", "pad", "slot", "basket", "shelf", "dish", "dock")
_FOOTPRINTS = ((1, 1), (3, 1), (3, 3))
_ACCEPTANCE = (
    ((0, 1, 2), (0, 1), (0,), (0, 2), (2,), ()),
    ((0, 2), (0,), (2,), (2,), (), ()),
    ((0, 1, 2), (1, 2), (2,), (1, 2), (1,), ()),
    ((0,), (0,), (0, 1), (1,), (2,), (2,)),
)


def design_cells():
    """Metadata only. The first eight cases cover both pairings and all maps."""
    return [{"map": map_name, "pairing": pairing, "eligibility": eligibility, "replicate": replicate,
             "placement_rule": "anchor_only_unchecked" if replicate == 3 else "initial_footprint_valid_if_available"}
            for replicate in range(4) for eligibility in ELIGIBILITY
            for map_name in MAPS for pairing in PAIRINGS]


def _predicate(grammar, shape, colors, rng):
    def atom(field, value):
        return {"op": "eq", "field": field, "value": value}

    def combine(operator, arguments):
        rng.shuffle(arguments)
        return {"op": operator, "args": arguments}

    if grammar == "atom":
        return atom("shape", shape)
    if grammar == "and":
        return combine("and", [atom("shape", shape), combine("or", [atom("color", color) for color in colors])])
    return combine("or", [combine("and", [atom("shape", shape), atom("color", color)]) for color in colors])


def _physical(eligible, rng):
    if eligible:
        return {"mass": round(rng.uniform(0.5, 4.0), 4), "movable": True}
    return {"mass": float(rng.choice((6, 7, 8))), "movable": bool(rng.randrange(2))}


def _initial_configuration_clear(cell, footprint, orientation, width, height, blocked):
    """Initial-scene geometry only; no robot reachability or final-goal search."""
    return all(0 <= x < width and 0 <= y < height and (x, y) not in blocked
               for x, y in footprint_cells(footprint, (*cell, orientation)))


def _generate_case(index, seed, stratum):
    prefix = f"{DATA_VERSION}:{seed}:{index}"
    world_rng = random.Random(prefix + ":world")
    semantic_rng = random.Random(prefix + ":semantics")
    attribute_rng = random.Random(prefix + ":attributes")
    source_colors = semantic_rng.sample(_COLORS, 4)
    target_colors = semantic_rng.sample(_COLORS, 4)
    source_shapes = semantic_rng.sample(_SHAPES, 2)
    target_shapes = semantic_rng.sample(_TARGET_SHAPES, 2)
    grammar = ("atom", "and", "or")[(stratum["replicate"] + MAPS.index(stratum["map"]) + PAIRINGS.index(stratum["pairing"])) % 3]
    source_predicate = _predicate(grammar, source_shapes[0], source_colors[:3], semantic_rng)
    target_predicate = _predicate(grammar, target_shapes[0], target_colors[:3], semantic_rng)
    width, height = 17, 11
    start = [1, 5]
    if stratum["map"] == "open":
        blocked, gate = set(), set()
    else:
        gate = ({(8, 5)} if stratum["map"] == "narrow_gate" else
                {(8, 4), (8, 5), (8, 6)} if stratum["map"] == "wide_gate" else set())
        blocked = {(8, y) for y in range(height)} - gate
    for y in range(height):
        for x in range(width):
            if [x, y] != start and (x, y) not in gate and world_rng.random() < 0.018:
                blocked.add((x, y))
    free = [(x, y) for y in range(height) for x in range(width) if (x, y) not in blocked and [x, y] != start]
    world_rng.shuffle(free)
    if len(free) < 16:
        raise ValueError("Predeclared scene recipe has insufficient distinct cells")
    identifiers = iter(f"entity_{value:012x}" for value in random.Random(prefix + ":ids").sample(range(1 << 48), 16))
    footprints = list(_FOOTPRINTS) * 2
    random.Random(prefix + ":footprints").shuffle(footprints)
    count_eligible = {"all_eligible": 6, "mixed": 3, "all_ineligible": 0}[stratum["eligibility"]]
    eligible_indices = set(random.Random(prefix + ":eligibility").sample(range(6), count_eligible))
    fallback_count = 0

    def source_cell(footprint, orientation):
        nonlocal fallback_count
        if stratum["placement_rule"] == "initial_footprint_valid_if_available":
            for position in range(len(free) - 1, -1, -1):
                if _initial_configuration_clear(free[position], footprint, orientation, width, height, blocked):
                    return list(free.pop(position))
            # Do not redraw a map, object, orientation, or final-goal label.
            # Preserve the object at an available anchor and record the failure.
            fallback_count += 1
        return list(free.pop())

    objects = []
    for item_index in range(6):
        physical = _physical(item_index in eligible_indices, attribute_rng)
        orientation = attribute_rng.randrange(2)
        footprint = list(footprints[item_index])
        objects.append({"id": next(identifiers), "cell": source_cell(footprint, orientation), "color": source_colors[item_index % 3],
                        "shape": source_shapes[0], **physical, "footprint": footprint, "orientation": orientation})
    # Noncandidates share ID, physical-attribute and footprint distributions.
    # Compound near matches each satisfy part, but not all, of the expression.
    near_source = ([(source_colors[0], source_shapes[1]), (source_colors[1], source_shapes[1])] if grammar == "atom"
                   else [(source_colors[0], source_shapes[1]), (source_colors[3], source_shapes[0])])
    for color, shape in near_source:
        eligible = attribute_rng.random() < count_eligible / 6
        physical = _physical(eligible, attribute_rng)
        footprint = list(attribute_rng.choice(_FOOTPRINTS))
        orientation = attribute_rng.randrange(2)
        objects.append({"id": next(identifiers), "cell": source_cell(footprint, orientation), "color": color, "shape": shape,
                        **physical, "footprint": footprint, "orientation": orientation})
    accept_patterns = list(_ACCEPTANCE[stratum["replicate"]])
    random.Random(prefix + ":acceptance_order").shuffle(accept_patterns)
    orientations = [[0], [1], [0, 1]] * 2
    random.Random(prefix + ":target_orientations").shuffle(orientations)
    targets = []
    for item_index in range(6):
        targets.append({"id": next(identifiers), "cell": list(free.pop()), "color": target_colors[item_index % 3],
                        "shape": target_shapes[0], "allowed_orientations": list(orientations[item_index]),
                        "accepts_colors": [source_colors[color] for color in accept_patterns[item_index]]})
    near_target = ([(target_colors[0], target_shapes[1]), (target_colors[1], target_shapes[1])] if grammar == "atom"
                   else [(target_colors[0], target_shapes[1]), (target_colors[3], target_shapes[0])])
    for color, shape in near_target:
        targets.append({"id": next(identifiers), "cell": list(free.pop()), "color": color, "shape": shape,
                        "allowed_orientations": list(attribute_rng.choice(([0], [1], [0, 1]))),
                        "accepts_colors": [source_colors[color_index] for color_index in attribute_rng.choice(accept_patterns)]})
    random.Random(prefix + ":object_order").shuffle(objects)
    random.Random(prefix + ":target_order").shuffle(targets)
    ast = {"action": "place", "source": source_predicate, "target": target_predicate, "pairing": stratum["pairing"]}
    runtime = {"instruction": render_instruction(ast), "grid": {"width": width, "height": height, "blocked": [list(cell) for cell in sorted(blocked)]},
               "robot": {"start": start, "payload": 5.0}, "objects": objects, "targets": targets,
               "task_contract": {"mode": "referential"}}
    validate_runtime(runtime)
    return {"instance_id": f"stage2_dev_{seed}_{index:04d}", "cluster_id": f"stage2_scene_{seed}_{index:04d}",
            "split": "dev", "data_version": DATA_VERSION, "schema_version": SCHEMA_VERSION,
            "stratum": {**stratum, "grammar": grammar, "initial_placement_fallback_count": fallback_count},
            "runtime": runtime, "ast": ast}


def generate_cases(n_cases=96, seed=20260923):
    if type(n_cases) is not int or not 0 <= n_cases <= 96:
        raise ValueError("n_cases must be an engineering prefix between 0 and 96")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    cells = design_cells()
    return [_generate_case(index, seed, cells[index]) for index in range(n_cases)]
