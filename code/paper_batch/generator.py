"""Disjoint generated worlds, fixed recipes, no reachability-based selection."""
import copy
import random

from adaptive_experiments.common import stable
from innovation_stage2.domain import (evaluate_ast, footprint_cells, goal_id,
                                     parse_families, render_instruction, validate_runtime)
from innovation_stage2b.generator import _cell, _generate_case, MAPS, FOOTPRINT_PAIRS

VERSION = "paper-independent-measurement-v1"
OOD = ("language", "topology", "footprint", "catalog")


def physical_fingerprint(runtime):
    """Ignore IDs, language and semantic variants; retain physical attributes."""
    value = {k: copy.deepcopy(runtime[k]) for k in ("grid", "robot")}
    value["grid"]["blocked"].sort()
    for collection in ("objects", "targets"):
        value[collection] = sorted(({k: v for k, v in entity.items() if k not in ("id", "color", "shape", "accepts_colors")}
                                    for entity in runtime[collection]), key=stable)
    return stable(value)


def catalog(runtime):
    return sorted(({"action": "place", "source_id": f["source_id"], "target_id": t}
                   for f in parse_families(runtime) for t in f["target_ids"]), key=goal_id)


def relocate(runtime, rng):
    """Place anchors after an OOD geometry change; never query reachability."""
    grid = runtime["grid"]
    blocked = {tuple(c) for c in grid["blocked"]}
    free = [(x, y) for y in range(grid["height"]) for x in range(grid["width"])
            if (x, y) not in blocked and [x, y] != runtime["robot"]["start"]]
    rng.shuffle(free)
    fallbacks = 0
    for obj in runtime["objects"]:
        selected = None
        for i, anchor in enumerate(free):
            cells = footprint_cells(obj["footprint"], (*anchor, obj["orientation"]))
            if all(0 <= x < grid["width"] and 0 <= y < grid["height"] and (x, y) not in blocked for x, y in cells):
                selected = i
                break
        if selected is None:
            selected, fallbacks = 0, fallbacks + 1
        obj["cell"] = list(free.pop(selected))
    for target in runtime["targets"]:
        target["cell"] = list(free.pop())
    return fallbacks


def make_case(split, index, seed):
    recipe = index // 4 if split == "ood" else index
    cell = _cell("core", ("balanced_2_2", "uneven_3_1")[(recipe // 24) % 2],
                 ("one_eligible", "all_eligible")[(recipe // 4) % 2],
                 tuple(FOOTPRINT_PAIRS)[(recipe // 8) % 3],
                 ("atom", "compound")[(recipe // 48) % 2], index, MAPS[recipe % 4])
    case = _generate_case(index, seed, cell)
    runtime, ast = case["runtime"], case["ast"]
    shift = OOD[index % 4] if split == "ood" else "in_distribution"
    rng = random.Random(f"{VERSION}:{split}:{seed}:{index}")
    if shift == "language":
        # Factorized AND of a shape and OR(colors) is absent from ID trees.
        candidates = evaluate_ast(ast, runtime)
        for field, collection in (("source", "objects"), ("target", "targets")):
            ids = {c["source_id" if field == "source" else "target_id"] for c in candidates}
            selected = [e for e in runtime[collection] if e["id"] in ids]
            ast[field] = {"op": "and", "args": [
                {"op": "eq", "field": "shape", "value": selected[0]["shape"]},
                {"op": "or", "args": [{"op": "eq", "field": "color", "value": c}
                                        for c in sorted({e["color"] for e in selected})]}]}
    elif shift == "topology":
        runtime["grid"] = {"width": 23, "height": 15, "blocked": [
            [x, y] for x, gate in ((7, 4), (15, 10)) for y in range(15) if abs(y - gate) > 1]}
        runtime["robot"]["start"] = [1, 7]
        case["stratum"]["initial_placement_fallback_count"] = relocate(runtime, rng)
    elif shift == "footprint":
        for i, obj in enumerate(runtime["objects"]):
            obj["footprint"] = [5, 1] if i % 2 else [1, 5]
        case["stratum"]["initial_placement_fallback_count"] = relocate(runtime, rng)
    elif shift == "catalog":
        # One new target for each of the four legal pairs: exactly eight goals.
        original = evaluate_ast(ast, runtime)
        sources = {o["id"]: o for o in runtime["objects"]}
        targets = {t["id"]: t for t in runtime["targets"]}
        taken = {tuple(e["cell"]) for e in runtime["objects"] + runtime["targets"]}
        taken.add(tuple(runtime["robot"]["start"]))
        taken.update(map(tuple, runtime["grid"]["blocked"]))
        free = [(x, y) for y in range(runtime["grid"]["height"]) for x in range(runtime["grid"]["width"]) if (x, y) not in taken]
        rng.shuffle(free)
        for i, goal in enumerate(original):
            target = copy.deepcopy(targets[goal["target_id"]])
            target.update(id="extra_" + stable([seed, index, i])[:12], cell=list(free.pop()),
                          accepts_colors=[sources[goal["source_id"]]["color"]])
            runtime["targets"].append(target)
        rng.shuffle(runtime["targets"])
    runtime["instruction"] = render_instruction(ast)
    validate_runtime(runtime)
    public, independent = catalog(runtime), evaluate_ast(ast, runtime)
    if public != independent or len(public) != (8 if shift == "catalog" else 4):
        raise ValueError("Public and independent semantics/declared size disagree")
    case.update(instance_id=f"paper_{split}_{seed}_{index:04d}", split=split, data_version=VERSION,
                cluster_id="paper_" + physical_fingerprint(runtime), shift=shift)
    return case


def make_cases(config):
    cases = [make_case(split, index, config["world_seeds"][split])
             for split, n in config["split_worlds"].items() for index in range(n)]
    if len({c["cluster_id"] for c in cases}) != len(cases):
        raise ValueError("Physical worlds overlap across splits")
    return cases
