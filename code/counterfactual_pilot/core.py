"""Paired construction, explicit runtime contracts and programmatic references."""
import copy
import json
import random

from adaptive_experiments.common import stable
from adaptive_experiments.protocol import DIRECT_SYSTEM
from gemma_direct_format.protocol import FORMAT_SUFFIX
from innovation_stage2.domain import evaluate_ast, goal_id, render_instruction, validate_runtime
from innovation_stage3c.protocol import show

VERSION = "counterfactual-contract-development-v1"
VARIANTS = ("base", "add_feasible", "add_infeasible")
CONTRACTS = ("referential", "any_valid")
CONTRACT_TEXT = {
    "referential": "The execution contract requires a UNIQUE semantically permitted physically feasible source-target pair. If two or more such pairs exist, CLARIFY. If none exist in a nonempty semantic set, REJECT. If the required facts cannot be established, ABSTAIN.",
    "any_valid": "The execution contract permits ANY ONE semantically permitted physically feasible source-target pair. EXECUTE may select any such pair, even when other feasible pairs exist. Multiple feasible alternatives do not require CLARIFY. If none exist in a nonempty semantic set, REJECT. If the required facts cannot be established, ABSTAIN.",
}


def physical(runtime):
    """Only the decision contract changes; legacy motion/semantics are identical."""
    mode = runtime["task_contract"]["mode"]
    if mode not in CONTRACTS:
        raise ValueError("Unknown task contract")
    result = copy.deepcopy(runtime)
    result["task_contract"] = {"mode": "referential"}
    validate_runtime(result)
    return result


def rotate(runtime):
    grid = runtime["grid"]
    height = grid["height"]
    def point(cell):
        return [height - 1 - cell[1], cell[0]]
    grid["width"], grid["height"] = grid["height"], grid["width"]
    grid["blocked"] = sorted(point(c) for c in grid["blocked"])
    runtime["robot"]["start"] = point(runtime["robot"]["start"])
    for obj in runtime["objects"]:
        obj["cell"] = point(obj["cell"])
        obj["orientation"] = 1 - obj["orientation"]
    for target in runtime["targets"]:
        target["cell"] = point(target["cell"])
        target["allowed_orientations"] = sorted(1 - v for v in target["allowed_orientations"])


def make_cases(config):
    """No rejection sampling or model-dependent filtering. Failed assertions stop the batch.

    Closed separating walls guarantee the inaccessible intervention. This small,
    deliberately constructed domain is a mechanism/interface development pilot.
    """
    if config["base_worlds"] != 24:
        raise ValueError("This entry is the fixed 24-base-world development pilot")
    cases = []
    for index in range(config["base_worlds"]):
        rng = random.Random(f"{VERSION}:{config['world_seed']}:{index}")
        width, height = rng.choice((19, 21, 23)), rng.choice((13, 15, 17))
        wall = width // 2
        colors = rng.sample(("red", "blue", "green", "yellow", "orange", "purple"), 6)
        primary, secondary, distractor, target_color, target_other, irrelevant = colors
        ids = ["entity_" + stable([config["world_seed"], index, i])[:12] for i in range(12)]
        left = [[x, y] for x in range(2, wall - 1) for y in range(2, height - 2)]
        right = [[x, y] for x in range(wall + 2, width - 2) for y in range(2, height - 2)]
        rng.shuffle(left)
        rng.shuffle(right)
        footprint = ([1, 1], [3, 1], [1, 3])[(index // 4) % 3]
        objects = [dict(id=ids[0], cell=left.pop(), color=primary, shape="block", mass=1,
                        movable=True, footprint=list(footprint), orientation=0)]
        for i in range(1, 6):
            objects.append(dict(id=ids[i], cell=left.pop(), color=secondary if i < 3 else distractor,
                shape="block", mass=9 if i == 1 else 1, movable=i != 2,
                footprint=[1, 1], orientation=0))
        targets = [dict(id=ids[6], cell=left.pop(), color=target_color, shape="tray",
                        allowed_orientations=[0, 1], accepts_colors=[primary]),
                   dict(id=ids[7], cell=left.pop(), color=target_other, shape="tray",
                        allowed_orientations=[0, 1], accepts_colors=[secondary])]
        for i in range(8, 11):
            targets.append(dict(id=ids[i], cell=right.pop(), color=irrelevant, shape="tray",
                                allowed_orientations=[0, 1], accepts_colors=[primary, secondary]))
        ast = {"action": "place", "pairing": "compatible",
               "source": {"op": "or", "args": [{"op": "eq", "field": "color", "value": c} for c in (primary, secondary)]},
               "target": {"op": "or", "args": [{"op": "eq", "field": "color", "value": c} for c in (target_color, target_other)]}}
        if index % 2:
            for field, shape in (("source", "block"), ("target", "tray")):
                ast[field] = {"op": "and", "args": [ast[field], {"op": "eq", "field": "shape", "value": shape}]}
        rng.shuffle(objects)
        rng.shuffle(targets)
        base = {"instruction": render_instruction(ast), "grid": {"width": width, "height": height,
                "blocked": [[wall, y] for y in range(height)]},
                "robot": {"start": left.pop(), "payload": 5}, "objects": objects,
                "targets": targets, "task_contract": {"mode": "referential"}}
        addition = dict(id=ids[11], color=target_color, shape="tray", allowed_orientations=[0, 1], accepts_colors=[primary])
        feasible_cell, infeasible_cell = left.pop(), right.pop()
        insert_at = rng.randrange(len(targets) + 1)
        cluster = f"cfdev_{config['world_seed']}_{index:03d}"
        for variant in VARIANTS:
            runtime = copy.deepcopy(base)
            if variant != "base":
                item = {**copy.deepcopy(addition), "cell": feasible_cell if variant == "add_feasible" else infeasible_cell}
                runtime["targets"].insert(insert_at, item)
            for _ in range(index % 4):
                rotate(runtime)
            validate_runtime(runtime)
            for mode in CONTRACTS:
                public = copy.deepcopy(runtime)
                public["task_contract"] = {"mode": mode}
                cases.append({"instance_id": cluster + "_" + variant + "_" + mode, "cluster_id": cluster,
                    "split": "development", "variant": variant, "contract": mode, "runtime": public,
                    "ast": copy.deepcopy(ast), "data_version": VERSION,
                    "primary_goal": f"place:{ids[0]}->{ids[6]}",
                    "added_goal": f"place:{ids[0]}->{ids[11]}",
                    "expected_feasible_count": 2 if variant == "add_feasible" else 1,
                    "rotation": index % 4, "footprint": list(footprint)})
    if len({c["instance_id"] for c in cases}) != 144:
        raise ValueError("Duplicate/missing development conditions")
    return cases


def requests(config, cases, servers):
    result = []
    physics_text = DIRECT_SYSTEM[DIRECT_SYSTEM.index("The robot moves"):]
    for case in cases:
        runtime, mode = case["runtime"], case["contract"]
        candidate_messages = show(physical(runtime), "L_STRUCTURED_SCHEMA")["messages"]
        payload = json.loads(candidate_messages[1]["content"])
        payload["task_contract"] = runtime["task_contract"]
        candidate_messages[1]["content"] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        candidate_messages[0]["content"] += "\nExecution contract: " + CONTRACT_TEXT[mode] + " This enumeration stage must still list ALL semantic alternatives; execution permission does not change semantic membership."
        direct_messages = [{"role": "system", "content": "Interpret the complete AND/OR instruction and Pairing rule. A goal is a source-target pair.\n" + CONTRACT_TEXT[mode] + "\n" + physics_text},
            {"role": "user", "content": json.dumps(runtime, sort_keys=True, separators=(",", ":")) + FORMAT_SUFFIX}]
        for model in servers:
            base = {"model": model, "instance_id": case["instance_id"], "runtime_sha256": stable(runtime), "contract": mode}
            for kind, seeds, messages in (("candidates", config["sample_seeds"], candidate_messages),
                                         ("direct", [config["direct_seed"]], direct_messages)):
                for seed in seeds:
                    result.append({**base, "kind": kind, "seed": seed, "messages": copy.deepcopy(messages),
                        "temperature": config["temperature"], "top_p": config["top_p"],
                        "max_tokens": config[kind[:-1] + "_max_tokens"] if kind == "candidates" else config["direct_max_tokens"]})
    random.Random(config["request_order_seed"]).shuffle(result)
    if len(result) != 1152 or len({stable(s) for s in result}) != 1152:
        raise ValueError("Expected 864 candidate and 288 direct generations")
    return result


def score(view, reference, mode):
    if mode not in CONTRACTS:
        raise ValueError("Unknown scoring contract")
    semantic = {goal_id(g) for g in reference["semantic_candidates"]}
    feasible = set(reference["feasible_goals"])
    scorable = bool(semantic) and reference["complete"]
    expected = ("EXECUTE" if feasible else "REJECT") if mode == "any_valid" else (
        "CLARIFY" if len(feasible) >= 2 else "EXECUTE" if feasible else "REJECT")
    execute = view["decision"] == "EXECUTE"
    correct = scorable and execute and view["selected_goal"] in feasible and (mode == "any_valid" or len(feasible) == 1)
    return {"scorable": scorable, "n_semantic": len(semantic), "n_feasible": len(feasible) if reference["complete"] else None,
        "expected_decision": expected if scorable else None, "correct_execution": correct if scorable else None,
        "execution_error": not correct if scorable and execute else None,
        "correct_decision": (view["decision"] == expected and (not execute or correct)) if scorable else None,
        "selected_feasible": view["selected_goal"] in feasible if execute and reference["complete"] else None}
