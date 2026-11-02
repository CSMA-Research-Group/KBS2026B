"""A bounded engineering pilot; no new-method or held-out accuracy claim."""
from __future__ import annotations
import copy
import json

from .common import stable
from innovation_stage2b.generator import _cell, _generate_case, MAPS, FOOTPRINT_PAIRS
from innovation_stage2.domain import goal_id
from innovation_stage3c.protocol import show

DIRECT_SYSTEM = """Decide whether a robot can autonomously act under the public referential contract.
Interpret the complete AND/OR instruction and Pairing rule. A goal is a source-target pair.
EXECUTE is authorized only if exactly ONE semantically permitted goal is physically feasible.
If at least two distinct permitted goals are feasible, CLARIFY; if a nonempty permitted set has
no feasible goal, REJECT. Use ABSTAIN when you cannot establish the required facts.
The robot moves on the finite grid; it can pick a movable source at its cell if mass <= payload.
Carried rectangles occupy their odd-width/height footprint, swap axes at orientation 1, and cannot
leave the grid or touch blocked cells. Translation requires both footprints to be clear. Rotation
requires a clear square of side max(width,height). Place requires the target cell and an allowed
orientation. Other objects are not dynamic obstacles in this declared discrete domain.
Return only JSON with exactly decision and selected_candidate. decision is EXECUTE, CLARIFY,
REJECT or ABSTAIN. selected_candidate is null except for EXECUTE, when it is exactly an object
with action:"place", source_id and target_id from the input. Do not add confidence or reasoning."""


def cases(config):
    """24 different worlds; no four-language variants counted as new worlds."""
    result = []
    footprints = list(FOOTPRINT_PAIRS)
    for index in range(config["worlds"]):
        cell = _cell("core", "balanced_2_2" if index % 2 == 0 else "uneven_3_1",
                     "one_eligible" if (index // 4) % 2 == 0 else "all_eligible",
                     footprints[(index // 8) % 3], "compound" if index % 2 else "atom",
                     index, MAPS[index % 4])
        case = _generate_case(index, config["world_seed"], cell)
        case["instance_id"] = f"adaptive_dev_{config['world_seed']}_{index:04d}"
        case["split"] = "engineering_dev"
        result.append(case)
    if len({c["cluster_id"] for c in result}) != len(result):
        raise ValueError("Engineering observations must have different worlds")
    return result


def requests(config, public_cases, servers):
    result = []
    for case in public_cases:
        runtime = case["runtime"]
        candidate_messages = show(runtime, "L_STRUCTURED_SCHEMA")["messages"]
        direct_messages = [{"role": "system", "content": DIRECT_SYSTEM},
                           {"role": "user", "content": json.dumps(runtime, ensure_ascii=False, separators=(",", ":"))}]
        for model in servers:
            for seed in config["sample_seeds"]:
                for kind, messages in (("candidates", candidate_messages), ("direct", direct_messages)):
                    result.append({"model": model, "instance_id": case["instance_id"], "kind": kind,
                                   "messages": copy.deepcopy(messages), "seed": seed,
                                   "temperature": config["temperature"], "top_p": config["top_p"],
                                   "max_tokens": config["max_tokens"], "runtime_sha256": stable(runtime)})
            result.append({"model": model, "instance_id": case["instance_id"], "kind": "decision_scores",
                           "messages": [{"role": "system", "content": DIRECT_SYSTEM.split("Return only JSON")[0]
                                + "Return exactly one letter: A=EXECUTE, B=CLARIFY, C=REJECT, D=ABSTAIN."},
                                        copy.deepcopy(direct_messages[1])],
                           "labels": ["A", "B", "C", "D"], "runtime_sha256": stable(runtime)})
    return result


def direct_output(response, runtime):
    from innovation_stage3.protocol import _unique
    if response["finish_reason"] != "stop":
        raise ValueError("truncated_direct_output")
    value = json.loads(response["content"], object_pairs_hook=_unique)
    if not isinstance(value, dict) or set(value) != {"decision", "selected_candidate"}:
        raise ValueError("direct_schema")
    decision, candidate = value["decision"], value["selected_candidate"]
    if decision not in ("EXECUTE", "CLARIFY", "REJECT", "ABSTAIN"):
        raise ValueError("direct_decision")
    selected = None
    if decision == "EXECUTE":
        selected = goal_id(candidate)
        if (candidate["source_id"] not in {o["id"] for o in runtime["objects"]}
                or candidate["target_id"] not in {t["id"] for t in runtime["targets"]}):
            raise ValueError("direct_unknown_id")
    elif candidate is not None:
        raise ValueError("direct_unexpected_selection")
    return {"decision": decision, "selected_goal": selected}
