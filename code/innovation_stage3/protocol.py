"""Frozen native candidate protocol; public membership and separate GT scoring."""
from __future__ import annotations

import json
import re
from innovation_stage1.domain import _parse_predicate
from innovation_stage2.domain import goal_id, validate_runtime

SYSTEM = '''Enumerate ALL distinct semantically permitted source-target pairs for the instruction.
Use only the provided objects, targets and their public attributes. AND requires all predicates; OR permits any listed predicate; parentheses specify grouping.
For Pairing: compatible, the source object's color must occur in the target's accepts_colors list. For Pairing: any, ignore accepts_colors.
Enumerate semantic alternatives, not your preferred action. Do not choose just one object or target. Do not filter by mass, movability, geometry, robot ability or imagined feasibility. Each permitted pair must appear once.
Return exactly one JSON object with the single key "candidates", whose value is a list. Each item has exactly "action":"place", "source_id":an existing object ID, and "target_id":an existing target ID. Preserve IDs exactly. If no pair is permitted return {"candidates":[]}.
Do not include explanations, Markdown fences or additional fields.'''

ARMS = ("SINGLE_PROPOSED", "SINGLE_AUDITED", "UNION3_PROPOSED", "UNION3_AUDITED",
        "UNION3_REPAIRED", "PUBLIC_CATALOG")


def messages(runtime):
    validate_runtime(runtime)
    public = {"instruction": runtime["instruction"], "task_contract": runtime["task_contract"],
              "objects": [{k: x[k] for k in ("id", "color", "shape")} for x in runtime["objects"]],
              "targets": [{k: x[k] for k in ("id", "color", "shape", "accepts_colors")} for x in runtime["targets"]]}
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": json.dumps(public, ensure_ascii=False, separators=(",", ":"))}]


def _unique(pairs):
    result = {}
    for k, v in pairs:
        if k in result:
            raise ValueError("duplicate_json_key")
        result[k] = v
    return result


def parse_output(content, finish_reason, runtime):
    """No completion repair: protocol failures remain distinct from valid empty sets.

    Well-formed but unknown-ID/semantically illegal entries are counted and
    excluded by public membership tests. No missing candidate is ever added.
    """
    result = {"valid_protocol": False, "failure": None, "candidates": [],
              "raw_candidate_count": None, "duplicate_count": 0,
              "unknown_id_candidates": [], "semantic_illegal_candidates": []}
    if finish_reason != "stop":
        result["failure"] = "truncated_or_missing_completion"
        return result
    try:
        value = json.loads(content, object_pairs_hook=_unique,
                           parse_constant=lambda x: (_ for _ in ()).throw(ValueError("nonfinite_json")))
        if not isinstance(value, dict) or set(value) != {"candidates"} or not isinstance(value["candidates"], list):
            raise ValueError("candidate_list_schema")
        raw = value["candidates"]
        if len(raw) > 64:
            raise ValueError("candidate_list_bound")
        for candidate in raw:
            goal_id(candidate)
    except (ValueError, TypeError, RecursionError) as exc:
        result["failure"] = "invalid_json_or_schema:" + type(exc).__name__
        return result
    pattern = re.fullmatch(r"Place an object satisfying (.+) into a target satisfying (.+)\. Pairing: (any|compatible)\.", runtime["instruction"])
    if pattern is None:
        raise ValueError("Unsupported frozen public instruction")
    source_predicate, target_predicate = _parse_predicate(pattern[1]), _parse_predicate(pattern[2])
    objects = {x["id"]: x for x in runtime["objects"]}
    targets = {x["id"]: x for x in runtime["targets"]}
    seen, accepted = set(), []
    for candidate in raw:
        key = goal_id(candidate)
        if key in seen:
            result["duplicate_count"] += 1
            continue
        seen.add(key)
        source, target = objects.get(candidate["source_id"]), targets.get(candidate["target_id"])
        if source is None or target is None:
            result["unknown_id_candidates"].append(candidate)
        elif not (source_predicate(source) and target_predicate(target)
                  and (pattern[3] == "any" or source["color"] in target["accepts_colors"])):
            result["semantic_illegal_candidates"].append(candidate)
        else:
            accepted.append(candidate)
    result.update(valid_protocol=True, raw_candidate_count=len(raw), candidates=sorted(accepted, key=goal_id))
    return result


def families(candidates):
    groups = {}
    for candidate in candidates:
        groups.setdefault(candidate["source_id"], set()).add(candidate["target_id"])
    return [{"source_id": source, "target_ids": sorted(targets)} for source, targets in sorted(groups.items())]


def union_proposals(parsed):
    good = [p for p in parsed if p["valid_protocol"]]
    values = {goal_id(c): c for p in good for c in p["candidates"]}
    return {"available": bool(good), "valid_draws": len(good),
            "candidates": [values[k] for k in sorted(values)]}


def decision_view(prediction, retained, complete, *, audit=False, available=True):
    missing = sorted(set(complete) - set(retained))
    decision, selected = prediction["decision"], prediction["selected_goal"]
    if not available or (audit and missing and decision != "CLARIFY"):
        decision, selected = "ABSTAIN", None
    return {"decision": decision, "selected_goal": selected, "missing_public_goals": missing if audit else None,
            "structural_complete": not missing if audit else None, "proposal_available": available}


def score(view, reference):
    semantic = {goal_id(c) for c in reference["semantic_candidates"]}
    feasible = set(reference["feasible_goals"])
    scorable = bool(semantic)
    expected = "EXECUTE" if len(feasible) == 1 else "CLARIFY" if len(feasible) > 1 else "REJECT"
    executed = view["decision"] == "EXECUTE"
    correct_execution = scorable and executed and len(feasible) == 1 and view["selected_goal"] in feasible
    correct_decision = scorable and view["decision"] == expected and (not executed or correct_execution)
    return {"scorable": scorable, "n_semantic": len(semantic), "n_feasible": len(feasible),
            "expected_decision": expected if scorable else None,
            "correct_execution": correct_execution, "execution_error": not correct_execution if executed else None,
            "correct_decision": correct_decision,
            "decision_error": not correct_decision if scorable and view["decision"] != "ABSTAIN" else None}
