"""Public-only reversible presentation changes; original goal IDs at the backend."""
from __future__ import annotations
import json
from pathlib import Path

from innovation_stage3.protocol import SYSTEM, messages as original_messages, parse_output
from innovation_stage3.storage import stable
from innovation_stage2.domain import goal_id
from src.local_inference.server import validate_chat_request

CONDITIONS = ("L_ORIGINAL", "S_ORIGINAL", "L_STRUCTURED", "S_STRUCTURED")
NEW_CONDITIONS = CONDITIONS[1:]
STRUCTURED_SYSTEM = SYSTEM + '''
Use the following systematic procedure internally before returning the JSON:
1. Check EVERY provided object against the entire source predicate, respecting AND, OR and parentheses. Keep every matching object.
2. Independently check EVERY provided target against the entire target predicate. Keep every matching target. Apply target shape/color predicates to the target's own attributes.
3. For EVERY combination of a matching object and a matching target, check Pairing. If it is any, keep the pair. If it is compatible, keep the pair exactly when the object's color is in that target's accepts_colors list. The target's own color is not its accepts_colors list.
4. Check that all permitted combinations have been included once and that every ID belongs to the correct provided list. Do not stop after finding one valid pair, one target per object, or one object per target.
Return only the required candidates JSON, without the intermediate procedure or explanation.'''


def presentation(runtime, condition):
    if condition not in CONDITIONS:
        raise ValueError("Unknown predeclared condition")
    original = original_messages(runtime)
    payload = json.loads(original[1]["content"])
    forward, reverse = {}, {}
    for collection, prefix in (("objects", "o"), ("targets", "t")):
        fwd = {obj["id"]: (prefix + str(i) if condition.startswith("S_") else obj["id"])
               for i, obj in enumerate(payload[collection])}
        if len(fwd) != len(payload[collection]) or len(set(fwd.values())) != len(fwd):
            raise ValueError("ID mapping is not one-to-one")
        forward[collection], reverse[collection] = fwd, {v: k for k, v in fwd.items()}
        for obj in payload[collection]:
            obj["id"] = fwd[obj["id"]]
    system = STRUCTURED_SYSTEM if condition.endswith("STRUCTURED") else SYSTEM
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False, separators=(",", ":"))}]
    if condition == "L_ORIGINAL" and messages != original:
        raise ValueError("Baseline prompt changed")
    return {"messages": messages, "public_payload": payload, "forward": forward, "reverse": reverse}


def parse_completion(content, finish_reason, runtime, condition):
    shown = presentation(runtime, condition)
    # Strict schema and membership are checked in the displayed ID vocabulary.
    # Thus an unrequested long ID in a short-ID response remains unknown.
    parsed = parse_output(content, finish_reason, shown["public_payload"])
    reverse = shown["reverse"]
    canonical = [{"action": c["action"], "source_id": reverse["objects"][c["source_id"]],
                  "target_id": reverse["targets"][c["target_id"]]} for c in parsed["candidates"]]
    parsed["candidates"] = sorted(canonical, key=goal_id)
    # Invalid/extra items remain in their original response vocabulary in the
    # diagnostics; accepted candidates alone are decoded, with no guess/repair.
    return parsed


def request_spec(plan, model, case, sample, condition):
    if condition not in plan["new_conditions"]:
        raise ValueError("The frozen L_ORIGINAL baseline must not be regenerated")
    shown = presentation(case["runtime"], condition)
    config = plan["servers"][model]
    body = {"model": config["served_model"], "messages": shown["messages"],
            "temperature": plan["config"]["temperature"], "top_p": plan["config"]["top_p"],
            "max_tokens": plan["config"]["max_tokens"], "seed": sample, "stream": False}
    validate_chat_request(body, config)
    value = {"model_key": model, "condition": condition, "instance_id": case["instance_id"],
             "sample_seed": sample, "plan_sha256": stable(plan), "request": body,
             "presentation_sha256": stable(shown)}
    return {**value, "binding_sha256": stable(value)}


def request_sequence(plan, model, cases):
    model_index = list(plan["servers"]).index(model)
    conditions = plan["new_conditions"]
    specs = []
    for case_index, case in enumerate(cases):
        for sample_index, sample in enumerate(plan["config"]["samples"]):
            shift = (case_index + sample_index + model_index) % len(conditions)
            order = conditions[shift:] + conditions[:shift]
            specs.extend(request_spec(plan, model, case, sample, condition) for condition in order)
    return specs


def record_path(bundle, spec):
    return (Path(bundle) / "requests" / spec["condition"] / spec["model_key"] /
            (spec["instance_id"] + "_" + str(spec["sample_seed"]) + ".json"))
