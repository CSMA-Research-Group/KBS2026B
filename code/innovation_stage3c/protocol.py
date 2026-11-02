"""Two fixed format-reminder interventions, without instance answers."""
import json
from pathlib import Path

from innovation_stage3.storage import stable
from innovation_stage3b.protocol import presentation, parse_completion
from src.local_inference.server import validate_chat_request

CONDITIONS = ("L_STRUCTURED_SCHEMA", "S_STRUCTURED_SCHEMA")
BASE_CONDITIONS = {"L_STRUCTURED_SCHEMA": "L_STRUCTURED", "S_STRUCTURED_SCHEMA": "S_STRUCTURED"}
OUTPUT_SCHEMA = {
    "type": "object", "properties": {"candidates": {"type": "array", "items": {
        "type": "object", "properties": {"action": {"const": "place"},
            "source_id": {"type": "string"}, "target_id": {"type": "string"}},
        "required": ["action", "source_id", "target_id"], "additionalProperties": False}}},
    "required": ["candidates"], "additionalProperties": False,
}
FORMAT_REMINDER = (
    "\nOutput syntax reminder: return one JSON instance satisfying the following JSON Schema, not the schema itself. "
    "The root is an OBJECT: start with { and end with }. Its candidates value is an ARRAY. "
    "Every array item is an OBJECT with action, source_id, and target_id, not a list of IDs. "
    "Use the actual IDs from the provided lists. Include every semantically permitted pair under the preceding rules; "
    "the array length is not fixed. Return only the JSON object, with no Markdown or explanation.\n"
    + json.dumps(OUTPUT_SCHEMA, ensure_ascii=False, separators=(",", ":"))
)


def show(runtime, condition):
    if condition not in CONDITIONS:
        raise ValueError("Only the two frozen new conditions may be sampled")
    shown = presentation(runtime, BASE_CONDITIONS[condition])
    shown["messages"][0]["content"] += FORMAT_REMINDER
    return shown


def spec(plan, case, seed, condition):
    shown = show(case["runtime"], condition)
    config = plan["servers"]["qwen"]
    body = {"model": config["served_model"], "messages": shown["messages"], "stream": False,
            **{k: plan["config"][k] for k in ("temperature", "top_p", "max_tokens")}, "seed": seed}
    validate_chat_request(body, config)
    value = {"model_key": "qwen", "condition": condition, "instance_id": case["instance_id"],
             "sample_seed": seed, "plan_sha256": stable(plan), "request": body,
             "presentation_sha256": stable(shown)}
    return {**value, "binding_sha256": stable(value)}


def sequence(plan, cases):
    result = []
    for index, case in enumerate(cases):
        for sample_index, seed in enumerate(plan["config"]["samples"]):
            order = CONDITIONS if (index + sample_index) % 2 == 0 else CONDITIONS[::-1]
            result.extend(spec(plan, case, seed, condition) for condition in order)
    return result


def record_path(bundle, request):
    return Path(bundle) / "requests" / request["condition"] / (request["instance_id"] + "_" + str(request["sample_seed"]) + ".json")


def parse(content, finish_reason, runtime, condition):
    return parse_completion(content, finish_reason, runtime, BASE_CONDITIONS[condition])
