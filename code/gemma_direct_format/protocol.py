"""Only the appended output serialization instruction changes between arms."""
import copy

from adaptive_experiments.common import stable
from adaptive_experiments.protocol import direct_output
from innovation_stage3.protocol import score

FORMAT_SUFFIX = '''

Output serialization requirement:
Return one JSON object as the entire response. Start with { and end with }.
Do not use Markdown, backticks, code fences, a language tag, or surrounding text.
The object has exactly two keys: "decision" and "selected_candidate".
"decision" is one of "EXECUTE", "CLARIFY", "REJECT", "ABSTAIN".
For "EXECUTE", "selected_candidate" has exactly "action": "place", "source_id",
and "target_id", using the exact identifiers from the input.
For every other decision, "selected_candidate" is null.
Apply the decision rules already given; this requirement changes only serialization.'''


def make_specs(old_records):
    specs = []
    for record in old_records:
        if record["status"] != "RESPONSE":
            raise ValueError("This control pins an entirely returned baseline")
        original = record["spec"]
        if original["model"] != "gemma" or original["kind"] != "direct":
            raise ValueError("Only Gemma direct decisions belong in this control")
        if original["messages"][-1]["role"] != "user":
            raise ValueError("Expected final user input for the serialization suffix")
        spec = copy.deepcopy(original)
        spec["messages"][-1]["content"] += FORMAT_SUFFIX
        spec.update(contrast="direct_json_suffix_v1", baseline_task_id=record["task_id"])
        specs.append(spec)
    if len({stable(s) for s in specs}) != len(specs):
        raise ValueError("Duplicate follow-up request")
    return specs


def compare_spec(new, old):
    """Reject any unnoticed world, seed, precision-independent setting change."""
    expected = make_specs([old])[0]
    if new != expected:
        raise ValueError("Follow-up changes more than the declared serialization suffix")


def evaluate(record, runtime, reference):
    try:
        if record["status"] != "RESPONSE":
            raise ValueError(record["status"])
        view = direct_output(record["response"], runtime)
        valid, failure = True, None
    except (ValueError, TypeError, KeyError) as exc:
        view = {"decision": "ABSTAIN", "selected_goal": None}
        valid, failure = False, str(exc)
    return {"task_id": record["task_id"], "instance_id": record["spec"]["instance_id"],
            "seed": record["spec"]["seed"], "status": record["status"],
            "valid_protocol": valid, "parse_failure": failure, **view, **score(view, reference),
            "executed": view["decision"] == "EXECUTE", "abstained": view["decision"] == "ABSTAIN",
            "elapsed_seconds": record.get("elapsed_seconds"),
            "raw_starts_fence": (record.get("response") or {}).get("content", "").lstrip().startswith("```")}
