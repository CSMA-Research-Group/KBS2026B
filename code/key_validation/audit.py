"""Audit saved decisions against checked proofs and a separately built reference."""
from collections import Counter

from e3_cpu_correctness.core import verify_certificate
from .cases import bounds
from .proofs import check_closure, check_witness


def audit_decision(public, result, reference):
    checks = []
    def check(name, value):
        checks.append({"check": name, "pass": bool(value)})
    families = {f["id"]: f for f in public["families"]}
    configs = {c["id"]: c for c in public["world"]["configurations"]}
    verified_goals, positives = set(), Counter()
    for witness in result["witnesses"]:
        goal = witness["goal"]
        valid = check_witness(public, goal, witness["path"])["valid"]
        check("positive_replay:" + goal["goal_id"], valid)
        check("positive_reference:" + goal["goal_id"], goal["goal_id"] in reference["feasible_goal_ids"])
        check("distinct_positive:" + goal["goal_id"], goal["goal_id"] not in verified_goals)
        verified_goals.add(goal["goal_id"])
        family_id = goal["goal_id"].rsplit(":", 1)[0]
        positives[family_id] += 1
    justified = {fid for fid, n in positives.items() if fid in families and n == families[fid]["count"]}
    closed_ids = []
    for proof in result["closures"]:
        cid = proof["configuration_id"]
        valid = cid in configs and check_closure(public["world"], configs[cid], proof)["valid"]
        check("concrete_closure:" + cid, valid)
        closed_ids.append(cid)
        justified.update(f["id"] for f in families.values() if f["configuration_id"] == cid)
    check("distinct_closures", len(closed_ids) == len(set(closed_ids)))
    certificate = result["certificate"]
    if certificate is None:
        check("no_exclusion_without_certificate", not result["abstract_exclusions"])
    else:
        check("abstract_certificate", verify_certificate(public["world"], certificate)["valid"])
        nodes = {tuple(p) for p in certificate["nodes"]}
        check("abstract_covers_all_reference_sources_and_reachable_poses",
              all(tuple(p[:2]) in nodes for c in reference["closures"] for p, _ in c["parents"]))
        for fid in result["abstract_exclusions"]:
            valid = fid in families
            if valid:
                a, b, c, d = bounds(families[fid])
                valid = not any(a <= x <= c and b <= y <= d for x, y in nodes)
            check("family_exclusion:" + fid, valid)
            if valid:
                justified.add(fid)
    resolved = result["resolved_families"]
    check("only_justified_resolutions", len(resolved) == len(set(resolved)) and set(resolved) <= justified)
    decision = result["decision"]
    check("known_decision", decision in ("EXECUTE", "CLARIFY", "REJECT", "ABSTAIN"))
    if decision == "EXECUTE":
        check("unique_checked_goal", len(verified_goals) == 1 and result["selected_goal"] in verified_goals)
        check("all_competitors_resolved", set(resolved) == set(families))
        check("unique_reference_goal", reference["feasible_goal_ids"] == [result["selected_goal"]])
    elif decision == "CLARIFY":
        check("two_distinct_positive_goals", len(verified_goals) >= 2)
    elif decision == "REJECT":
        check("no_positive_and_all_resolved", not verified_goals and set(resolved) == set(families))
    if decision != "ABSTAIN":
        check("reference_decision", decision == reference["expected_decision"])
        check("complete_termination", result["termination"] == "checked_evidence_complete")
        check("within_cpu_budget", result["budget_seconds"] is None or result["policy_cpu_seconds"] <= result["budget_seconds"])
    else:
        check("no_abstention_selection", result["selected_goal"] is None)
    check("nonnegative_costs", all(type(v) is int and v >= 0 for v in result["cost"].values()))
    return {"status": "PASS" if all(r["pass"] for r in checks) else "FAIL", "checks": checks,
            "correct_decision": decision == reference["expected_decision"],
            "abstained": decision == "ABSTAIN", "execution_error": decision == "EXECUTE" and
                (reference["expected_decision"] != "EXECUTE" or result["selected_goal"] not in reference["feasible_goal_ids"])}
