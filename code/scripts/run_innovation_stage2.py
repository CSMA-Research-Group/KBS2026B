#!/usr/bin/env python3
"""Automatically labelled carrying-domain development experiment; CPU only."""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import random
import statistics
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "code"))
from innovation_stage2.domain import evaluate_ast, goal_id, parse_families, validate_witness
from innovation_stage2.generator import generate_cases
from innovation_stage2.reference import reference_oracle
from innovation_stage2.policies import METHODS, run_policy
from src.utils.io import file_sha256, stable_hash, run_timestamp, utc_now, write_json

OUTPUT_ROOT = ROOT / "data/results/innovation_stage2"
CONFIG_PATH = ROOT / "code/configs/innovation_stage2_dev.json"
SCOPE = {"stage": "dev", "experiment": "stage2", "diagnostic_only": True, "calibrated": False,
         "domain": "finite_center_anchored_rectangular_carry_with_two_orientations",
         "annotation_source": "program_AST_and_independent_augmented_state_search",
         "model_calls": 0, "gpu_processes_started": 0, "network_requests": 0,
         "synthetic_timeout": False, "physical_execution_measured": False,
         "shared_configuration_graph": True,
         "candidate_scope": "finite parsed source-target families; audit output materializes goal states; no open-language completeness",
         "budget_units": "source_checks_plus_empty_and_loaded_configuration_node_expansions",
         "cost_scope": "collision cells, transition checks and graph traversal separately counted; unit weights do not denote equal hardware time",
         "claim_scope": "coupled-domain development prototype; not a validated novel algorithm, calibrated risk bound or continuous physics evaluation"}


def prepare_plan(n_cases=None, output=None):
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    if n_cases is not None:
        config["n_cases"] = n_cases
    if config["stage"] != "dev" or config["model_calls"] != 0:
        raise ValueError("Only CPU development is supported")
    if type(config["n_cases"]) is not int or not 1 <= config["n_cases"] <= 96:
        raise ValueError("n_cases must be 1..96; 96 is the complete development recipe")
    for key in ("data_seed", "policy_seed", "execution_order_seed"):
        if type(config[key]) is not int or not 0 <= config[key] < 2**32:
            raise ValueError("Invalid seed: " + key)
    if type(config["reference_max_nodes"]) is not int or config["reference_max_nodes"] <= 0:
        raise ValueError("reference_max_nodes must be positive")
    budgets = config["budgets"]
    if None not in budgets or len(set(budgets)) != len(budgets) or any(b is not None and (type(b) is not int or b < 1) for b in budgets):
        raise ValueError("Distinct positive budgets and a full-search control are required")
    target = Path(output) if output else OUTPUT_ROOT / (run_timestamp() + "_" + uuid.uuid4().hex[:8])
    target = (target if target.is_absolute() else ROOT / target).resolve()
    if target.exists() or target == OUTPUT_ROOT.resolve() or not target.is_relative_to(OUTPUT_ROOT.resolve()):
        raise ValueError("Output must be a NEW directory below data/results/innovation_stage2")
    paths = sorted((ROOT / "code/innovation_stage2").glob("*.py")) + [
        ROOT / "code/innovation_stage1/__init__.py", ROOT / "code/innovation_stage1/domain.py",
        ROOT / "code/src/utils/io.py", ROOT / "code/src/__init__.py", Path(__file__).resolve(), CONFIG_PATH]
    hashes = {str(p.relative_to(ROOT)): file_sha256(p) for p in paths}
    return {**SCOPE, "status": "CHECKED_ONLY", "config": config, "output": str(target),
            "source_file_hashes": hashes, "source_tree_sha256": stable_hash(hashes),
            "config_file_sha256": file_sha256(CONFIG_PATH), "resolved_config_sha256": stable_hash(config),
            "methods": list(METHODS), "expected_cases": config["n_cases"],
            "expected_decisions": config["n_cases"] * len(METHODS) * len(budgets),
            "complete_design": config["n_cases"] == 96, "policies_executed": False}


def score_prediction(prediction, runtime, oracle):
    action = prediction["decision"]
    if action not in {"EXECUTE", "CLARIFY", "REJECT", "ABSTAIN"}:
        raise ValueError("Unsupported decision")
    complete = bool(oracle["complete"])
    nonempty = bool(oracle["semantic_candidates"])
    scorable = complete and nonempty
    feasible = set(oracle["feasible_goals"])
    expected = ("EXECUTE" if len(feasible) == 1 else "CLARIFY" if feasible else "REJECT") if scorable else None
    candidate = prediction.get("selected_candidate")
    legal = isinstance(candidate, dict) and candidate in oracle["semantic_candidates"]
    valid = bool(action == "EXECUTE" and legal and goal_id(candidate) == prediction.get("selected_goal")
                 and validate_witness(runtime, candidate, prediction.get("witness", [])))
    correct = bool(scorable and action == expected == "EXECUTE" and prediction.get("selected_goal") in feasible and valid)
    wrong = (not correct if action == "EXECUTE" else action != expected) if scorable and action != "ABSTAIN" else None
    return {"reference_complete": complete, "semantic_nonempty": nonempty, "scorable": scorable,
            "reference_decision": expected, "n_feasible_gt": len(feasible) if complete else None,
            "witness_valid": valid, "selected_candidate_semantically_legal": legal,
            "correct_autonomous_execute": correct, "execution_error": (not correct) if scorable and action == "EXECUTE" else None,
            "decision_error": wrong, "correct_resolved_decision": scorable and action != "ABSTAIN" and wrong is False}


def validate_prediction(prediction, runtime, oracle, budget):
    cost = prediction["cost"]
    keys = ("source_checks", "empty_node_expansions", "loaded_node_expansions", "total_units")
    if any(type(cost[k]) is not int or cost[k] < 0 for k in keys):
        raise ValueError("Invalid operation count")
    if cost["total_units"] != sum(cost[k] for k in keys[:-1]):
        raise ValueError("Operation totals do not add up")
    if budget is not None and cost["total_units"] > budget:
        raise ValueError("Budget exceeded")
    trace = prediction["trace"]
    events = Counter(event["operation"] for event in trace)
    if set(events) - {"source_check", "expand_empty", "expand_loaded"}:
        raise ValueError("Unknown operation in budget trace")
    if (any(event["cost_units"] != 1 for event in trace) or len(trace) != cost["total_units"]
            or events["source_check"] != cost["source_checks"] or events["expand_empty"] != cost["empty_node_expansions"]
            or events["expand_loaded"] != cost["loaded_node_expansions"]):
        raise ValueError("Operation trace/count mismatch")
    world_hash = stable_hash(runtime)
    if prediction.get("world_sha256") != world_hash or any(event.get("world_sha256") != world_hash for event in trace):
        raise ValueError("Evidence is not bound to the current public world")
    checked = [event["source_id"] for event in trace if event["operation"] == "source_check"]
    expanded = {(event["scope"], tuple(event["node"])): event for event in trace if event["operation"] != "source_check"}
    if len(set(checked)) != len(checked) or len(expanded) != len(trace) - len(checked):
        raise ValueError("Duplicate source check or configuration expansion despite shared cache")
    for cert in prediction["source_certificates"]:
        if cert["world_sha256"] != world_hash or cert["source_id"] not in checked:
            raise ValueError("Source certificate has no bound paid query")
    component_ids = set()
    for cert in prediction["closed_components"]:
        if cert["world_sha256"] != world_hash or not cert["closed"] or cert["component_id"] in component_ids:
            raise ValueError("Invalid closed-component binding")
        component_ids.add(cert["component_id"])
        nodes = {tuple(node) for node in cert["nodes"]}
        if not nodes or len(nodes) != len(cert["nodes"]) or cert["expanded_nodes"] != len(nodes):
            raise ValueError("Closed component node count mismatch")
        for node in nodes:
            event = expanded.get((cert["scope"], node))
            if event is None or any(tuple(neighbor) not in nodes for neighbor in event["neighbors"]):
                raise ValueError("A closed-component certificate leaves an unexpanded boundary")
    for key in ("collision_cell_checks", "transition_checks"):
        if type(cost[key]) is not int or cost[key] < 0:
            raise ValueError("Missing/invalid physical-work counter: " + key)
    if not isinstance(prediction["wall_seconds"], (int, float)) or not 0 <= prediction["wall_seconds"] < float("inf"):
        raise ValueError("Invalid wall time")
    states = prediction["goal_states"]
    if set(states) != {goal_id(c) for c in oracle["semantic_candidates"]}:
        raise ValueError("Semantic candidates were lost or invented")
    if any(value not in {"FEASIBLE", "INFEASIBLE", "UNKNOWN"} for value in states.values()):
        raise ValueError("Invalid evidence state")
    feasible = {goal for goal, value in states.items() if value == "FEASIBLE"}
    decision = prediction["decision"]
    if decision == "EXECUTE" and (len(feasible) != 1 or "UNKNOWN" in states.values() or prediction["selected_goal"] not in feasible):
        raise ValueError("Execution lacks exclusion of every competitor")
    if decision == "CLARIFY" and len(feasible) < 2:
        raise ValueError("Clarification lacks two feasible goal classes")
    if decision == "REJECT" and (not states or any(value != "INFEASIBLE" for value in states.values())):
        raise ValueError("Rejection lacks nonempty complete exclusion")
    if not states and decision != "ABSTAIN":
        raise ValueError("Empty semantics must remain a separate abstention")
    if oracle["complete"]:
        gt = set(oracle["feasible_goals"])
        if any((value == "FEASIBLE" and goal not in gt) or (value == "INFEASIBLE" and goal in gt) for goal, value in states.items()):
            raise ValueError("Evidence contradicts the independent state oracle")
    score = score_prediction(prediction, runtime, oracle)
    if score["decision_error"] is True or decision == "EXECUTE" and not score["witness_valid"]:
        raise ValueError("Incorrect certified decision or invalid carrying witness")
    if budget is None and states and decision == "ABSTAIN":
        raise ValueError("Full budget left a finite supported nonempty task unresolved")
    return score


def metrics(rows):
    evaluated = [r for r in rows if r["scoring"]["scorable"]]
    n = len(evaluated)
    counts = Counter(r["decision"] for r in evaluated)
    correct = sum(r["scoring"]["correct_autonomous_execute"] for r in evaluated)
    ratio = lambda a, b: a / b if b else None
    result = {"n_records": len(rows), "n_unique_scenes": len({r["cluster_id"] for r in rows}),
              "n_scorable": n, "n_reference_incomplete": sum(not r["scoring"]["reference_complete"] for r in rows),
              "n_empty_semantics": sum(not r["scoring"]["semantic_nonempty"] for r in rows),
              "decision_counts_scorable": dict(counts), "decision_counts_all": dict(Counter(r["decision"] for r in rows)),
              "A_correct": ratio(correct, n), "R_exec": ratio(sum(r["scoring"]["execution_error"] is True for r in evaluated), counts["EXECUTE"]),
              "Exec_S": ratio(correct, sum(r["scoring"]["reference_decision"] == "EXECUTE" for r in evaluated)),
              "resolved_rate": ratio(sum(r["scoring"]["correct_resolved_decision"] for r in evaluated), n),
              "mean_wall_seconds": statistics.fmean(r["wall_seconds"] for r in rows) if rows else None,
              "mean_candidate_count": statistics.fmean(r["candidate_count"] for r in rows) if rows else None}
    for action in ("CLARIFY", "REJECT", "ABSTAIN"):
        result[action.lower() + "_rate"] = ratio(counts[action], n)
    for key in ("source_checks", "empty_node_expansions", "loaded_node_expansions", "total_units", "collision_cell_checks", "transition_checks", "graph_traversals"):
        values = [r["cost"][key] for r in rows if key in r["cost"]]
        result["mean_" + key] = statistics.fmean(values) if len(values) == len(rows) and values else None
    return result


def summarize(rows, cases, oracles, config):
    result = {**SCOPE, "status": "COMPLETE", "n_cases": len(cases), "n_decisions": len(rows),
              "complete_design": len(cases) == 96, "config": config,
              "reference": {"complete_cases": sum(o["complete"] for o in oracles.values()),
                            "unresolved_ids": [key for key, o in oracles.items() if not o["complete"]],
                            "empty_semantic_cases": sum(not o["semantic_candidates"] for o in oracles.values()),
                            "label_counts": dict(Counter(o["label"] for o in oracles.values() if o["complete"]))},
              "by_budget": {}, "by_stratum": {}, "paired_family_comparisons": {}}
    family_method = "FAMILY_PRIORITY_SHARED_CONFIG"
    for budget in config["budgets"]:
        key = "full" if budget is None else str(budget)
        chosen = [r for r in rows if r["budget"] == budget]
        result["by_budget"][key] = {method: metrics([r for r in chosen if r["method"] == method]) for method in METHODS}
        indexed = {(r["instance_id"], r["method"]): r for r in chosen}
        result["paired_family_comparisons"][key] = {}
        for baseline in METHODS:
            if baseline == family_method:
                continue
            diffs, transitions = [], Counter()
            for case in cases:
                own, base = indexed[(case["instance_id"], family_method)], indexed[(case["instance_id"], baseline)]
                transitions[base["decision"] + "->" + own["decision"]] += 1
                diffs.append({"instance_id": case["instance_id"], "scorable": own["scoring"]["scorable"],
                              "saved_total_units": base["cost"]["total_units"] - own["cost"]["total_units"],
                              "saved_loaded_expansions": base["cost"]["loaded_node_expansions"] - own["cost"]["loaded_node_expansions"],
                              "saved_collision_cells": base["cost"]["collision_cell_checks"] - own["cost"]["collision_cell_checks"],
                              "saved_wall_seconds": base["wall_seconds"] - own["wall_seconds"],
                              "correct_execute_gain": (int(own["scoring"]["correct_autonomous_execute"]) - int(base["scoring"]["correct_autonomous_execute"])) if own["scoring"]["scorable"] else None})
            result["paired_family_comparisons"][key][baseline] = {
                "action_transitions_baseline_to_family": dict(transitions),
                "cost_wins": sum(d["saved_total_units"] > 0 for d in diffs),
                "cost_ties": sum(d["saved_total_units"] == 0 for d in diffs),
                "cost_losses": sum(d["saved_total_units"] < 0 for d in diffs),
                "case_differences": diffs}
    for field in cases[0]["stratum"]:
        result["by_stratum"][field] = {
            value: {("full" if b is None else str(b)): {m: metrics([r for r in rows if r["budget"] == b and r["method"] == m
                        and str(r["stratum"][field]) == value]) for m in METHODS} for b in config["budgets"]}
            for value in sorted({str(c["stratum"][field]) for c in cases})}
    degrees = []
    for case in cases:
        families = parse_families(case["runtime"])
        degrees.append({"instance_id": case["instance_id"], "family_sizes": [len(f["target_ids"]) for f in families]})
    result["family_design"] = {"unequal_nonempty_family_sizes_cases": sum(len(set(d["family_sizes"])) > 1 for d in degrees),
                                "per_case": degrees,
                                "scope": "initial semantic family sizes; actual scheduling discrimination requires trace analysis"}
    result["invariants"] = {"status": "PASS", "checked_predictions": len(rows),
                             "checks": ["parser_AST_semantics", "independent_reference_witness_replay", "complete_schedule", "runtime_unchanged",
                                        "paid_operation_trace", "three_way_evidence_sound", "certified_decision", "carrying_witness", "full_budget_resolved"]}
    result["statistics_scope"] = "Development scenes equally weighted. Truth metrics require complete nonempty reference; empty semantics and incomplete references remain separate. No seed inflation or learned/risk guarantee."
    return result


def markdown(summary):
    lines = ["# Stage 2：携物配置搜索与候选族证据开发实验", "",
             f"状态：{summary['status']}；场景 {summary['n_cases']}；决策 {summary['n_decisions']}；完整配方：{summary['complete_design']}。", "",
             "CPU、程序真值；携物足迹和朝向改变通行能力。所有策略共享同样的按足迹配置图与闭合分量证据。", "",
             "| 预算 | 方法 | 正确自主率 | 充分任务执行率 | 已解决率 | 执行错误率 | 源检查 | 空手展开 | 携物展开 | 总单位 | 碰撞单元检查 | 时间 ms |",
             "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for budget, methods in summary["by_budget"].items():
        for method, m in methods.items():
            values = ["NA" if m[k] is None else f"{m[k]:.4f}" for k in
                      ("A_correct", "Exec_S", "resolved_rate", "R_exec", "mean_source_checks", "mean_empty_node_expansions", "mean_loaded_node_expansions", "mean_total_units", "mean_collision_cell_checks")]
            lines.append("| " + budget + " | " + method + " | " + " | ".join(values) + f" | {m['mean_wall_seconds']*1000:.3f} |")
    lines += ["", "参考状态：" + json.dumps(summary["reference"], ensure_ascii=False), "",
              f"初始非空候选族规模不等的场景：{summary['family_design']['unequal_nonempty_family_sizes_cases']} / {summary['n_cases']}。", "",
              "以上为开发结果，不能据运行完成宣称新算法成立。空语义或参考未完成分别保留；真值指标分母见 summary.json。", "",
              "预算是源检查+空手/携物配置节点展开的声明单位。真实碰撞单元检查、转移检查、图遍历及wall time分项保存；没有人为sleep或模拟timeout。", "",
              "静态障碍、中心锚点矩形、两种离散朝向，旋转按声明的方框占据规则检查；对象/目标是地面标记。这不是连续旋转扫掠或真实机器人执行保证。", "",
              "候选族仍来自有限全文语法；审计输出显式展开goal_states，不声称开放语言覆盖或已实现指数候选空间的隐式求解。", "",
              "比较 FAMILY 与条件优先、简单混合、交错及完整共享基线。提前澄清/拒绝、自主执行、操作数量和实际耗时必须分别解释。", ""]
    return "\n".join(lines)


def run_stage2(n_cases=None, output=None, *, check_only=False):
    plan = prepare_plan(n_cases, output)
    if check_only:
        return plan
    target = Path(plan["output"])
    target.mkdir(parents=True, exist_ok=False)
    manifest = {**plan, "status": "RUNNING", "started_at": utc_now(), "files": {}}
    write_json(target / "manifest.json", manifest)
    started = time.perf_counter()
    rows, oracles = [], {}
    try:
        snapshot = {name: (ROOT / name).read_text(encoding="utf-8") for name in plan["source_file_hashes"]}
        if any(hashlib.sha256(text.encode()).hexdigest() != plan["source_file_hashes"][name] for name, text in snapshot.items()):
            raise ValueError("Source changed after preparation")
        write_json(target / "source_snapshot.json", {"files": snapshot, "file_hashes": plan["source_file_hashes"]})
        config = plan["config"]
        cases = generate_cases(config["n_cases"], config["data_seed"])
        if len(cases) != config["n_cases"] or len({c["instance_id"] for c in cases}) != len(cases) or len({c["cluster_id"] for c in cases}) != len(cases):
            raise ValueError("Wrong scene count or duplicate scene/instance IDs")
        if any(c["split"] != "dev" for c in cases):
            raise ValueError("Only development data are supported")
        with (target / "runtime.jsonl").open("x", encoding="utf-8") as handle:
            for case in cases:
                handle.write(json.dumps({"instance_id": case["instance_id"], "runtime": case["runtime"]}, ensure_ascii=False) + "\n")
        with (target / "ground_truth.jsonl").open("x", encoding="utf-8") as handle:
            for index, case in enumerate(cases):
                runtime, ast = case["runtime"], case["ast"]
                semantic = evaluate_ast(ast, runtime)
                families = parse_families(runtime)
                parsed = {goal_id({"action": "place", "source_id": f["source_id"], "target_id": t}) for f in families for t in f["target_ids"]}
                by_goal = {goal_id(c): c for c in semantic}
                if set(by_goal) != parsed:
                    raise ValueError("Public family parser disagrees with AST semantics")
                oracle = reference_oracle(runtime, ast, max_nodes=config["reference_max_nodes"])
                if set(by_goal) != {goal_id(c) for c in oracle["semantic_candidates"]}:
                    raise ValueError("Reference semantic set mismatch")
                if set(oracle["feasible_goals"]) != set(oracle["witnesses"]):
                    raise ValueError("Reference positives lack witnesses")
                if any(goal not in by_goal or not validate_witness(runtime, by_goal[goal], actions) for goal, actions in oracle["witnesses"].items()):
                    raise ValueError("Independent reference witness cannot be replayed")
                oracles[case["instance_id"]] = oracle
                handle.write(json.dumps({k: v for k, v in case.items() if k != "runtime"} | {"reference": oracle}, ensure_ascii=False) + "\n")
                handle.flush()
                if (index + 1) % 8 == 0 or index + 1 == len(cases):
                    print(f"携物参考求解：{index + 1}/{len(cases)}", flush=True)
        manifest.update(phase="POLICIES", policies_executed=True)
        write_json(target / "manifest.json", manifest)
        order_rng = random.Random(config["execution_order_seed"])
        with (target / "decisions.jsonl").open("x", encoding="utf-8") as handle:
            for index, case in enumerate(cases):
                for budget in config["budgets"]:
                    order = list(METHODS)
                    order_rng.shuffle(order)
                    for position, method in enumerate(order):
                        runtime = copy.deepcopy(case["runtime"])
                        runtime_hash = stable_hash(runtime)
                        prediction = run_policy(runtime, method, budget, seed=config["policy_seed"])
                        row = {**prediction, "instance_id": case["instance_id"], "cluster_id": case["cluster_id"], "split": "dev",
                               "stratum": case["stratum"], "execution_order_index": position, "runtime_input_sha256": runtime_hash}
                        try:
                            if stable_hash(runtime) != runtime_hash:
                                raise ValueError("Policy changed public runtime")
                            if prediction["method"] != method or prediction["budget"] != budget or prediction["seed"] != config["policy_seed"]:
                                raise ValueError("Prediction provenance mismatch")
                            row["scoring"] = validate_prediction(prediction, runtime, oracles[case["instance_id"]], budget)
                        except Exception as exc:
                            row["validation_error"] = str(exc)
                            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                            handle.flush()
                            raise
                        rows.append(row)
                        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
                handle.flush()
                if (index + 1) % 8 == 0 or index + 1 == len(cases):
                    print(f"共享配置策略比较：{index + 1}/{len(cases)}，已完成 {len(rows)} 条决策", flush=True)
        expected = {(c["instance_id"], b, m) for c in cases for b in config["budgets"] for m in METHODS}
        actual = {(r["instance_id"], r["budget"], r["method"]) for r in rows}
        if actual != expected or len(rows) != len(expected) or len(rows) != plan["expected_decisions"]:
            raise ValueError("Incomplete or duplicated method schedule")
        summary = summarize(rows, cases, oracles, config)
        write_json(target / "summary.json", summary)
        (target / "SUMMARY.md").write_text(markdown(summary), encoding="utf-8")
        manifest.update(status="COMPLETE", phase="DONE", n_decisions=len(rows), reference=summary["reference"], invariants=summary["invariants"])
    except BaseException as exc:
        manifest.update(status="FAILED", n_decisions=len(rows), error_category=type(exc).__name__, error=str(exc))
        raise
    finally:
        manifest.update(finished_at=utc_now(), elapsed_seconds=time.perf_counter() - started)
        manifest["files"] = {p.name: file_sha256(p) for p in sorted(target.iterdir()) if p.is_file() and p.name != "manifest.json"}
        write_json(target / "manifest.json", manifest)
    return {**SCOPE, "status": "COMPLETE", "output": str(target), "n_cases": len(cases), "n_decisions": len(rows),
            "summary": str(target / "SUMMARY.md"), "manifest": str(target / "manifest.json")}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-cases", type=int, help="1..96; fewer than 96 is an engineering prefix")
    parser.add_argument("--output", help="New directory under data/results/innovation_stage2")
    parser.add_argument("--check-only", action="store_true", help="Read-only configuration/source check; no data or policies")
    args = parser.parse_args(argv)
    try:
        result = run_stage2(args.n_cases, args.output, check_only=args.check_only)
    except Exception as exc:
        print(json.dumps({"status": "FAILED", "error_category": type(exc).__name__, "error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
