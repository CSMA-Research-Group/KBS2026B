"""Fixed goal order with FIFO or greedy best-first paid configuration queries.

The two Stage 2c controls delegate without changes. Four goal policies cross
forward/reversed public pair order with FIFO/GBFS, after identical paid target
and source prefixes. Each same-order pair changes only frontier selection.
GBFS uses h alone, not g+h: this is not A*, and has no shortest-path or
minimum-checker-work claim.
"""
from __future__ import annotations

import copy
import time
from typing import Any

from innovation_stage2.domain import goal_id, parse_families, validate_witness
from innovation_stage2.policies import _decision
from innovation_stage2b.policies import _graph_query, _source_query
from innovation_stage2c.policies import _TargetEngine, _execute_query, _metadata, _schedule_event
from innovation_stage2c.policies import run_policy as run_stage2c_policy


METHODS = ("CHEAP_FIRST_TARGET", "COST_MENU_TARGET", "GOAL_FIFO_LEX_TARGET", "GOAL_GBFS_LEX_TARGET",
           "GOAL_FIFO_REVERSE_TARGET", "GOAL_GBFS_REVERSE_TARGET")
CONTROL_METHODS = METHODS[:2]
GOAL_METHODS = METHODS[2:]


def _goal_metadata(method):
    result = _metadata("CHEAP_FIRST_TARGET")
    order = "reverse" if "_REVERSE_" in method else "lex"
    search = "FIFO" if "_FIFO_" in method else "GBFS"
    result.update({
        "method": method, "policy_stage": "stage2d",
        "base_method": "FIXED_GOAL_" + order.upper() + "_" + search, "goal_order": order,
        "policy_kind": "fixed_goal_order_FIFO_or_greedy_best_first_control; no_learning_or_novelty_claim",
        "goal_order_semantics": "After common target/source prefixes choose the first UNKNOWN pair in sorted(source_id,target_id) order or its exact reverse; EMPTY path to that source, then loaded path to that target. Reverse changes only pair order. No alternate goal if its next query is unaffordable.",
        "source_prefix_semantics": "After target prefix, all pending sources with at least one UNKNOWN goal are checked in source-ID order, choosing first affordable. If pending source queries all exceed remaining budget, stop with source_prefix_budget_blocked. Same for all four goal policies, independent of goal order.",
        "frontier_selection_semantics": "FIFO selects earliest discovered unexpanded node of the current known component. GBFS selects min(h,discovery_order,node) over that same frontier; h is Manhattan to the current source for EMPTY, or Manhattan to target plus 0/1 disallowed-orientation indicator for loaded. Equal-h ties agree with FIFO. GBFS is h-only, not A*, and claims no path-length/checker-budget optimality.",
        "frontier_selection_cost_scope": "Frontier scans, distance/orientation arithmetic, cached-graph bookkeeping and witness routing are included in wall_seconds, excluded from the separately declared collision/source checker-work budget.",
        "target_precheck_semantics": "Unchanged Stage2c paid public terminal-pose prefix; first affordable canonical key, then fixed goal scheduler when no remaining target query fits. No source-eligibility filtering or graph warm-start.",
    })
    return result


def _heuristic(node, destination, orientations=None):
    return abs(node[0] - destination[0]) + abs(node[1] - destination[1]) + (
        int(node[2] not in orientations) if orientations is not None else 0)


def _goal_menu(engine, method, states, remaining):
    """Select an active goal and reconstruct its legal paid next query."""
    pairs = [(source, target) for source in engine.sources for target in engine.families[source]
             if states[goal_id({"action": "place", "source_id": source, "target_id": target})] == "UNKNOWN"]
    source, target = pairs[-1] if "_REVERSE_" in method else pairs[0]
    active = goal_id({"action": "place", "source_id": source, "target_id": target})
    counts = engine.unknown_counts(states)
    if source not in engine.certificates or not engine.certificates[source]["passed"]:
        raise RuntimeError("An active UNKNOWN goal requires its paid passing source-prefix certificate")
    if not engine.source_reached(source):
        branch, graph, origin = "GOAL_EMPTY_ROUTE", engine.empty, engine.start
        destination, orientations = engine.objects[source]["cell"], None
    else:
        branch, graph, origin = "GOAL_LOADED_ROUTE", engine.graph(source), engine.source_node(source)
        destination, orientations = engine.targets[target]["cell"], engine.targets[target]["allowed_orientations"]
    root = graph.root(origin)
    frontier = graph.members[root] - graph.expanded
    if not frontier:
        raise RuntimeError("An UNKNOWN active goal lacks an unexpanded evidence frontier")
    frontier_size = len(frontier)
    if "_FIFO_" in method:
        node, rule = graph.frontier(origin), "FIFO_DISCOVERY"
    else:
        node = min(frontier, key=lambda node: (_heuristic(node, destination, orientations), graph.order[node], node))
        rule = "GBFS_MANHATTAN_ORIENTATION"
    h = _heuristic(node, destination, orientations)
    # Retain the original meaning: all unresolved goals currently covered
    # by this paid eligible source/component, not a new value estimate.
    affected = sum(affected for job_graph, job_node, affected in engine.search_jobs(counts)
                   if job_graph.scope == graph.scope and job_graph.root(job_node) == root)
    query = _graph_query(graph, node, affected)
    query.update({"active_goal": active, "goal_source_id": source, "goal_target_id": target,
                  "frontier_rule": rule, "heuristic_value": h, "frontier_size": frontier_size,
                  "score_numerator": 1, "score_denominator": 1, "priority_score": 1.0,
                  "affordable": remaining is None or query["reserved_work_units"] <= remaining})
    return branch, [query]


def _source_prefix_menu(engine, states, remaining):
    counts, result = engine.unknown_counts(states), []
    for source in engine.sources:
        if source in engine.certificates or counts[source] == 0:
            continue
        query = _source_query(engine, source, counts[source])
        query.update({"active_goal": None, "goal_source_id": source, "goal_target_id": None,
                      "frontier_rule": "source_condition_prefix", "heuristic_value": None, "frontier_size": 0,
                      "score_numerator": 1, "score_denominator": 1, "priority_score": 1.0,
                      "affordable": remaining is None or query["reserved_work_units"] <= remaining})
        result.append(query)
    return result


def run_policy(runtime: dict[str, Any], method: str, budget: int | None, seed: int = 101) -> dict[str, Any]:
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError("Unknown Stage2d policy: " + str(method))
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be a nonnegative prepaid-work integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if method in CONTROL_METHODS:
        return run_stage2c_policy(runtime, method, budget, seed)
    parse_started = time.perf_counter()
    try:
        families = parse_families(runtime)
    except ValueError:
        result = run_stage2c_policy(runtime, "CHEAP_FIRST_TARGET", budget, seed)
        result.update(_goal_metadata(method))
        result.update({"source_prefix_complete": False, "source_prefix_budget_blocked": False})
        result["wall_seconds"] = time.perf_counter() - started
        return result
    parse_seconds = time.perf_counter() - parse_started
    engine = _TargetEngine(runtime, families)
    reserved, schedule = 0, []
    prefix_active, prefix_budget_skipped, budget_blocked = True, False, False
    source_prefix_active, source_prefix_budget_blocked = True, False
    while True:
        states, reasons = engine.states()
        decision, _ = _decision(states)
        if not states:
            stop_reason = "EMPTY_SEMANTICS"
            break
        if decision != "ABSTAIN":
            stop_reason = "decision_certified"
            break
        remaining = None if budget is None else budget - reserved
        if prefix_active:
            ranked = engine.target_menu(states, remaining)
            if not ranked:
                prefix_active = False
            else:
                chosen = next((query for query in ranked if query["affordable"]), None)
                event = _schedule_event(engine, schedule, states, remaining, reserved, ranked, chosen,
                                        "FIXED_TARGET_TERMINAL_PRECHECK")
                if chosen is not None:
                    reserved = _execute_query(engine, chosen, event, reserved, budget)
                    schedule.append(event)
                    continue
                event.update({"outcome": "precheck_budget_skipped_continue_base", "reserved_work_delta": 0,
                              "actual_checker_work_delta": 0, "remaining_budget_after": remaining})
                schedule.append(event)
                prefix_active, prefix_budget_skipped = False, True
        if source_prefix_active:
            ranked = _source_prefix_menu(engine, states, remaining)
            if not ranked:
                source_prefix_active = False
            else:
                chosen = next((query for query in ranked if query["affordable"]), None)
                event = _schedule_event(engine, schedule, states, remaining, reserved, ranked, chosen,
                                        "SHARED_SOURCE_CONDITION_PREFIX")
                if chosen is not None:
                    reserved = _execute_query(engine, chosen, event, reserved, budget)
                    schedule.append(event)
                    continue
                event.update({"outcome": "budget_blocked", "reserved_work_delta": 0,
                              "actual_checker_work_delta": 0, "remaining_budget_after": remaining})
                schedule.append(event)
                budget_blocked, source_prefix_budget_blocked = True, True
                stop_reason = "source_prefix_budget_blocked"
                break
        branch, ranked = _goal_menu(engine, method, states, remaining)
        chosen = ranked[0] if ranked[0]["affordable"] else None
        event = _schedule_event(engine, schedule, states, remaining, reserved, ranked, chosen, branch)
        if chosen is None:
            event.update({"outcome": "budget_blocked", "reserved_work_delta": 0,
                          "actual_checker_work_delta": 0, "remaining_budget_after": remaining})
            schedule.append(event)
            budget_blocked, stop_reason = True, "budget_no_affordable_query"
            break
        reserved = _execute_query(engine, chosen, event, reserved, budget)
        schedule.append(event)
    states, reasons = engine.states()
    decision, selected_goal = _decision(states)
    selected = None
    if selected_goal is not None:
        selected = next({"action": "place", "source_id": source, "target_id": target}
                        for source in engine.sources for target in engine.families[source]
                        if goal_id({"action": "place", "source_id": source, "target_id": target}) == selected_goal)
    witness = engine.witness(selected, reasons[selected_goal]) if selected is not None else []
    if selected is not None and not validate_witness(runtime, selected, witness):
        raise RuntimeError("Goal-directed policy produced an invalid carrying witness")
    graphs = [engine.empty] + [engine.loaded[key] for key in sorted(engine.loaded)]
    closed = [certificate for graph in graphs for certificate in graph.closed_certificates()]
    return {**_goal_metadata(method), "budget": budget, "seed": seed,
            "budget_semantics": "reserved_checker_work_units; no refunds",
            "cost_semantics": "Unchanged Stage2c: source=1+area, EMPTY=4, loaded=5wh+2(w+h)+max(w,h)^2, target=area*allowed orientations. actual_checker_work_units=source_checks+collision_cell_checks; bookkeeping, heuristic arithmetic and final witness replay are wall time only.",
            "candidate_catalog_materialized": False, "audit_goal_states_materialized": True,
            "runtime_candidate_representation": "source_target_families_with_explicit_target_ids",
            "canonical_tie_semantics": "target prefix (-1,width,height,target_id); source prefix by source ID; goals (source_id,target_id) or reverse; FIFO earliest graph discovery; GBFS (h,discovery_order,node) inside active source component",
            "supported": True, "reason": stop_reason, "decision": decision,
            "selected_goal": selected_goal, "selected_candidate": selected, "witness": witness,
            "goal_states": states, "goal_evidence": reasons, "families": copy.deepcopy(families),
            "candidate_count": len(states), "family_count": len(engine.families),
            "trace": engine.trace, "schedule_trace": schedule,
            "source_certificates": list(engine.certificates.values()),
            "target_certificates": list(engine.target_certificates.values()),
            "target_keys_total": len(engine.target_keys),
            "target_precheck_complete": len(engine.target_certificates) == len(engine.target_keys),
            "target_precheck_budget_skipped": prefix_budget_skipped,
            "source_prefix_complete": not _source_prefix_menu(engine, states, None),
            "source_prefix_budget_blocked": source_prefix_budget_blocked,
            "world_sha256": engine.world_hash, "closed_components": closed,
            "graph_statistics": {"footprint_graphs": len(engine.loaded),
                                 "known_nodes": sum(len(graph.parents) for graph in graphs),
                                 "expanded_nodes": sum(len(graph.expanded) for graph in graphs),
                                 "closed_components": len(closed),
                                 "auxiliary_count_scope": "Paid collision checks include terminal, source and successor queries; terminal certificates never seed path graphs. Route visits counted separately; heuristic scans and final witness replay excluded from checker units but included in wall time."},
            "cost": {**engine.cost, **engine.aux, **engine.target_cost,
                     "total_units": sum(engine.cost.values()) + engine.target_cost["target_checks"],
                     "reserved_checker_work_units": reserved,
                     "actual_checker_work_units": engine.cost["source_checks"] + engine.aux["collision_cell_checks"]},
            "remaining_budget": None if budget is None else budget - reserved,
            "budget_blocked": budget_blocked, "parse_seconds": parse_seconds,
            "wall_seconds": time.perf_counter() - started}
