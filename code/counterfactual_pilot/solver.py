"""Existing GBFS with explicit contract-dependent stopping; NOT a new algorithm.

Forked from innovation_stage3/policies.py.
Explicit-catalog adaptation of frozen Stage2d GBFS. No parser/oracle calls.

Scientific changes: caller supplies the supported semantic subset; the paid
prefix, graph cache, lexical frontier rule, witnesses and costs are unchanged.
Global sufficiency requires a separate completeness check by the caller.
"""
from __future__ import annotations
import copy
import time
from typing import Any
from innovation_stage2.domain import goal_id, validate_runtime, validate_witness
from innovation_stage2.policies import _decision as referential_decision
from innovation_stage2c.policies import _TargetEngine, _execute_query, _schedule_event
from innovation_stage2d.policies import _goal_metadata, _goal_menu, _source_prefix_menu


def decision_rule(mode):
    if mode == "referential":
        return referential_decision
    if mode != "any_valid":
        raise ValueError("Unknown execution contract")
    def decide(states):
        feasible = sorted(key for key, value in states.items() if value == "FEASIBLE")
        if feasible:
            return "EXECUTE", feasible[0]
        if states and all(value == "INFEASIBLE" for value in states.values()):
            return "REJECT", None
        return "ABSTAIN", None
    return decide


def checked_families(runtime, families):
    sources = {o["id"] for o in runtime["objects"]}
    targets = {o["id"] for o in runtime["targets"]}
    if not isinstance(families, list):
        raise ValueError("Catalog must be a family list")
    seen = set()
    for family in families:
        if not isinstance(family, dict) or set(family) != {"source_id", "target_ids"}:
            raise ValueError("Invalid family")
        source, ts = family["source_id"], family["target_ids"]
        if source not in sources or source in seen or not isinstance(ts, list) or not ts:
            raise ValueError("Invalid family source or targets")
        if any(t not in targets for t in ts) or ts != sorted(set(ts)):
            raise ValueError("Unknown/duplicate/noncanonical target")
        seen.add(source)
    if [f["source_id"] for f in families] != sorted(seen):
        raise ValueError("Catalog must be sorted by source ID")
    return copy.deepcopy(families)

def run_contract(runtime: dict[str, Any], families: list[dict[str, Any]], budget: int | None, seed: int = 101, mode: str = "referential") -> dict[str, Any]:
    decide = decision_rule(mode)
    started = time.perf_counter()
    method = "GOAL_GBFS_LEX_TARGET"
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("Invalid budget")
    validate_runtime(runtime)
    families = checked_families(runtime, families)
    parse_seconds = 0.0  # The caller records catalog construction/membership costs separately.
    engine = _TargetEngine(runtime, families)
    reserved, schedule = 0, []
    prefix_active, prefix_budget_skipped, budget_blocked = True, False, False
    source_prefix_active, source_prefix_budget_blocked = True, False
    while True:
        states, reasons = engine.states()
        decision, _ = decide(states)
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
    decision, selected_goal = decide(states)
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

