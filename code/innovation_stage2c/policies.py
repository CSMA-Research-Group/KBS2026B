"""A fixed, paid terminal-pose prefix crossed with two unchanged schedulers.

This is a solver-strengthening control, not a learned or optimal scheduler.
The OFF controls delegate directly to frozen Stage 2b. ON first queries public
(original footprint, target ID) keys in canonical order, skipping unaffordable
keys. If none can be afforded, the remaining budget goes to the original
source/graph menu. A valid terminal pose alone never certifies reachability.
"""
from __future__ import annotations

import copy
import time
from typing import Any

from innovation_stage2.domain import configuration_valid, goal_id, parse_families, validate_witness
from innovation_stage2.policies import _Engine, _decision
from innovation_stage2b.policies import _candidate_menu, _graph_query, _public_menu, _rank_menu
from innovation_stage2b.policies import run_policy as run_stage2b_policy


METHODS = ("CHEAP_FIRST_NO_TARGET", "COST_MENU_NO_TARGET", "CHEAP_FIRST_TARGET", "COST_MENU_TARGET")
BASE_METHODS = {
    "CHEAP_FIRST_NO_TARGET": "CHEAP_FIRST_SHARED_CONFIG",
    "COST_MENU_NO_TARGET": "COST_MENU_SHARED_CONFIG",
    "CHEAP_FIRST_TARGET": "CHEAP_FIRST_SHARED_CONFIG",
    "COST_MENU_TARGET": "COST_MENU_SHARED_CONFIG",
}
TARGET_METHODS = METHODS[2:]
TARGET_COUNTERS = ("target_checks", "target_orientation_checks", "target_collision_cell_checks", "target_cache_hits")


class _TargetEngine(_Engine):
    def __init__(self, runtime, families):
        super().__init__(runtime, families)
        self.target_certificates = {}
        self.target_cost = {name: 0 for name in TARGET_COUNTERS}
        self.target_keys = {}
        for source in self.sources:
            footprint = tuple(self.objects[source]["footprint"])
            for target in self.families[source]:
                self.target_keys.setdefault((footprint, target), []).append(source)

    def target_menu(self, states, remaining):
        result = []
        for (footprint, target), sources in sorted(self.target_keys.items()):
            if (footprint, target) in self.target_certificates:
                continue
            allowed = self.targets[target]["allowed_orientations"]
            reserve = footprint[0] * footprint[1] * len(allowed)
            affected = sum(states[goal_id({"action": "place", "source_id": source, "target_id": target})] == "UNKNOWN"
                           for source in sources)
            result.append({"query_id": f"target:{footprint[0]}x{footprint[1]}:{target}",
                           "operation": "target_check", "footprint": list(footprint), "target_id": target,
                           "source_ids": list(sources), "allowed_orientations": list(allowed),
                           "affected_unknown_goals": affected, "reserved_work_units": reserve,
                           "canonical_tie": [-1, *footprint, target], "score_numerator": 1,
                           "score_denominator": 1, "priority_score": 1.0,
                           "affordable": remaining is None or reserve <= remaining})
        return result

    def target_check(self, query):
        footprint, target = tuple(query["footprint"]), query["target_id"]
        key = (footprint, target)
        if key in self.target_certificates:
            raise RuntimeError("A terminal-pose key was paid twice")
        target_spec = self.targets[target]
        poses = []
        for orientation in target_spec["allowed_orientations"]:
            before = self.aux["collision_cell_checks"]
            valid = configuration_valid(self.runtime, footprint, (*target_spec["cell"], orientation), self.aux)
            poses.append({"orientation": orientation, "valid": valid,
                          "collision_cell_checks": self.aux["collision_cell_checks"] - before})
        cells = sum(pose["collision_cell_checks"] for pose in poses)
        reuse = len(self.target_keys[key]) - 1
        self.target_cost["target_checks"] += 1
        self.target_cost["target_orientation_checks"] += len(poses)
        self.target_cost["target_collision_cell_checks"] += cells
        self.target_cost["target_cache_hits"] += reuse
        certificate = {"query_id": query["query_id"], "footprint": list(footprint), "target_id": target,
                       "target_cell": list(target_spec["cell"]),
                       "source_ids": list(self.target_keys[key]),
                       "allowed_orientations": list(target_spec["allowed_orientations"]),
                       "orientation_results": poses,
                       "all_terminal_poses_invalid": not any(pose["valid"] for pose in poses),
                       "world_sha256": self.world_hash,
                       "scope": "necessary_terminal_occupancy_only; no_reachability_or_source_eligibility",
                       "cache_reuses": reuse}
        self.target_certificates[key] = certificate
        self.trace.append({"operation": "target_check", "cost_units": 1, **copy.deepcopy(certificate)})

    def states(self):
        states, reasons = super().states()
        for (footprint, target), certificate in self.target_certificates.items():
            if not certificate["all_terminal_poses_invalid"]:
                continue
            for source in self.target_keys[(footprint, target)]:
                key = goal_id({"action": "place", "source_id": source, "target_id": target})
                if states[key] == "FEASIBLE":
                    raise RuntimeError("Terminal negative certificate contradicts an existing path")
                if states[key] == "UNKNOWN":
                    states[key] = "INFEASIBLE"
                    reasons[key] = {"kind": "target_terminal_necessary_condition", "footprint": list(footprint),
                                    "target_id": target, "query_id": certificate["query_id"]}
        return states, reasons


def _metadata(method):
    return {"method": method, "base_method": BASE_METHODS[method],
            "target_precheck_enabled": method in TARGET_METHODS,
            "policy_kind": "fixed_paid_terminal_precheck_control; Stage2b_base_scheduler_with_resolved_source_pruning_ON_only; no_novelty_claim",
            "target_precheck_semantics": "Public (original footprint,target ID) fixed prefix before source checks; each allowed orientation checked; negative certificates only. First affordable key, then original scheduler if no remaining key fits. No source-eligibility filtering or graph warm-start.",
            "target_cache_semantics": "Single policy invocation, keyed by (ordered original footprint,target ID). target_cache_hits counts additional public source-target pairs served by each paid certificate once, not repeated state-map reads.",
            "target_cost_semantics": "target reservation=footprint area*allowed orientation count; all cells visited so actual target collision cells equal reservation. collision_cell_checks includes these target cells; target_collision_cell_checks is an overlapping subset. total_units counts source, target and graph queries. No refund."}


def _base_candidate_menu(engine, base_method, states):
    """Use paid negatives to deactivate irrelevant sources for both ON policies.

The cost menu already filters unresolved counts. Apply the same inactivity rule
to CHEAP, preserving its canonical source-first order among remaining sources.
OFF delegates to the old implementation without this addition.
    """
    branch, candidates = _candidate_menu(engine, base_method, states)
    if base_method == "CHEAP_FIRST_SHARED_CONFIG" and branch == "SOURCE_CONDITIONS_FIRST":
        candidates = [query for query in candidates if query["affected_unknown_goals"] > 0]
        if not candidates:
            counts = engine.unknown_counts(states)
            return "CHECKED_SOURCE_SEARCH", [_graph_query(graph, node, affected)
                                              for graph, node, affected in engine.search_jobs(counts)]
    return branch, candidates


def run_policy(runtime: dict[str, Any], method: str, budget: int | None, seed: int = 101) -> dict[str, Any]:
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError("Unknown Stage 2c policy: " + str(method))
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be a nonnegative prepaid-work integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    base_method = BASE_METHODS[method]
    if method not in TARGET_METHODS:
        result = run_stage2b_policy(runtime, base_method, budget, seed)
        result.update(_metadata(method))
        result["cost"].update({name: 0 for name in TARGET_COUNTERS})
        result.update({"target_certificates": [], "target_precheck_complete": None,
                       "target_precheck_budget_skipped": False, "target_keys_total": 0})
        return result
    parse_started = time.perf_counter()
    try:
        families = parse_families(runtime)
    except ValueError:
        # Preserve exactly the frozen parser's unsupported-input response.
        result = run_stage2b_policy(runtime, base_method, budget, seed)
        result.update(_metadata(method))
        result["cost"].update({name: 0 for name in TARGET_COUNTERS})
        result.update({"target_certificates": [], "target_precheck_complete": False,
                       "target_precheck_budget_skipped": False, "target_keys_total": 0})
        result["wall_seconds"] = time.perf_counter() - started
        return result
    parse_seconds = time.perf_counter() - parse_started
    engine = _TargetEngine(runtime, families)
    reserved, schedule = 0, []
    prefix_active, prefix_budget_skipped, budget_blocked = True, False, False
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
        feasible = sum(state == "FEASIBLE" for state in states.values())
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
        branch, candidates = _base_candidate_menu(engine, base_method, states)
        if not candidates:
            raise RuntimeError("Supported unresolved finite task has no source or graph query")
        ranked = _rank_menu(base_method, candidates, feasible, remaining)
        chosen = next((query for query in ranked if query["affordable"]), None)
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
        raise RuntimeError("Terminal-prefix policy produced an invalid carrying witness")
    graphs = [engine.empty] + [engine.loaded[key] for key in sorted(engine.loaded)]
    closed = [certificate for graph in graphs for certificate in graph.closed_certificates()]
    return {**_metadata(method), "budget": budget, "seed": seed,
            "budget_semantics": "reserved_checker_work_units; no refunds",
            "cost_semantics": "source=1+area, EMPTY=4, loaded=5wh+2(w+h)+max(w,h)^2, target=area*allowed orientations. actual_checker_work_units=source_checks+collision_cell_checks; excludes final witness replay and bookkeeping, which remain in wall time.",
            "candidate_catalog_materialized": False, "audit_goal_states_materialized": True,
            "runtime_candidate_representation": "source_target_families_with_explicit_target_ids",
            "canonical_tie_semantics": "target prefix (-1,width,height,target_id), then frozen Stage2b ties",
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
            "world_sha256": engine.world_hash, "closed_components": closed,
            "graph_statistics": {"footprint_graphs": len(engine.loaded),
                                 "known_nodes": sum(len(graph.parents) for graph in graphs),
                                 "expanded_nodes": sum(len(graph.expanded) for graph in graphs),
                                 "closed_components": len(closed),
                                 "auxiliary_count_scope": "collision counters include target checks and paid source/successor checks; target certificates never add graph nodes. Final witness replay excluded; witness route visits counted separately."},
            "cost": {**engine.cost, **engine.aux, **engine.target_cost,
                     "total_units": sum(engine.cost.values()) + engine.target_cost["target_checks"],
                     "reserved_checker_work_units": reserved,
                     "actual_checker_work_units": engine.cost["source_checks"] + engine.aux["collision_cell_checks"]},
            "remaining_budget": None if budget is None else budget - reserved,
            "budget_blocked": budget_blocked, "parse_seconds": parse_seconds,
            "wall_seconds": time.perf_counter() - started}


def _schedule_event(engine, schedule, states, remaining, reserved, ranked, chosen, branch):
    feasible = sum(state == "FEASIBLE" for state in states.values())
    return {"step": len(engine.trace) + 1, "schedule_index": len(schedule) + 1,
            "world_sha256": engine.world_hash, "menu_scope": branch,
            "decision_phase": {"known_feasible_goals": feasible, "phase_cap": max(1, 2 - feasible),
                               "current_evidence_decision": _decision(states)[0]},
            "remaining_budget_before": remaining, "reserved_work_before": reserved,
            "candidate_menu": _public_menu(ranked),
            "skipped_unaffordable_query_ids": [query["query_id"] for query in ranked if not query["affordable"]],
            "selected_query_id": chosen["query_id"] if chosen else None}


def _execute_query(engine, chosen, event, reserved, budget):
    prepaid = chosen["reserved_work_units"]
    if budget is not None and prepaid > budget - reserved:
        raise RuntimeError("A query lacks a prepaid reservation")
    before = engine.cost["source_checks"] + engine.aux["collision_cell_checks"]
    # Reserve before entering any checker, without a refund afterwards.
    reserved += prepaid
    if chosen["operation"] == "target_check":
        engine.target_check(chosen)
    elif chosen["operation"] == "source_check":
        engine.source_check(chosen["_source"])
    else:
        chosen["_graph"].expand(tuple(chosen["node"]))
    delta = engine.cost["source_checks"] + engine.aux["collision_cell_checks"] - before
    if not 0 <= delta <= prepaid:
        raise RuntimeError("A checker exceeded its prepaid work reservation")
    engine.trace[-1].update({"query_id": chosen["query_id"], "reserved_work_units": prepaid,
                             "actual_checker_work_delta": delta})
    event.update({"outcome": "queried", "reserved_work_delta": prepaid,
                  "actual_checker_work_delta": delta,
                  "remaining_budget_after": None if budget is None else budget - reserved})
    return reserved
