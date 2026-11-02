"""Fixed Stage 2b scheduling controls over the unchanged Stage 2 evidence engine.

Reservations are conservative prepaid checker work, not measured hardware time:
source=1+footprint area, EMPTY=4, loaded=5wh+2(w+h)+max(w,h)^2.
Reservations are never refunded, even when a negative source or a cached initial
configuration uses fewer actual checks. All methods share the same graph cache.
The decision-capped priority is a fixed heuristic, not expected information gain,
an optimal decision rule, a learned scheduler or a new-algorithm claim.
"""
from __future__ import annotations

import copy
from fractions import Fraction
import time
from typing import Any

from innovation_stage2.domain import goal_id, parse_families, validate_witness
from innovation_stage2.policies import _Engine, _decision


METHODS = (
    "FULL_SHARED_CONFIG", "CHEAP_FIRST_SHARED_CONFIG", "SIMPLE_TIED_SHARED_CONFIG",
    "COUNT_TIED_SHARED_CONFIG", "COUNT_MENU_SHARED_CONFIG", "COST_MENU_SHARED_CONFIG",
    "DECISION_COST_SHARED_CONFIG", "DECISION_UNIT_SHARED_CONFIG",
)
MENU_METHODS = METHODS[4:]


def _source_query(engine, source, affected):
    footprint = tuple(engine.objects[source]["footprint"])
    return {"query_id": "source:" + source, "operation": "source_check", "source_id": source,
            "footprint": list(footprint), "affected_unknown_goals": affected,
            "reserved_work_units": 1 + footprint[0] * footprint[1],
            "canonical_tie": [0, source], "_source": source}


def _graph_query(graph, node, affected):
    node, root = tuple(node), graph.root(node)
    if graph.footprint is None:
        reserved, operation, kind = 4, "expand_empty", 1
    else:
        width, height = graph.footprint
        reserved = 5 * width * height + 2 * (width + height) + max(width, height) ** 2
        operation, kind = "expand_loaded", 2
    return {"query_id": graph.scope + ":root=" + ",".join(map(str, root)) + ":node=" + ",".join(map(str, node)),
            "operation": operation, "scope": graph.scope, "node": list(node),
            "component_root": list(root), "footprint": list(graph.footprint) if graph.footprint else None,
            "affected_unknown_goals": affected, "reserved_work_units": reserved,
            "canonical_tie": [kind, graph.scope, *root, *node], "_graph": graph}


def _candidate_menu(engine, method, states):
    """Form the legal branch/menu using paid evidence and public family structure."""
    counts = engine.unknown_counts(states)
    all_pending = [source for source in engine.sources if source not in engine.certificates]
    pending = [source for source in all_pending if counts[source] > 0]

    def source_menu(sources):
        return [_source_query(engine, source, counts[source]) for source in sources]

    def graph_menu(include_unchecked=False, full=False):
        jobs = engine.search_jobs(counts, include_unchecked=include_unchecked, full=full)
        if full:
            # FULL also completes components that no longer contain UNKNOWN
            # goals; retain those queries without mislabelling their count.
            unresolved = {(graph.scope, graph.root(node)): affected for graph, node, affected
                          in engine.search_jobs(counts)}
            return [_graph_query(graph, node, unresolved.get((graph.scope, graph.root(node)), 0))
                    for graph, node, _ in jobs]
        return [_graph_query(graph, node, affected) for graph, node, affected in jobs]

    if method in MENU_METHODS:
        return "COMMON_SOURCE_EMPTY_LOADED_MENU", source_menu(pending) + graph_menu(include_unchecked=True)
    if method in ("FULL_SHARED_CONFIG", "CHEAP_FIRST_SHARED_CONFIG"):
        if all_pending:
            return "SOURCE_CONDITIONS_FIRST", source_menu(all_pending)
        if method == "FULL_SHARED_CONFIG" and not engine.empty.closed(engine.start):
            affected = sum(counts[source] for source in engine.sources if not engine.source_reached(source))
            return "FULL_EMPTY_COMPONENT", [_graph_query(engine.empty, engine.empty.frontier(engine.start), affected)]
        return "FULL_REQUIRED_COMPONENTS" if method == "FULL_SHARED_CONFIG" else "CHECKED_SOURCE_SEARCH", graph_menu(full=method == "FULL_SHARED_CONFIG")
    ready = [source for source in pending if engine.source_reached(source)]
    if ready:
        return "READY_SOURCE_CONDITIONS", source_menu(ready)
    jobs = graph_menu()
    if jobs:
        return "CHECKED_SOURCE_SEARCH", jobs
    return "PENDING_SOURCE_CONDITIONS", source_menu(pending)


def _rank_menu(method, candidates, n_feasible, remaining):
    """Exact rational scores; one canonical tie convention across all methods."""
    ranked = []
    cap = max(1, 2 - n_feasible)
    for candidate in candidates:
        query = dict(candidate)
        affected = query["affected_unknown_goals"]
        capped = min(affected, cap)
        if method in ("FULL_SHARED_CONFIG", "CHEAP_FIRST_SHARED_CONFIG", "SIMPLE_TIED_SHARED_CONFIG"):
            numerator, denominator = 1, 1
        elif method in ("COUNT_TIED_SHARED_CONFIG", "COUNT_MENU_SHARED_CONFIG"):
            numerator, denominator = affected, 1
        elif method == "COST_MENU_SHARED_CONFIG":
            numerator, denominator = affected, query["reserved_work_units"]
        elif method == "DECISION_COST_SHARED_CONFIG":
            numerator, denominator = capped, query["reserved_work_units"]
        elif method == "DECISION_UNIT_SHARED_CONFIG":
            numerator, denominator = capped, 1
        else:
            raise ValueError("Unknown Stage 2b method")
        query.update({"phase_capped_value": capped, "score_numerator": numerator,
                      "score_denominator": denominator, "priority_score": float(Fraction(numerator, denominator)),
                      "affordable": remaining is None or query["reserved_work_units"] <= remaining})
        ranked.append(query)
    return sorted(ranked, key=lambda query: (-Fraction(query["score_numerator"], query["score_denominator"]), tuple(query["canonical_tie"])))


def _public_menu(ranked):
    return [{key: value for key, value in query.items() if not key.startswith("_")} for query in ranked]


def _empty_cost():
    return {"source_checks": 0, "empty_node_expansions": 0, "loaded_node_expansions": 0, "total_units": 0,
            "collision_cell_checks": 0, "transition_checks": 0, "cache_hits": 0, "graph_traversals": 0,
            "reserved_checker_work_units": 0, "actual_checker_work_units": 0}


def run_policy(runtime: dict[str, Any], method: str, budget: int | None, seed: int = 101) -> dict[str, Any]:
    started = time.perf_counter()
    if method not in METHODS:
        raise ValueError("Unknown Stage 2b policy: " + str(method))
    if budget is not None and (type(budget) is not int or budget < 0):
        raise ValueError("budget must be a nonnegative prepaid-work integer or None")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    base = {"method": method, "budget": budget, "seed": seed,
            "budget_semantics": "reserved_checker_work_units; source reservations never refunded",
            "candidate_catalog_materialized": False, "audit_goal_states_materialized": True,
            "runtime_candidate_representation": "source_target_families_with_explicit_target_ids",
            "policy_kind": "fixed_development_heuristic; capped value is not expected gain or an optimal/learned decision rule",
            "canonical_tie_semantics": "source: (0,source_id); EMPTY/loaded: (1/2,scope,component_root,frontier_node)",
            "cost_semantics": "reserved source=1+footprint_area, EMPTY=4, loaded=5wh+2(w+h)+max(w,h)^2; no refunds. actual_checker_work_units=source_checks+collision_cell_checks. total_units remains source/node-query count. Final witness validation and graph bookkeeping are wall time only; graph_traversals counts witness-route node visits only"}
    parse_started = time.perf_counter()
    try:
        families = parse_families(runtime)
    except ValueError as exc:
        return {**base, "supported": False, "reason": "UNSUPPORTED: " + str(exc), "decision": "ABSTAIN",
                "selected_goal": None, "selected_candidate": None, "witness": [], "goal_states": {},
                "trace": [], "schedule_trace": [], "source_certificates": [], "closed_components": [],
                "candidate_count": 0, "cost": _empty_cost(), "remaining_budget": budget, "budget_blocked": False,
                "parse_seconds": time.perf_counter() - parse_started, "wall_seconds": time.perf_counter() - started}
    parse_seconds = time.perf_counter() - parse_started
    engine = _Engine(runtime, families)
    reserved, schedule = 0, []
    stop_reason, budget_blocked = "budget_no_affordable_query", False
    while True:
        states, reasons = engine.states()
        decision, _ = _decision(states)
        if not states:
            stop_reason = "EMPTY_SEMANTICS"
            break
        if method != "FULL_SHARED_CONFIG" and decision != "ABSTAIN":
            stop_reason = "decision_certified"
            break
        remaining = None if budget is None else budget - reserved
        branch, candidates = _candidate_menu(engine, method, states)
        if not candidates:
            if decision == "ABSTAIN":
                raise RuntimeError("A supported nonempty finite task has no evidence query but remains unresolved")
            stop_reason = "all_required_shared_components_complete"
            break
        n_feasible = sum(state == "FEASIBLE" for state in states.values())
        ranked = _rank_menu(method, candidates, n_feasible, remaining)
        chosen = next((query for query in ranked if query["affordable"]), None)
        event = {"step": len(engine.trace) + 1, "world_sha256": engine.world_hash, "menu_scope": branch,
                 "decision_phase": {"known_feasible_goals": n_feasible, "phase_cap": max(1, 2 - n_feasible),
                                    "current_evidence_decision": decision},
                 "remaining_budget_before": remaining, "reserved_work_before": reserved,
                 "candidate_menu": _public_menu(ranked),
                 "skipped_unaffordable_query_ids": [query["query_id"] for query in ranked if not query["affordable"]],
                 "selected_query_id": chosen["query_id"] if chosen else None}
        if chosen is None:
            event.update({"outcome": "budget_blocked", "reserved_work_delta": 0, "actual_checker_work_delta": 0,
                          "remaining_budget_after": remaining})
            schedule.append(event)
            budget_blocked, stop_reason = True, "budget_no_affordable_query"
            break
        prepaid = chosen["reserved_work_units"]
        actual_before = engine.cost["source_checks"] + engine.aux["collision_cell_checks"]
        if chosen["operation"] == "source_check":
            engine.source_check(chosen["_source"])
        else:
            chosen["_graph"].expand(tuple(chosen["node"]))
        actual_delta = engine.cost["source_checks"] + engine.aux["collision_cell_checks"] - actual_before
        if not 0 <= actual_delta <= prepaid:
            raise RuntimeError("Conservative reservation failed to bound the executed checker work")
        reserved += prepaid
        if budget is not None and reserved > budget:
            raise RuntimeError("A query exceeded its prechecked reservation budget")
        engine.trace[-1].update({"query_id": chosen["query_id"], "reserved_work_units": prepaid,
                                 "actual_checker_work_delta": actual_delta})
        event.update({"outcome": "queried", "reserved_work_delta": prepaid, "actual_checker_work_delta": actual_delta,
                      "remaining_budget_after": None if budget is None else budget - reserved})
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
        raise RuntimeError("Shared graph evidence produced an invalid carrying witness")
    graphs = [engine.empty] + [engine.loaded[key] for key in sorted(engine.loaded)]
    closed = [certificate for graph in graphs for certificate in graph.closed_certificates()]
    return {**base, "supported": True, "reason": stop_reason, "decision": decision,
            "selected_goal": selected_goal, "selected_candidate": selected, "witness": witness,
            "goal_states": states, "goal_evidence": reasons, "families": copy.deepcopy(families),
            "candidate_count": len(states), "family_count": len(engine.families),
            "trace": engine.trace, "schedule_trace": schedule, "source_certificates": list(engine.certificates.values()),
            "world_sha256": engine.world_hash, "closed_components": closed,
            "graph_statistics": {"footprint_graphs": len(engine.loaded), "known_nodes": sum(len(graph.parents) for graph in graphs),
                                 "expanded_nodes": sum(len(graph.expanded) for graph in graphs), "closed_components": len(closed),
                                 "auxiliary_count_scope": "collision/transition counters cover paid source/successor queries only; final witness replay excluded. graph_traversals counts witness-route node visits, not all graph bookkeeping"},
            "cost": {**engine.cost, "total_units": sum(engine.cost.values()), **engine.aux,
                     "reserved_checker_work_units": reserved,
                     "actual_checker_work_units": engine.cost["source_checks"] + engine.aux["collision_cell_checks"]},
            "remaining_budget": None if budget is None else budget - reserved, "budget_blocked": budget_blocked,
            "parse_seconds": parse_seconds, "wall_seconds": time.perf_counter() - started}
