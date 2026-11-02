"""Validate Stage 2d goal-search controls using only independently paid evidence.

Stage 2c controls are validated unchanged by their frozen validator. New goal
FIFO/greedy arms share its independent geometry and graph-replay helpers, but
their active goal, complete known frontier and chosen node are reconstructed
here. The policy and its target/source/expansion checkers are never called.
"""
from __future__ import annotations

import copy
from collections import Counter

from innovation_stage2.domain import goal_id, parse_families
from innovation_stage2.policies import _Engine, _decision
from innovation_stage2b.policies import _graph_query, _public_menu, _source_query
from innovation_stage2c import validation as previous


CONTROLS = ("CHEAP_FIRST_TARGET", "COST_MENU_TARGET")
GOAL_METHODS = ("GOAL_FIFO_LEX_TARGET", "GOAL_GBFS_LEX_TARGET",
                "GOAL_FIFO_REVERSE_TARGET", "GOAL_GBFS_REVERSE_TARGET")
_require = previous._require
_integer = previous._integer


def _active_goal(engine, states, method):
    pairs = sorted((source, target) for source in engine.sources for target in engine.families[source])
    if "_REVERSE_" in method:
        pairs.reverse()
    for source, target in pairs:
        identifier = goal_id({"action": "place", "source_id": source, "target_id": target})
        if states[identifier] == "UNKNOWN":
            return identifier, source, target
    raise ValueError("An unresolved goal search has no UNKNOWN goal")


def _goal_frontier(engine, method, source, target):
    """Independently enumerate the whole eligible paid component frontier."""
    if not engine.source_reached(source):
        graph, start, destination = engine.empty, engine.start, engine.objects[source]["cell"]
        branch = "GOAL_EMPTY_ROUTE"
    else:
        graph, start, destination = engine.graph(source), engine.source_node(source), engine.targets[target]["cell"]
        branch = "GOAL_LOADED_ROUTE"
    _require(start in graph.parents, "Goal graph has no paid initialization")
    frontier = graph.members[graph.root(start)] - graph.expanded
    _require(frontier, "An UNKNOWN goal has an empty verified search frontier")
    allowed = engine.targets[target]["allowed_orientations"]

    def distance(node):
        result = abs(node[0] - destination[0]) + abs(node[1] - destination[1])
        return result + int(graph.footprint is not None and node[2] not in allowed)

    if "_FIFO_" in method:
        node = min(frontier, key=graph.order.__getitem__)
    else:
        node = min(frontier, key=lambda node: (distance(node), graph.order[node], node))
    return graph, node, branch, distance(node), len(frontier)


def _goal_menu(engine, method, states, remaining):
    counts = engine.unknown_counts(states)
    pending = [source for source in engine.sources if source not in engine.certificates and counts[source] > 0]
    if pending:
        menu = []
        for source in pending:
            query = _source_query(engine, source, counts[source])
            query.update({"active_goal": None, "goal_source_id": source, "goal_target_id": None,
                          "frontier_rule": "source_condition_prefix", "heuristic_value": None, "frontier_size": 0,
                          "score_numerator": 1, "score_denominator": 1, "priority_score": 1.0,
                          "affordable": remaining is None or query["reserved_work_units"] <= remaining})
            menu.append(query)
        return "SHARED_SOURCE_CONDITION_PREFIX", menu
    identifier, source, target = _active_goal(engine, states, method)
    _require(source in engine.certificates and engine.certificates[source]["passed"],
             "Active goal lacks the common paid source-condition prefix")
    graph, node, branch, heuristic, size = _goal_frontier(engine, method, source, target)
    affected = next((count for other, frontier, count in engine.search_jobs(counts)
                     if other.scope == graph.scope and other.root(frontier) == graph.root(node)), 0)
    query = _graph_query(graph, node, affected)
    rule = "FIFO_DISCOVERY" if "_FIFO_" in method else "GBFS_MANHATTAN_ORIENTATION"
    query.update({"active_goal": identifier, "goal_source_id": source, "goal_target_id": target,
                  "frontier_rule": rule, "heuristic_value": heuristic, "frontier_size": size,
                  "score_numerator": 1, "score_denominator": 1, "priority_score": 1.0,
                  "affordable": remaining is None or query["reserved_work_units"] <= remaining})
    return branch, [query]


def validate_prediction(prediction, runtime, oracle, budget):
    """Return the frozen score schema after checking every saved paid query."""
    if prediction.get("method") in CONTROLS:
        return previous.validate_prediction(prediction, runtime, oracle, budget)
    try:
        return _validate_goal_prediction(prediction, runtime, oracle, budget)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("Malformed Stage 2d prediction/evidence: " + str(exc)) from exc


def _validate_goal_prediction(prediction, runtime, oracle, budget):
    method = prediction["method"]
    _require(method in GOAL_METHODS, "Unknown Stage 2d policy")
    _require(budget is None or type(budget) is int and budget >= 0, "Invalid prepaid budget")
    strategy = "FIFO" if "_FIFO_" in method else "GBFS"
    order = "REVERSE" if "_REVERSE_" in method else "LEX"
    _require(prediction["base_method"] == "FIXED_GOAL_" + order + "_" + strategy
             and prediction["goal_order"] == order.lower()
             and prediction["policy_stage"] == "stage2d" and prediction["target_precheck_enabled"] is True,
             "Goal policy method metadata mismatch")
    _require(prediction["budget"] == budget, "Prediction budget differs from requested budget")
    try:
        families = parse_families(runtime)
    except ValueError:
        # The goal scheduler shares the frozen parser's unsupported response.
        # Convert only the method metadata to let the existing zero-work guard
        # check that no semantic or physical evidence was fabricated.
        _require(prediction["source_prefix_complete"] is False and prediction["source_prefix_budget_blocked"] is False,
                 "Unsupported input contains a source-prefix certificate or budget block")
        legacy = copy.deepcopy(prediction)
        legacy.update(method="CHEAP_FIRST_TARGET", base_method="CHEAP_FIRST_SHARED_CONFIG")
        return previous.validate_prediction(legacy, runtime, oracle, budget)
    _require(prediction["supported"] is True, "A supported runtime was marked unsupported")
    engine = _Engine(runtime, families)
    _require(prediction["world_sha256"] == engine.world_hash and prediction["families"] == families,
             "Public world binding or semantic families mismatch")
    target_keys = {}
    for source in engine.sources:
        footprint = tuple(engine.objects[source]["footprint"])
        for target in engine.families[source]:
            target_keys.setdefault((footprint, target), []).append(source)
    target_certificates, counters = {}, Counter()
    trace, schedules = prediction["trace"], prediction["schedule_trace"]
    _require(isinstance(trace, list) and isinstance(schedules, list), "Paid trace and schedule must be lists")
    reserved = actual = cursor = 0
    prefix_active, skipped, blocked, source_blocked = True, False, False, False
    for index, entry in enumerate(schedules):
        states, _ = previous._states(engine, target_keys, target_certificates)
        decision, _ = _decision(states)
        _require(states and decision == "ABSTAIN" and not blocked, "Schedule continues after a certified or empty decision")
        remaining = None if budget is None else budget - reserved
        feasible = sum(value == "FEASIBLE" for value in states.values())
        phase = {"known_feasible_goals": feasible, "phase_cap": max(1, 2 - feasible),
                 "current_evidence_decision": decision}
        _require(entry["step"] == cursor + 1 and entry["schedule_index"] == index + 1
                 and entry["world_sha256"] == engine.world_hash and entry["decision_phase"] == phase
                 and entry["remaining_budget_before"] == remaining and entry["reserved_work_before"] == reserved,
                 "Schedule phase/world/prepayment history mismatch")
        ranked = previous._target_menu(engine, target_keys, target_certificates, states, remaining) if prefix_active else []
        target_phase = bool(ranked)
        if target_phase:
            branch = "FIXED_TARGET_TERMINAL_PRECHECK"
        else:
            prefix_active = False
            branch, ranked = _goal_menu(engine, method, states, remaining)
        _require(ranked and entry["menu_scope"] == branch and entry["candidate_menu"] == _public_menu(ranked),
                 "Goal/frontier/priority menu differs from independent reconstruction")
        _require(entry["skipped_unaffordable_query_ids"] == [q["query_id"] for q in ranked if not q["affordable"]],
                 "Unaffordable-query log mismatch")
        chosen = next((query for query in ranked if query["affordable"]), None)
        if chosen is None:
            outcome = "precheck_budget_skipped_continue_base" if target_phase else "budget_blocked"
            _require(entry["outcome"] == outcome and entry["selected_query_id"] is None
                     and entry["reserved_work_delta"] == 0 and entry["actual_checker_work_delta"] == 0
                     and entry["remaining_budget_after"] == remaining, "Invalid zero-work scheduling event")
            if target_phase:
                _require(index < len(schedules) - 1, "Unaffordable target prefix did not continue to the goal menu")
                prefix_active, skipped = False, True
            else:
                _require(index == len(schedules) - 1, "Budget-blocked goal schedule is not terminal")
                blocked = True
                source_blocked = branch == "SHARED_SOURCE_CONDITION_PREFIX"
            continue
        _require(cursor < len(trace), "Scheduled query has no paid evidence event")
        event = trace[cursor]
        _require(event["operation"] == chosen["operation"] and entry["outcome"] == "queried"
                 and entry["selected_query_id"] == chosen["query_id"] and event["query_id"] == chosen["query_id"]
                 and event["world_sha256"] == engine.world_hash and type(event["cost_units"]) is int and event["cost_units"] == 1,
                 "Paid event does not match the selected bound goal query")
        for name in ("source_id", "target_id", "footprint", "node"):
            if name in chosen:
                _require(event.get(name) == chosen[name], "Paid query operands differ from its schedule")
        estimate = _integer(chosen["reserved_work_units"], "query reservation")
        _require(budget is None or reserved + estimate <= budget, "Reserved checker budget exceeded before a query")
        if event["operation"] == "target_check":
            _require(target_phase, "Target query outside the fixed paid prefix")
            expected, measured, cells, orientations, reuses = previous._target_event(engine, event, target_keys, target_certificates)
            counters.update(target_checks=1, target_orientation_checks=orientations,
                            target_collision_cell_checks=cells, target_cache_hits=reuses)
            transitions = 0
        elif event["operation"] == "source_check":
            expected, measured, cells, transitions = previous._source_event(engine, event)
            counters["source_checks"] += 1
        else:
            expected, measured, cells, transitions = previous._graph_event(engine, event)
            counters["empty_node_expansions" if event["operation"] == "expand_empty" else "loaded_node_expansions"] += 1
        _require(estimate == expected and _integer(event["reserved_work_units"], "paid reservation") == expected
                 and _integer(event["actual_checker_work_delta"], "actual paid work") == measured
                 and entry["reserved_work_delta"] == expected and entry["actual_checker_work_delta"] == measured
                 and 0 <= measured <= expected, "Per-query reservation/actual checker-work ledger mismatch")
        reserved += expected
        actual += measured
        counters.update(collision_cell_checks=cells, transition_checks=transitions, total_units=1)
        _require(entry["remaining_budget_after"] == (None if budget is None else budget - reserved),
                 "Post-query budget ledger mismatch")
        cursor += 1
    _require(cursor == len(trace), "Paid trace has missing scheduling entries")
    states, reasons = previous._states(engine, target_keys, target_certificates)
    decision, selected_goal = _decision(states)
    _require(prediction["goal_states"] == states and prediction["goal_evidence"] == reasons,
             "Replayed paid evidence disagrees with final goal states/reasons")
    _require(prediction["decision"] == decision and prediction["selected_goal"] == selected_goal,
             "Final decision is inconsistent with paid evidence")
    _require(prediction["candidate_count"] == len(states) and prediction["family_count"] == len(engine.families),
             "Semantic candidate/family counts mismatch")
    _require(not states or decision != "ABSTAIN" or blocked,
             "An unresolved goal task stopped without exhausting affordable evidence")
    _require(prediction["source_certificates"] == list(engine.certificates.values()),
             "Source certificates are missing, forged, or unbacked by paid events")
    _require(prediction["target_certificates"] == list(target_certificates.values()),
             "Target certificates are missing, forged, or unbacked by paid events")
    _require(prediction["target_keys_total"] == len(target_keys)
             and prediction["target_precheck_complete"] is (len(target_certificates) == len(target_keys))
             and prediction["target_precheck_budget_skipped"] is skipped and prediction["budget_blocked"] is blocked
             and prediction["remaining_budget"] == (None if budget is None else budget - reserved),
             "Terminal prefix/cache/budget metadata mismatch")
    pending = [source for source, count in engine.unknown_counts(states).items()
               if source not in engine.certificates and count > 0]
    _require(prediction["goal_order"] == ("reverse" if "_REVERSE_" in method else "lex")
             and prediction["source_prefix_complete"] is (not pending)
             and prediction["source_prefix_budget_blocked"] is source_blocked,
             "Goal order or common source-prefix metadata mismatch")
    expected_reason = ("EMPTY_SEMANTICS" if not states else "decision_certified" if decision != "ABSTAIN"
                       else "source_prefix_budget_blocked" if source_blocked else "budget_no_affordable_query")
    _require(prediction["reason"] == expected_reason, "Terminal stopping reason mismatch")
    if selected_goal is None:
        _require(prediction["selected_candidate"] is None and prediction["witness"] == [],
                 "A non-execution decision contains an execution witness")
    else:
        candidate = next({"action": "place", "source_id": source, "target_id": target}
                         for source in engine.sources for target in engine.families[source]
                         if goal_id({"action": "place", "source_id": source, "target_id": target}) == selected_goal)
        _require(prediction["selected_candidate"] == candidate
                 and prediction["witness"] == engine.witness(candidate, reasons[selected_goal]),
                 "Selected witness is not grounded in the paid graph")
    graphs = [engine.empty] + [engine.loaded[key] for key in sorted(engine.loaded)]
    closed = [certificate for graph in graphs for certificate in graph.closed_certificates()]
    _require(prediction["closed_components"] == closed, "Closed-component certificates differ from paid graph replay")
    graph_counts = {"footprint_graphs": len(engine.loaded), "known_nodes": sum(len(g.parents) for g in graphs),
                    "expanded_nodes": sum(len(g.expanded) for g in graphs), "closed_components": len(closed)}
    _require(all(prediction["graph_statistics"].get(key) == value for key, value in graph_counts.items()),
             "Graph statistics contain unpaid nodes or inconsistent components")
    cost = prediction["cost"]
    counters["reserved_checker_work_units"] = reserved
    counters["actual_checker_work_units"] = actual
    for key in ("source_checks", "empty_node_expansions", "loaded_node_expansions", "total_units",
                "collision_cell_checks", "transition_checks", "reserved_checker_work_units", "actual_checker_work_units",
                *previous._TARGET_COUNTERS):
        _require(_integer(cost[key], key) == counters[key], "Aggregate checker-work ledger mismatch: " + key)
    _require(actual == cost["source_checks"] + cost["collision_cell_checks"] and actual <= reserved,
             "Actual work exceeds or disagrees with the paid ledger")
    _integer(cost["cache_hits"], "cache_hits")
    _require(_integer(cost["graph_traversals"], "graph_traversals") == engine.aux["graph_traversals"],
             "Witness route traversal count mismatch")
    legacy = copy.deepcopy(prediction)
    legacy["trace"] = [event for event in legacy["trace"] if event["operation"] != "target_check"]
    legacy["cost"]["total_units"] -= cost["target_checks"]
    legacy["cost"]["collision_cell_checks"] -= cost["target_collision_cell_checks"]
    return previous.previous.validate_prediction(legacy, runtime, oracle,
                                                 None if budget is None else legacy["cost"]["total_units"])
