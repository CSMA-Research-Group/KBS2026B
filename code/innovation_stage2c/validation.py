"""Independent paid-evidence validation for the Stage 2c development controls.

The saved oracle is an evaluation input only.  Target occupancy, source checks,
and motion successors are recomputed here from the public world, without calling
the online target checker or executing the policy.  Only verified, paid events
are inserted into a fresh Stage 2 evidence engine; target checks never seed its
configuration graphs.  This is artifact validation, not a deployment guarantee.
"""
from __future__ import annotations

import copy
from collections import Counter
from typing import Any

from innovation_stage2.domain import goal_id, parse_families
from innovation_stage2.policies import _Engine, _decision
from innovation_stage2b.policies import _candidate_menu, _public_menu, _rank_menu
from scripts import run_innovation_stage2 as previous


_BASE_METHODS = {"CHEAP_FIRST_NO_TARGET": "CHEAP_FIRST_SHARED_CONFIG",
                 "COST_MENU_NO_TARGET": "COST_MENU_SHARED_CONFIG",
                 "CHEAP_FIRST_TARGET": "CHEAP_FIRST_SHARED_CONFIG",
                 "COST_MENU_TARGET": "COST_MENU_SHARED_CONFIG"}
_TARGET_COUNTERS = ("target_checks", "target_orientation_checks", "target_collision_cell_checks", "target_cache_hits")


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _integer(value, name):
    _require(type(value) is int and value >= 0, "Invalid nonnegative integer: " + name)
    return value


def _occupancy(footprint, node):
    """Independent centered rectangle construction; no domain checker call."""
    x, y, orientation = node
    width, height = footprint
    if orientation == 1:
        width, height = height, width
    return {(cx, cy) for cx in range(x - width // 2, x + width // 2 + 1)
            for cy in range(y - height // 2, y + height // 2 + 1)}


def _invalid_cells(runtime, cells):
    grid = runtime["grid"]
    blocked = set(map(tuple, grid["blocked"]))
    ordered = sorted(cells)
    outside = [[x, y] for x, y in ordered
               if not (0 <= x < grid["width"] and 0 <= y < grid["height"])]
    obstructed = [[x, y] for x, y in ordered if (x, y) in blocked]
    return outside, obstructed


def _clear(runtime, cells):
    outside, blocked = _invalid_cells(runtime, cells)
    return not outside and not blocked


def _target_geometry(runtime, footprint, target):
    """Inspect every allowed pose, including those following a valid one."""
    poses = []
    for orientation in target["allowed_orientations"]:
        cells = _occupancy(footprint, (*target["cell"], orientation))
        outside, obstructed = _invalid_cells(runtime, cells)
        poses.append({"orientation": orientation, "valid": not outside and not obstructed,
                      "collision_cell_checks": len(cells), "outside_grid_cells": outside,
                      "blocked_cells": obstructed})
    return {"footprint": list(footprint), "target_id": target["id"],
            "target_cell": list(target["cell"]),
            "allowed_orientations": list(target["allowed_orientations"]),
            "checked_orientations": [pose["orientation"] for pose in poses],
            "all_allowed_orientations_checked": True,
            "all_terminal_poses_invalid": all(not pose["valid"] for pose in poses),
            "poses": poses}


def _successors(runtime, footprint, node):
    """Independent replay of the frozen discrete translation/square-turn rule."""
    x, y, orientation = node
    if footprint is None:
        neighbors = [(x - 1, y, 0), (x, y - 1, 0), (x, y + 1, 0), (x + 1, y, 0)]
        return [following for following in neighbors if _clear(runtime, {following[:2]})], 4, 4
    current = _occupancy(footprint, node)
    _require(_clear(runtime, current), "Paid loaded expansion starts at an invalid configuration")
    cells_visited, neighbors = len(current), []
    for dx, dy in ((1, 0), (0, 1), (-1, 0), (0, -1)):
        following = (x + dx, y + dy, orientation)
        swept = current | _occupancy(footprint, following)
        cells_visited += len(swept)
        if _clear(runtime, swept):
            neighbors.append(following)
    side = max(footprint)
    swept = _occupancy((side, side), (x, y, 0))
    cells_visited += len(swept)
    if _clear(runtime, swept):
        neighbors.append((x, y, 1 - orientation))
    return neighbors, cells_visited, 5


def _source_event(engine, event):
    """Validate source truth and actual cache-sensitive work, then install it."""
    source = event["source_id"]
    _require(source in engine.sources and source not in engine.certificates,
             "Illegal or duplicate paid source query")
    obj = engine.objects[source]
    footprint, node = tuple(obj["footprint"]), engine.source_node(source)
    failures = []
    if not obj["movable"]:
        failures.append("immovable_source")
    if obj["mass"] > engine.runtime["robot"]["payload"]:
        failures.append("payload_exceeded")
    collision_work = 0
    if not failures:
        graph = engine.graph(source)
        if node not in graph.parents:
            collision_work = footprint[0] * footprint[1]
            if not _clear(engine.runtime, _occupancy(footprint, node)):
                failures.append("invalid_initial_loaded_configuration")
        if not failures:
            graph.add(node)
    certificate = {"source_id": source, "passed": not failures,
                   "failed_conditions": failures,
                   "scope": "all_semantically_admissible_targets_of_source",
                   "world_sha256": engine.world_hash, "footprint": list(footprint),
                   "initial_node": list(node)}
    _require(all(event.get(key) == value for key, value in certificate.items()),
             "Source certificate differs from independent public-world check")
    engine.certificates[source] = certificate
    return 1 + footprint[0] * footprint[1], 1 + collision_work, collision_work, 0


def _graph_event(engine, event):
    operation = event["operation"]
    footprint = None if operation == "expand_empty" else tuple(event["footprint"])
    _require(operation in {"expand_empty", "expand_loaded"}, "Unknown paid operation")
    if footprint is None:
        graph, reserved = engine.empty, 4
    else:
        _require(footprint in engine.loaded, "Loaded graph has no paid source initialization")
        graph = engine.loaded[footprint]
        width, height = footprint
        reserved = 5 * width * height + 2 * (width + height) + max(width, height) ** 2
    node = tuple(event["node"])
    _require(event.get("scope") == graph.scope and
             event.get("footprint") == (list(footprint) if footprint else None),
             "Configuration expansion has wrong graph scope")
    _require(node in graph.parents and node not in graph.expanded,
             "Expansion uses an unpaid, fabricated, or already expanded node")
    neighbors, cells, transitions = _successors(engine.runtime, footprint, node)
    _require(event.get("neighbors") == [list(n) for n in neighbors],
             "Paid successors differ from independent collision/transition checks")
    discovered = []
    for following in neighbors:
        if following not in graph.parents:
            discovered.append(list(following))
            graph.add(following)
        graph.adjacency[node].add(following)
        graph.adjacency[following].add(node)
        graph.union(node, following)
    graph.expanded.add(node)
    _require(event.get("discovered") == discovered and
             event.get("component_id") == graph.component_id(node) and
             event.get("component_closed") == graph.closed(node),
             "Paid graph discovery/component evidence mismatch")
    return reserved, cells, cells, transitions


def _states(engine, target_keys, certificates):
    states, reasons = engine.states()
    for key, certificate in certificates.items():
        if not certificate["all_terminal_poses_invalid"]:
            continue
        footprint, target = key
        for source in target_keys[key]:
            identifier = goal_id({"action": "place", "source_id": source, "target_id": target})
            _require(states[identifier] != "FEASIBLE", "Target exclusion contradicts a paid feasible path")
            if states[identifier] == "UNKNOWN":
                states[identifier] = "INFEASIBLE"
                reasons[identifier] = {"kind": "target_terminal_necessary_condition", "footprint": list(footprint),
                                       "target_id": target, "query_id": certificate["query_id"]}
    return states, reasons


def _target_menu(engine, target_keys, certificates, states, remaining):
    menu = []
    for (footprint, target), sources in sorted(target_keys.items()):
        if (footprint, target) in certificates:
            continue
        allowed = engine.targets[target]["allowed_orientations"]
        reserve = footprint[0] * footprint[1] * len(allowed)
        affected = sum(states[goal_id({"action": "place", "source_id": source, "target_id": target})] == "UNKNOWN"
                       for source in sources)
        menu.append({"query_id": f"target:{footprint[0]}x{footprint[1]}:{target}",
                     "operation": "target_check", "footprint": list(footprint), "target_id": target,
                     "source_ids": list(sources), "allowed_orientations": list(allowed),
                     "affected_unknown_goals": affected, "reserved_work_units": reserve,
                     "canonical_tie": [-1, *footprint, target], "score_numerator": 1,
                     "score_denominator": 1, "priority_score": 1.0,
                     "affordable": remaining is None or reserve <= remaining})
    return menu


def _target_event(engine, event, target_keys, certificates):
    key = tuple(event["footprint"]), event["target_id"]
    _require(key in target_keys and key not in certificates,
             "Target certificate is outside semantics or duplicates a paid cache key")
    footprint, target = key
    geometry = _target_geometry(engine.runtime, footprint, engine.targets[target])
    certificate = {"query_id": f"target:{footprint[0]}x{footprint[1]}:{target}",
                   "footprint": list(footprint), "target_id": target,
                   "target_cell": geometry["target_cell"], "source_ids": list(target_keys[key]),
                   "allowed_orientations": geometry["allowed_orientations"],
                   "orientation_results": [{name: pose[name] for name in ("orientation", "valid", "collision_cell_checks")}
                                           for pose in geometry["poses"]],
                   "all_terminal_poses_invalid": geometry["all_terminal_poses_invalid"],
                   "world_sha256": engine.world_hash,
                   "scope": "necessary_terminal_occupancy_only; no_reachability_or_source_eligibility",
                   "cache_reuses": len(target_keys[key]) - 1}
    _require(all(event.get(name) == value for name, value in certificate.items()),
             "Target certificate differs from independent occupancy evidence")
    for pose in event["orientation_results"]:
        _require(type(pose["valid"]) is bool and type(pose["collision_cell_checks"]) is int,
                 "Target orientation result has invalid field types")
    _require(type(event["all_terminal_poses_invalid"]) is bool, "Invalid target exclusion flag")
    certificates[key] = certificate
    cells = sum(pose["collision_cell_checks"] for pose in geometry["poses"])
    return cells, cells, cells, len(geometry["poses"]), certificate["cache_reuses"]


def _base_menu(engine, base_method, enabled, states, remaining):
    # OFF is exactly Stage 2b. ON's cheap source-first control omits sources
    # whose goals have already been excluded by the common target prefix.
    if enabled and base_method == "CHEAP_FIRST_SHARED_CONFIG":
        counts = engine.unknown_counts(states)
        pending = [source for source in engine.sources if source not in engine.certificates and counts[source] > 0]
        if pending:
            from innovation_stage2b.policies import _source_query
            branch = "SOURCE_CONDITIONS_FIRST"
            candidates = [_source_query(engine, source, counts[source]) for source in pending]
        else:
            from innovation_stage2b.policies import _graph_query
            branch = "CHECKED_SOURCE_SEARCH"
            candidates = [_graph_query(graph, node, affected) for graph, node, affected in engine.search_jobs(counts)]
    else:
        branch, candidates = _candidate_menu(engine, base_method, states)
    feasible = sum(value == "FEASIBLE" for value in states.values())
    return branch, _rank_menu(base_method, candidates, feasible, remaining)


def validate_prediction(prediction: dict[str, Any], runtime: dict[str, Any], oracle: dict[str, Any],
                        budget: int | None):
    """Reject forged/unpaid evidence and return the frozen Stage 2 score schema."""
    try:
        return _validate_prediction(prediction, runtime, oracle, budget)
    except (KeyError, TypeError, IndexError, AttributeError) as exc:
        raise ValueError("Malformed Stage 2c prediction/evidence: " + str(exc)) from exc


def _validate_prediction(prediction, runtime, oracle, budget):
    _require(budget is None or type(budget) is int and budget >= 0, "Invalid prepaid budget")
    method = prediction["method"]
    _require(method in _BASE_METHODS, "Unknown Stage 2c method")
    base_method, enabled = _BASE_METHODS[method], not method.endswith("NO_TARGET")
    _require(prediction["base_method"] == base_method and prediction["target_precheck_enabled"] is enabled,
             "Stage 2c factorial method metadata mismatch")
    _require(prediction["budget"] == budget, "Prediction budget differs from requested budget")
    try:
        families = parse_families(runtime)
    except ValueError:
        _require(prediction["supported"] is False and prediction["decision"] == "ABSTAIN" and
                 not prediction["trace"] and not prediction["schedule_trace"] and
                 not prediction["goal_states"] and not prediction["source_certificates"] and
                 not prediction["target_certificates"] and not prediction["closed_components"] and
                 prediction["selected_goal"] is None and prediction["selected_candidate"] is None and
                 not prediction["witness"] and all(type(value) is int and value == 0 for value in prediction["cost"].values()),
                 "Unsupported runtime must preserve a zero-work abstention")
        _require(prediction["remaining_budget"] == budget and prediction["budget_blocked"] is False,
                 "Unsupported runtime has an invalid budget ledger")
        return previous.score_prediction(prediction, runtime, oracle)
    _require(prediction["supported"] is True, "A supported runtime was marked unsupported")
    engine = _Engine(runtime, families)
    _require(prediction["world_sha256"] == engine.world_hash, "Evidence is not bound to the public world")
    _require(prediction["families"] == families, "Public semantic families were lost or changed")
    target_keys = {}
    if enabled:
        for source in engine.sources:
            footprint = tuple(engine.objects[source]["footprint"])
            for target in engine.families[source]:
                target_keys.setdefault((footprint, target), []).append(source)
    target_certificates, counters = {}, Counter()
    trace, schedules = prediction["trace"], prediction["schedule_trace"]
    _require(isinstance(trace, list) and isinstance(schedules, list), "Paid trace and schedule must be lists")
    reserved = actual = cursor = 0
    prefix_active, skipped, blocked = enabled, False, False
    for index, entry in enumerate(schedules):
        states, _ = _states(engine, target_keys, target_certificates)
        decision, _ = _decision(states)
        _require(states and decision == "ABSTAIN" and not blocked, "Schedule continues after a certified or empty decision")
        remaining = None if budget is None else budget - reserved
        feasible = sum(value == "FEASIBLE" for value in states.values())
        phase = {"known_feasible_goals": feasible, "phase_cap": max(1, 2 - feasible),
                 "current_evidence_decision": decision}
        _require(entry["step"] == cursor + 1 and entry["world_sha256"] == engine.world_hash and
                 entry["decision_phase"] == phase and entry["remaining_budget_before"] == remaining and
                 entry["reserved_work_before"] == reserved,
                 "Schedule phase/world/prepayment history mismatch")
        if enabled:
            _require(entry["schedule_index"] == index + 1, "Schedule index mismatch")
        ranked = _target_menu(engine, target_keys, target_certificates, states, remaining) if prefix_active else []
        target_phase = bool(ranked)
        if target_phase:
            branch = "FIXED_TARGET_TERMINAL_PRECHECK"
        else:
            prefix_active = False
            branch, ranked = _base_menu(engine, base_method, enabled, states, remaining)
        _require(ranked and entry["menu_scope"] == branch and entry["candidate_menu"] == _public_menu(ranked),
                 "Schedule menu differs from the complete canonical paid-evidence menu")
        _require(entry["skipped_unaffordable_query_ids"] == [q["query_id"] for q in ranked if not q["affordable"]],
                 "Unaffordable-query log mismatch")
        chosen = next((query for query in ranked if query["affordable"]), None)
        if chosen is None:
            outcome = "precheck_budget_skipped_continue_base" if target_phase else "budget_blocked"
            _require(entry["outcome"] == outcome and entry["selected_query_id"] is None and
                     entry["reserved_work_delta"] == 0 and entry["actual_checker_work_delta"] == 0 and
                     entry["remaining_budget_after"] == remaining, "Invalid zero-work scheduling event")
            if target_phase:
                _require(index < len(schedules) - 1, "Unaffordable target prefix did not continue to the base menu")
                prefix_active, skipped = False, True
            else:
                _require(index == len(schedules) - 1, "Budget-blocked schedule is not terminal")
                blocked = True
            continue
        _require(cursor < len(trace), "Scheduled query has no paid evidence event")
        event = trace[cursor]
        _require(event["operation"] == chosen["operation"] and entry["outcome"] == "queried" and
                 entry["selected_query_id"] == chosen["query_id"] and event["query_id"] == chosen["query_id"] and
                 event["world_sha256"] == engine.world_hash and type(event["cost_units"]) is int and event["cost_units"] == 1,
                 "Paid event does not match the selected bound query")
        for name in ("source_id", "target_id", "footprint", "node"):
            if name in chosen:
                _require(event.get(name) == chosen[name], "Paid query operands differ from its schedule")
        estimate = _integer(chosen["reserved_work_units"], "query reservation")
        _require(budget is None or reserved + estimate <= budget, "Reserved checker budget exceeded before a query")
        # Only independently grounded events are installed into the fresh graph.
        if event["operation"] == "target_check":
            _require(enabled and target_phase, "Target query outside the fixed paid prefix")
            expected, measured, cells, orientations, reuses = _target_event(engine, event, target_keys, target_certificates)
            counters.update(target_checks=1, target_orientation_checks=orientations,
                            target_collision_cell_checks=cells, target_cache_hits=reuses)
            transitions = 0
        elif event["operation"] == "source_check":
            expected, measured, cells, transitions = _source_event(engine, event)
            counters["source_checks"] += 1
        else:
            expected, measured, cells, transitions = _graph_event(engine, event)
            counters["empty_node_expansions" if event["operation"] == "expand_empty" else "loaded_node_expansions"] += 1
        _require(estimate == expected and _integer(event["reserved_work_units"], "paid reservation") == expected and
                 _integer(event["actual_checker_work_delta"], "actual paid work") == measured and
                 entry["reserved_work_delta"] == expected and entry["actual_checker_work_delta"] == measured and
                 0 <= measured <= expected, "Per-query reservation/actual checker-work ledger mismatch")
        reserved += expected
        actual += measured
        counters.update(collision_cell_checks=cells, transition_checks=transitions, total_units=1)
        _require(entry["remaining_budget_after"] == (None if budget is None else budget - reserved),
                 "Post-query budget ledger mismatch")
        cursor += 1
    _require(cursor == len(trace), "Paid trace has missing scheduling entries")
    states, reasons = _states(engine, target_keys, target_certificates)
    decision, selected_goal = _decision(states)
    _require(prediction["goal_states"] == states and prediction["goal_evidence"] == reasons,
             "Replayed paid evidence disagrees with final goal states/reasons")
    _require(prediction["decision"] == decision and prediction["selected_goal"] == selected_goal,
             "Final decision is inconsistent with paid evidence")
    _require(prediction["candidate_count"] == len(states) and prediction["family_count"] == len(engine.families),
             "Semantic candidate/family counts mismatch")
    _require(not states or decision != "ABSTAIN" or blocked,
             "An unresolved task stopped without exhausting affordable evidence")
    _require(prediction["source_certificates"] == list(engine.certificates.values()),
             "Source certificates are missing, forged, or unbacked by paid events")
    _require(prediction["target_certificates"] == list(target_certificates.values()),
             "Target certificates are missing, forged, or unbacked by paid events")
    _require(prediction["target_keys_total"] == len(target_keys) and
             prediction["target_precheck_complete"] is ((len(target_certificates) == len(target_keys)) if enabled else None) and
             prediction["target_precheck_budget_skipped"] is skipped and prediction["budget_blocked"] is blocked and
             prediction["remaining_budget"] == (None if budget is None else budget - reserved),
             "Terminal prefix/cache/budget metadata mismatch")
    expected_reason = "EMPTY_SEMANTICS" if not states else "decision_certified" if decision != "ABSTAIN" else "budget_no_affordable_query"
    _require(prediction["reason"] == expected_reason, "Terminal stopping reason mismatch")
    if selected_goal is None:
        _require(prediction["selected_candidate"] is None and prediction["witness"] == [],
                 "A non-execution decision contains an execution witness")
    else:
        candidate = next({"action": "place", "source_id": source, "target_id": target}
                         for source in engine.sources for target in engine.families[source]
                         if goal_id({"action": "place", "source_id": source, "target_id": target}) == selected_goal)
        _require(prediction["selected_candidate"] == candidate and
                 prediction["witness"] == engine.witness(candidate, reasons[selected_goal]),
                 "Selected witness is not grounded in the paid graph")
    graphs = [engine.empty] + [engine.loaded[key] for key in sorted(engine.loaded)]
    closed = [certificate for graph in graphs for certificate in graph.closed_certificates()]
    _require(prediction["closed_components"] == closed, "Closed-component certificates differ from paid graph replay")
    expected_graph_counts = {"footprint_graphs": len(engine.loaded), "known_nodes": sum(len(g.parents) for g in graphs),
                             "expanded_nodes": sum(len(g.expanded) for g in graphs), "closed_components": len(closed)}
    _require(all(prediction["graph_statistics"].get(key) == value for key, value in expected_graph_counts.items()),
             "Graph statistics contain unpaid graph nodes or inconsistent components")
    cost = prediction["cost"]
    counters["reserved_checker_work_units"] = reserved
    counters["actual_checker_work_units"] = actual
    for key in ("source_checks", "empty_node_expansions", "loaded_node_expansions", "total_units",
                "collision_cell_checks", "transition_checks", "reserved_checker_work_units", "actual_checker_work_units", *_TARGET_COUNTERS):
        _require(_integer(cost[key], key) == counters[key], "Aggregate checker-work ledger mismatch: " + key)
    _require(actual == cost["source_checks"] + cost["collision_cell_checks"] and actual <= reserved,
             "Actual work exceeds or disagrees with paid ledger")
    _integer(cost["cache_hits"], "cache_hits")
    _require(_integer(cost["graph_traversals"], "graph_traversals") == engine.aux["graph_traversals"],
             "Witness route traversal count mismatch")
    # Preserve the original semantic/oracle/witness checks. Its legacy budget
    # denotes source/node query count, so target events and their overlapping
    # collision work are removed only from this validation view.
    legacy = copy.deepcopy(prediction)
    legacy["trace"] = [event for event in legacy["trace"] if event["operation"] != "target_check"]
    legacy["cost"]["total_units"] -= cost["target_checks"]
    legacy["cost"]["collision_cell_checks"] -= cost["target_collision_cell_checks"]
    return previous.validate_prediction(legacy, runtime, oracle,
                                        None if budget is None else legacy["cost"]["total_units"])
