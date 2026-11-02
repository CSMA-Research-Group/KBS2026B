"""Two finite domains; construction, runtime solvers and separate reference search.

No model outcome or reference search selects a world. Workflow references use
frozenset states; runtime search uses independently compiled integer states.
This is a second constructed task family, not real-robot or natural-language validation.
"""
from collections import deque
import copy
import random
import re
from adaptive_experiments.common import stable
from counterfactual_pilot.core import make_cases as old_cases, VARIANTS
from counterfactual_pilot.solver import run_contract
from counterfactual_pilot.validation import validate_contract_prediction
from innovation_stage2.domain import evaluate_ast, goal_id, validate_witness, render_instruction
from innovation_stage2.reference import reference_oracle
from innovation_stage3.protocol import families
from paper_batch.generator import catalog, physical_fingerprint

VERSION = "reviewer-validation-v1"
CONTRACTS = ("referential", "any_valid")


def short_goal(goal):
    return goal["source_id"] + "->" + goal["target_id"]


def workflow_world(seed, index, sanity=False):
    rng = random.Random(f"workflow:{seed}:{index}")
    phases = 3 + index % 4
    noise_count = 3 + (index // 4) % 4
    names = ["f" + str(x) for x in rng.sample(range(100, 999), phases+3+noise_count)]
    chain = names[:phases]
    s, a, ready = chain[0], chain[1], chain[-1]
    x, y, impossible = names[phases:phases+3]
    noise = names[phases+3:]
    actions = []
    edges = list(zip(chain, chain[1:]))+[(ready, x), (ready, y)]
    if not sanity:
        edges += [(chain[i], chain[j]) for i in range(phases) for j in range(i+2, phases) if rng.random() < .5]
    for pre, post in edges:
        actions.append({"id": "", "requires": [pre], "delete": [pre], "add": [post]})
    if not sanity:
        for i, fact in enumerate(noise):
            prerequisites = rng.sample(noise[:i], rng.randrange(min(i, 2)+1))
            actions.extend([
                {"id": "", "requires": prerequisites, "delete": [], "add": [fact]},
                {"id": "", "requires": [fact], "delete": [fact], "add": []}])
    initial = [s] + ([] if sanity else [f for f in noise if rng.random() < .5])
    primary_requirements = [x] + ([] if sanity else rng.sample(noise, rng.randrange(3)))
    added_requirements = [y] + ([] if sanity else rng.sample(noise, rng.randrange(3)))
    normalized = {f: i for i, f in enumerate(names)}
    topology = stable({"initial": sorted(normalized[f] for f in initial),
        "actions": sorted([{k: sorted(normalized[f] for f in action[k]) for k in ("requires", "add", "delete")} for action in actions], key=stable),
        "primary": sorted(normalized[f] for f in primary_requirements),
        "alternative": sorted(normalized[f] for f in added_requirements)})
    rng.shuffle(actions)
    for i, action in enumerate(actions):
        action["id"] = "a" + str(i)
    tags = rng.sample(["red", "blue", "green", "amber", "purple", "white"], 3)
    ids = ["g" + str(i) for i in rng.sample(range(10, 99), 6)]
    goals = [
        {"id": ids[0], "tag": tags[0], "kind": "finished", "requires": primary_requirements},
        {"id": ids[1], "tag": tags[1], "kind": "finished", "requires": [impossible]},
        {"id": ids[2], "tag": tags[1], "kind": "finished", "requires": [s, x]},
        {"id": ids[3], "tag": tags[2], "kind": "finished", "requires": [y]},
        {"id": ids[4], "tag": tags[0], "kind": "intermediate", "requires": [a]},
    ]
    if sanity:
        goals = goals[:1]
    rng.shuffle(goals)
    runtime = {"instruction": f"Complete one endpoint whose tag is {tags[0]} or {tags[1]} and whose kind is finished.",
               "facts": sorted(names), "initial": initial, "actions": actions, "goals": goals}
    return runtime, {"tags": tags[:2], "kind": "finished", "structure_sha256": topology}, ids[0], {
        "id": ids[5], "tag": tags[0], "kind": "finished", "requires": added_requirements}, impossible


def make_cases(config):
    n = sum(config["split_worlds_per_domain"].values())
    assert n == 144
    roles = [role for role, count in config["split_worlds_per_domain"].items() for _ in range(count)]
    random.Random(config["split_seed"]).shuffle(roles)
    grid = []
    for block in range(n // 24):
        grid.extend(c for c in old_cases({"base_worlds": 24, "world_seed": config["world_seed"] + block})
                    if c["contract"] == "referential")
    cases = []
    for index, split in enumerate(roles):
        cluster = f"rv_carry_{config['world_seed']}_{index:03d}"
        group = grid[index*3:index*3+3]
        # Short runtime IDs save GPU tokens; permutation is fixed without references.
        all_objects = sorted(o["id"] for o in group[0]["runtime"]["objects"])
        all_targets = sorted({t["id"] for c in group for t in c["runtime"]["targets"]})
        names = {v: "o"+str(i) for i, v in enumerate(all_objects)}
        names.update({v: "t"+str(i) for i, v in enumerate(all_targets)})
        for old in group:
            if split in ("knowledge", "sanity") and old["variant"] != "base":
                continue
            runtime = copy.deepcopy(old["runtime"])
            for key in ("objects", "targets"):
                for obj in runtime[key]:
                    obj["id"] = names[obj["id"]]
            primary_old = old["primary_goal"][6:].split("->")
            added_old = old["added_goal"][6:].split("->")
            primary = names[primary_old[0]] + "->" + names[primary_old[1]]
            added = names[added_old[0]] + "->" + names[added_old[1]]
            ast = copy.deepcopy(old["ast"])
            if split == "sanity":
                runtime["objects"] = [o for o in runtime["objects"] if o["id"] == primary.split("->")[0]]
                runtime["targets"] = [t for t in runtime["targets"] if t["id"] == primary.split("->")[1]]
                runtime["grid"]["blocked"] = []
                obj, target = runtime["objects"][0], runtime["targets"][0]
                obj.update(footprint=[1, 1], orientation=0, mass=1, movable=True)
                ast = {"action": "place", "pairing": "compatible",
                    "source": {"op": "eq", "field": "color", "value": obj["color"]},
                    "target": {"op": "eq", "field": "color", "value": target["color"]}}
                runtime["instruction"] = render_instruction(ast)
            cases.append({"domain": "carrying", "cluster_id": cluster,
                "instance_id": cluster + "_" + old["variant"], "split": split, "variant": old["variant"],
                "runtime": runtime, "ast": ast, "expected_feasible": [primary, added] if old["variant"] == "add_feasible" else [primary]})
        cluster = f"rv_flow_{config['world_seed']}_{index:03d}"
        base, ast, primary, addition, impossible = workflow_world(config["world_seed"], index, split == "sanity")
        for variant in VARIANTS:
            if split in ("knowledge", "sanity") and variant != "base":
                continue
            runtime = copy.deepcopy(base)
            if variant != "base":
                extra = copy.deepcopy(addition)
                if variant == "add_infeasible":
                    extra["requires"] = [impossible]
                runtime["goals"].insert(index % (len(runtime["goals"])+1), extra)
            cases.append({"domain": "workflow", "cluster_id": cluster,
                "instance_id": cluster + "_" + variant, "split": split, "variant": variant,
                "runtime": runtime, "ast": ast,
                "expected_feasible": [primary, addition["id"]] if variant == "add_feasible" else [primary]})
    assert len(cases) == 800 and len({c["instance_id"] for c in cases}) == 800
    probes = set()
    for domain in ("carrying", "workflow"):
        test = sorted({c["cluster_id"] for c in cases if c["domain"] == domain and c["split"] == "test"},
                      key=lambda x: stable([config["order_seed"], x]))
        probes.update(test[:config["contract_prompt_probe_worlds_per_domain"]])
    for c in cases:
        c["contract_prompt_probe"] = c["cluster_id"] in probes
    return cases


def public_catalog(case):
    r = case["runtime"]
    if case["domain"] == "carrying":
        return sorted(short_goal(g) for g in catalog(r))
    match = re.fullmatch(r"Complete one endpoint whose tag is (\w+) or (\w+) and whose kind is (\w+)\.", r["instruction"])
    if not match:
        raise ValueError("Unsupported public workflow instruction")
    a, b, kind = match.groups()
    return sorted(g["id"] for g in r["goals"] if g["tag"] in (a, b) and g["kind"] == kind)


def known_ids(case):
    r = case["runtime"]
    return ({short_goal({"source_id": s["id"], "target_id": t["id"]})
             for s in r["objects"] for t in r["targets"]} if case["domain"] == "carrying"
            else {g["id"] for g in r["goals"]})


def workflow_reference(case, limit):
    r = case["runtime"]
    start = frozenset(r["initial"])
    parents = {start: None}
    queue = deque([start])
    while queue:
        state = queue.popleft()
        for action in r["actions"]:
            if set(action["requires"]).issubset(state):
                child = frozenset((state - set(action["delete"])) | set(action["add"]))
                if child not in parents:
                    if len(parents) >= limit:
                        raise RuntimeError("Reference bound exceeded; no replacement world allowed")
                    parents[child] = (state, action["id"])
                    queue.append(child)
    semantic = sorted(g["id"] for g in r["goals"]
        if g["tag"] in case["ast"]["tags"] and g["kind"] == case["ast"]["kind"])
    witnesses = {}
    for goal in r["goals"]:
        if goal["id"] not in semantic:
            continue
        end = next((s for s in parents if set(goal["requires"]).issubset(s)), None)
        if end is not None:
            path = []
            while parents[end] is not None:
                end, aid = parents[end]
                path.append(aid)
            witnesses[goal["id"]] = list(reversed(path))
    return {"complete": True, "semantic": semantic, "feasible": sorted(witnesses),
            "witnesses": witnesses, "closed_states": sorted([sorted(s) for s in parents])}


def reference(case, config):
    if case["domain"] == "workflow":
        ref = workflow_reference(case, config["reference_max_nodes"])
    else:
        old = reference_oracle(case["runtime"], case["ast"], config["reference_max_nodes"])
        if not old["complete"]:
            raise RuntimeError("Carrying reference incomplete; retain and stop")
        ref = {"complete": True, "semantic": sorted(short_goal(g) for g in old["semantic_candidates"]),
               "feasible": sorted(g[6:] for g in old["feasible_goals"]),
               "witnesses": {g[6:]: w for g, w in old["witnesses"].items()}, "raw": old}
    if ref["semantic"] != public_catalog(case) or set(ref["feasible"]) != set(case["expected_feasible"]):
        raise ValueError("Construction/public parser/independent reference mismatch")
    for goal, path in ref["witnesses"].items():
        if not witness_valid(case, goal, path):
            raise ValueError("Reference witness did not replay")
    return ref


def witness_valid(case, goal, path):
    r = case["runtime"]
    if case["domain"] == "carrying":
        source, target = goal.split("->")
        return validate_witness(r, {"action": "place", "source_id": source, "target_id": target}, path)
    actions = {a["id"]: a for a in r["actions"]}
    state = set(r["initial"])
    for aid in path:
        if aid not in actions or not set(actions[aid]["requires"]).issubset(state):
            return False
        state.difference_update(actions[aid]["delete"])
        state.update(actions[aid]["add"])
    target = next(g for g in r["goals"] if g["id"] == goal)
    return set(target["requires"]).issubset(state)


def rule(states, mode):
    good = sorted(g for g, v in states.items() if v == "FEASIBLE")
    if mode == "any_valid" and good:
        return "EXECUTE", good[0]
    if mode == "referential" and len(good) >= 2:
        return "CLARIFY", None
    if len(good) == 1 and all(v != "UNKNOWN" for v in states.values()):
        return "EXECUTE", good[0]
    if states and all(v == "INFEASIBLE" for v in states.values()):
        return "REJECT", None
    return "ABSTAIN", None


def solve(case, goals, budget, mode):
    goals = sorted(set(goals))
    if not set(goals).issubset(public_catalog(case)):
        raise ValueError("Runtime solver received inadmissible goal")
    if case["domain"] == "carrying":
        expanded = [{"action": "place", "source_id": g.split("->")[0], "target_id": g.split("->")[1]} for g in goals]
        prediction = run_contract(case["runtime"], families(expanded), budget, mode=mode)
        validate_contract_prediction(prediction, case["runtime"], families(expanded), budget, mode)
        return {"decision": prediction["decision"],
                "selected_goal": prediction["selected_goal"][6:] if prediction["selected_goal"] else None,
                "work_units": prediction["cost"]["actual_checker_work_units"], "raw": prediction}
    r = case["runtime"]
    bits = {v: 1 << i for i, v in enumerate(sorted(r["facts"]))}
    mask = lambda xs: sum(bits[x] for x in xs)
    actions = [(a["id"], mask(a["requires"]), mask(a["delete"]), mask(a["add"])) for a in r["actions"]]
    requirements = {g["id"]: mask(g["requires"]) for g in r["goals"] if g["id"] in goals}
    start = mask(r["initial"])
    parents, queue, reached = {start: None}, deque([start]), {}
    work, exhausted, stopped = 0, False, False
    while queue and not stopped:
        node = queue.popleft()
        for goal, required in requirements.items():
            if node & required == required:
                reached.setdefault(goal, node)
        states = {g: "FEASIBLE" if g in reached else "UNKNOWN" for g in goals}
        if rule(states, mode)[0] != "ABSTAIN":
            stopped = True
            break
        for aid, pre, delete, add in actions:
            if budget is not None and work >= budget:
                exhausted, stopped = True, True
                break
            work += 1
            if node & pre == pre:
                child = (node & ~delete) | add
                if child not in parents:
                    parents[child] = (node, aid)
                    queue.append(child)
    closed = not queue and not exhausted and not stopped
    # With no goals the rule abstains; it does not infer global infeasibility.
    states = {g: "FEASIBLE" if g in reached else "INFEASIBLE" if closed else "UNKNOWN" for g in goals}
    decision, selected = rule(states, mode)
    witnesses = {}
    for goal, node in reached.items():
        path = []
        while parents[node] is not None:
            node, aid = parents[node]
            path.append(aid)
        witnesses[goal] = list(reversed(path))
    if any(not witness_valid(case, g, p) for g, p in witnesses.items()):
        raise ValueError("Runtime witness invalid")
    if closed:
        decoded = {frozenset(f for f, bit in bits.items() if node & bit) for node in parents}
        if frozenset(r["initial"]) not in decoded:
            raise ValueError("Closure lacks start")
        for state in decoded:
            for action in r["actions"]:
                if set(action["requires"]).issubset(state):
                    if frozenset((state-set(action["delete"])) | set(action["add"])) not in decoded:
                        raise ValueError("Runtime negative certificate is not closed")
        for goal, verdict in states.items():
            if verdict == "INFEASIBLE":
                req = next(g["requires"] for g in r["goals"] if g["id"] == goal)
                if any(set(req).issubset(s) for s in decoded):
                    raise ValueError("Negative certificate contains a goal")
    return {"decision": decision, "selected_goal": selected, "work_units": work,
            "raw": {"goal_states": states, "witnesses": witnesses, "closed": closed,
                    "visited_integer_states": sorted(parents), "facts_in_bit_order": sorted(bits),
                    "budget": budget, "budget_exhausted": exhausted, "cost_unit": "action_precondition_test"}}


def score(view, ref, mode):
    feasible = set(ref["feasible"])
    expected = ("EXECUTE" if feasible else "REJECT") if mode == "any_valid" else (
        "CLARIFY" if len(feasible) > 1 else "EXECUTE" if feasible else "REJECT")
    execute = view["decision"] == "EXECUTE"
    correct = execute and view["selected_goal"] in feasible and (mode == "any_valid" or len(feasible) == 1)
    return {"n_feasible": len(feasible), "expected_decision": expected,
            "correct_execution": bool(correct), "execution_error": bool(execute and not correct),
            "correct_decision": view["decision"] == expected and (not execute or correct),
            "selected_feasible": view.get("selected_goal") in feasible if execute else None}
