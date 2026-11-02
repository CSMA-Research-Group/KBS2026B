"""E5 (Experiment E): shared-search certificates inside the calibrated language pipeline.

Minimal integration prototype.  For each available held-out world and policy envelope (L,U):
  PG  per-goal lazy checking exactly as in the paper (saved per-goal checker records, frozen
      order, legacy actual_checker_work_units)
  SH  the project's existing shared goal-family GBFS (counterfactual_pilot.solver.run_contract)
      run once over U grouped by source object, with the calibrated Theorem-2 stopping rule
      injected (carrying only); same legacy cost unit, so PG and SH costs are comparable
  IND per-goal vs shared closure in the audit's independent physics (one uniform unit:
      configurations expanded / action tests), for both domains
Checks: certified decisions identical (they must be: both are sound and run to determination),
witness validity for executions, and cost ratios.  Cost of failed/unused search is charged.
Called by code/project/replay_costs.py; outputs are written under work/cost_replays/.
"""
import csv
import json
import os
import sys
import time
from collections import deque, defaultdict
from datetime import datetime, timezone
from fractions import Fraction as F
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[1] / 'code'))
import ra_core as rc  # noqa: E402
import ra_calibration as cal  # noqa: E402
import physics_independent as phy  # noqa: E402
import counterfactual_pilot.solver as solver  # noqa: E402
from innovation_stage2.domain import validate_witness  # noqa: E402

_original_rule = solver.decision_rule


def shared_gbfs(runtime, L, U):
    """Run the existing shared GBFS once over U with the calibrated stopping rule injected."""
    if not U:
        return {'decision': 'REJECT', 'selected': None, 'work': 0, 'states': {}}
    Lk = {'place:' + g for g in L}
    Uk = {'place:' + g for g in U}

    def rule(states):
        W = {k for k, v in states.items() if v == 'FEASIBLE'}
        N = {k for k, v in states.items() if v == 'INFEASIBLE'}
        d, sel = rc.decide_sets(Lk, Uk, W, N)
        if d != 'ABSTAIN':
            return d, sel
        if all(v != 'UNKNOWN' for v in states.values()):
            return 'DONE', None
        return 'ABSTAIN', None
    by_source = defaultdict(list)
    for g in U:
        s, t = g.split('->')
        by_source[s].append(t)
    fams = [{'source_id': s, 'target_ids': sorted(set(ts))} for s, ts in sorted(by_source.items())]
    solver.decision_rule = lambda mode: rule
    try:
        res = solver.run_contract(runtime, fams, None, mode='calibrated')
    finally:
        solver.decision_rule = _original_rule
    states = res['goal_states']
    assert set(states) == Uk
    W = {k[6:] for k, v in states.items() if v == 'FEASIBLE'}
    N = {k[6:] for k, v in states.items() if v == 'INFEASIBLE'}
    d, sel = rc.decide_sets(L, U, W, N)
    if d == 'EXECUTE':
        s, t = sel.split('->')
        assert res['decision'] == 'EXECUTE' and res['selected_goal'] == 'place:' + sel
        assert validate_witness(runtime, {'action': 'place', 'source_id': s, 'target_id': t}, res['witness'])
    return {'decision': d, 'selected': sel, 'work': res['cost']['actual_checker_work_units'],
            'resolved': len(W) + len(N), 'W': W, 'N': N}


# ---- independent physics, per-goal vs shared, one unit -------------------------------

def ind_per_goal_cost_carrying(runtime, goal):
    s, t = goal.split('->')
    feas, work = phy.carrying_feasible(runtime, sources={s})
    # Per-goal search with early exit: rerun a goal-directed BFS that stops at the target.
    return goal in feas, _carrying_goal_bfs(runtime, s, t)


def _carrying_goal_bfs(runtime, sid, tid):
    g = runtime['grid']
    W, H = g['width'], g['height']
    blocked = {tuple(c) for c in g['blocked']}
    free = lambda c: 0 <= c[0] < W and 0 <= c[1] < H and c not in blocked
    obj = next(o for o in runtime['objects'] if o['id'] == sid)
    tgt = next(t for t in runtime['targets'] if t['id'] == tid)
    start = tuple(runtime['robot']['start'])
    work = 0
    seen = {start}
    q = deque([start])
    cell = tuple(obj['cell'])
    while q:
        x, y = q.popleft()
        work += 1
        if (x, y) == cell:
            break
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n not in seen and free(n) and free((x, y)):
                seen.add(n)
                q.append(n)
    fp = obj['footprint']
    if not (obj['movable'] and obj['mass'] <= runtime['robot']['payload'] and cell in seen
            and all(free(c) for c in phy._cells(fp, cell[0], cell[1], obj['orientation']))):
        return work
    s0 = (cell[0], cell[1], obj['orientation'])
    reach = {s0}
    q = deque([s0])
    side = max(fp)
    while q:
        x, y, o = q.popleft()
        work += 1
        if [x, y] == tgt['cell'] and o in tgt['allowed_orientations']:
            return work
        here = set(phy._cells(fp, x, y, o))
        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            n = (nx, ny, o)
            if n not in reach and all(free(c) for c in here | set(phy._cells(fp, nx, ny, o))):
                reach.add(n)
                q.append(n)
        r = side // 2
        n = (x, y, 1 - o)
        if n not in reach and all(free((x + dx, y + dy)) for dx in range(-r, r + 1) for dy in range(-r, r + 1)):
            reach.add(n)
            q.append(n)
    return work


def _workflow_goal_bfs(runtime, gid):
    req = set(next(gl['requires'] for gl in runtime['goals'] if gl['id'] == gid))
    init = frozenset(runtime['initial'])
    seen = {init}
    q = deque([init])
    work = 0
    while q:
        s = q.popleft()
        if req <= s:
            return True, work
        for a in runtime['actions']:
            work += 1
            if set(a['requires']) <= s:
                n = frozenset((s - set(a['delete'])) | set(a['add']))
                if n not in seen:
                    seen.add(n)
                    q.append(n)
    return False, work


def ind_lazy(public, L, U, shared):
    """Lazy evidence acquisition with the independent physics.  shared=True: a query on goal g
    computes the class-level closure that resolves every goal sharing g's source (carrying) or
    every goal (workflow), and all resolved goals become evidence."""
    rt = public['runtime']
    order = sorted(L) + sorted(set(U) - set(L))
    Wset, Nset, work, closures = set(), set(), 0, 0
    for g in order:
        d, sel = rc.decide_sets(L, U, Wset, Nset)
        if d != 'ABSTAIN':
            break
        if g in Wset or g in Nset:
            continue
        if public['domain'] == 'carrying':
            s = g.split('->')[0]
            if shared:
                feas, wk = phy.carrying_feasible(rt, sources={s})
                work += wk['configurations_expanded']
                closures += 1
                for h in U:
                    if h.split('->')[0] == s:
                        (Wset if h in feas else Nset).add(h)
            else:
                feas, _ = phy.carrying_feasible(rt, sources={s})
                work += _carrying_goal_bfs(rt, s, g.split('->')[1])
                (Wset if g in feas else Nset).add(g)
        else:
            if shared:
                feas, wk = phy.workflow_feasible(rt)
                work += wk['action_tests']
                closures += 1
                for h in U:
                    (Wset if h in feas else Nset).add(h)
            else:
                f, wk = _workflow_goal_bfs(rt, g)
                work += wk
                (Wset if f else Nset).add(g)
    d, sel = rc.decide_sets(L, U, Wset, Nset)
    return {'decision': d, 'selected': sel, 'work': work, 'closures': closures}


def cell_rows(args):
    idx, model, domain, rep, cap, thr = args
    b = rc.Batch(check_index_dir=Path(idx))
    checks = b.checks()
    resp = b.responses(model)
    rows = []
    tests = [e['case_id'] for e in b.cases('test', domain, 'primary')]
    t = next((x for x in thr if x['model'] == model and x['domain'] == domain and x['representation'] == rep and x['alpha'] == '0.05'), None)
    if t is None:
        return rows
    qdec = F(t['q_decision'])
    n = 0
    for cid in tests:
        rec = cal.world_record(b, resp, cid, (rep,))
        if rec is None:
            continue
        n += 1
        if cap and n > cap:
            break
        pub = b.label(cid)['public']
        for pol, (L, U) in [('UNION', (rec['union'], rec['union'])), ('DECISION_CALIBRATED', cal.env_q(rec, qdec)),
                            ('CATALOG_ORACLE', (rec['S'], rec['S'])), ('FULL_UNIVERSE', (frozenset(), frozenset(rec['G'])))]:
            if not rec['valid_any'] and pol not in ('CATALOG_ORACLE', 'FULL_UNIVERSE'):
                continue
            pg = rc.lazy(L, U, lambda g, c=checks[cid]: (c[g][0], c[g][1]))
            row = {'model': model, 'domain': domain, 'representation': rep, 'case_id': cid, 'policy': pol,
                   'L': len(L), 'U': len(U), 'E': len(rec['E']),
                   'pg_decision': pg['decision'], 'pg_queries': pg['queries'], 'pg_work_legacy': pg['work']}
            if domain == 'carrying':
                sh = shared_gbfs(pub['runtime'], L, U)
                row.update({'sh_decision': sh['decision'], 'sh_work_legacy': sh['work'],
                            'sh_equal': (sh['decision'], sh['selected']) == (pg['decision'], pg['selected'])})
            ip = ind_lazy(pub, L, U, shared=False)
            isd = ind_lazy(pub, L, U, shared=True)
            row.update({'ind_pg_work': ip['work'], 'ind_sh_work': isd['work'], 'ind_sh_closures': isd['closures'],
                        'ind_equal': (ip['decision'], ip['selected']) == (isd['decision'], isd['selected']) == (pg['decision'], pg['selected'])})
            rows.append(row)
    return rows
