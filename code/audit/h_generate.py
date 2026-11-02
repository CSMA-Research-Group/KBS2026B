#!/usr/bin/env python3
"""Experiment H (decision-relevance manipulation): build the scenes, labels, physical references and per-goal
checks for a carrying arm in which a fraction pi of distractor objects is made physically feasible.

Everything else is the frozen generator (`language_cases.make_cluster`, same seeds, same instructions, same hidden
specifications).  For each selected distractor object: movable=True, mass=1, footprint [1,1], orientation 0, and
the anchor is moved to a free cell that the robot can reach empty-handed (seeded choice).  Semantic labels are
recomputed with the generator's independent evaluator (proximity restrictions depend on the new cell).  Physical
labels come from the frozen reference search; every goal is also checked with the frozen per-goal checker, and the
audit's independent physics must agree (else the scene is reported and the run stops).

Output layout mirrors the frozen batch; analysis code reads it with AUDIT_BATCH:
  <out>/language/{case_manifest.json, labels/, public/, reference/, ready/}   <out>/checks/checks_<k>.jsonl.gz
  python3 code/audit/h_generate.py --out work/H_<id>/batch --pi 0.25 [--workers 10] [--limit-test N]
"""
import argparse
import gzip
import hashlib
import json
import os
import random
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
for p in [ROOT / 'code' / 'revision_20261022', ROOT / 'code', HERE]:
    sys.path.insert(0, str(p))
sys.dont_write_bytecode = True


def reachable_cells(runtime):
    g = runtime['grid']
    blocked = {tuple(c) for c in g['blocked']}
    free = lambda c: 0 <= c[0] < g['width'] and 0 <= c[1] < g['height'] and c not in blocked
    start = tuple(runtime['robot']['start'])
    seen, q = {start}, deque([start])
    while q:
        x, y = q.popleft()
        for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
            if n not in seen and free(n):
                seen.add(n)
                q.append(n)
    return sorted(seen)


def transform(case, pi):
    """Return a copy of the primary case with a seeded fraction of distractors made feasible."""
    import copy
    from language_cases import hidden_members
    from language_interface import universe
    c = copy.deepcopy(case)
    rt = c['public']['runtime']
    occupied = {tuple(o['cell']) for o in rt['objects']} | {tuple(t['cell']) for t in rt['targets']} | {tuple(rt['robot']['start'])}
    cells = [x for x in reachable_cells(rt) if x not in occupied]
    rng = random.Random(f"H:{pi}:{c['case_id']}")
    changed = []
    for o in sorted(rt['objects'], key=lambda o: o['id']):
        if not o['id'].startswith('d'):
            continue
        if rng.random() < pi and cells:
            cell = cells.pop(rng.randrange(len(cells)))
            o.update({'movable': True, 'mass': 1, 'footprint': [1, 1], 'orientation': 0, 'cell': list(cell)})
            changed.append(o['id'])
    c['semantic'] = hidden_members(c['public'], c['hidden_spec'])
    c['universe'] = universe(c['public'])
    c['variant'] = 'primary'
    c['experiment_H'] = {'pi': pi, 'feasible_distractors': changed}
    return c


def digest(v):
    return hashlib.sha256(json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()).hexdigest()


def job(args):
    out, split, index, pi = args
    from common import save
    from language_cases import make_cluster, physical_reference
    from language_interface import universe, expand
    from runtime_checks import check
    import physics_independent as phy
    out = Path(out)
    base = make_cluster('carrying', split, index)[0]
    c = transform(base, pi)
    pub, cid = c['public'], c['case_id']
    assert expand(pub, 'RULE', c['hidden_spec']) == c['semantic'], 'denotation mismatch'
    lang = out / 'language'
    save(lang / 'labels' / (cid + '.json.gz'), c)
    save(lang / 'public' / (cid + '.json'), pub)
    ref = physical_reference(pub)
    ref['public_sha256'] = digest(pub)
    save(lang / 'reference' / (cid + '.json.gz'), ref)
    mine, _ = phy.feasible(pub)
    if mine != set(ref['feasible']):
        raise AssertionError(f'{cid}: independent physics disagrees with reference')
    rows = []
    for g in universe(pub):
        v = check(pub, g)
        if v['feasible'] != (g in ref['feasible']):
            raise AssertionError(f'{cid} {g}: per-goal checker disagrees with reference')
        rows.append({'case_id': cid, 'goal': g, 'feasible': v['feasible'], 'work_units': v['work_units'],
                     'cost_unit': v['cost_unit'], 'public_sha256': digest(pub)})
    save(lang / 'ready' / (cid + '.json'), {'status': 'PASS', 'public_sha256': digest(pub), 'all_goal_queries': len(rows),
                                             'independent_reference_checked': True, 'independent_physics_checked': True})
    meta = {k: c[k] for k in ['case_id', 'cluster_id', 'domain', 'split', 'index', 'variant', 'style', 'base_type', 'distractors']}
    meta['feasible_distractors'] = len(c['experiment_H']['feasible_distractors'])
    return meta, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--pi', type=float, default=0.25)
    ap.add_argument('--workers', type=int, default=max(1, min(10, (os.cpu_count() or 2) - 1)))
    ap.add_argument('--limit-test', type=int, help='smoke tests only')
    a = ap.parse_args()
    out = Path(a.out).resolve()
    (out / 'language').mkdir(parents=True, exist_ok=False)
    (out / 'checks').mkdir()
    tasks = [(str(out), 'calibration', i, a.pi) for i in range(128)] + \
            [(str(out), 'test', i, a.pi) for i in range(a.limit_test if a.limit_test is not None else 512)]
    entries, nrows = [], 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex, gzip.open(out / 'checks' / 'checks_0_of1.jsonl.gz', 'wt') as f:
        for meta, rows in ex.map(job, tasks, chunksize=4):
            entries.append(meta)
            for r in rows:
                f.write(json.dumps(r, sort_keys=True) + '\n')
            nrows += len(rows)
    manifest = {'entries': entries, 'cases': len(entries),
                'protocol': {'experiment': 'H', 'pi': a.pi, 'base_generator': 'frozen language_cases.make_cluster (carrying, primary)',
                             'transform': 'seeded fraction pi of distractor objects -> movable, mass 1, 1x1 footprint, reachable free cell',
                             'created_at': datetime.now(timezone.utc).isoformat()}}
    (out / 'language' / 'case_manifest.json').write_text(json.dumps(manifest, indent=1))
    print(json.dumps({'cases': len(entries), 'goal_checks': nrows,
                      'mean_feasible_distractors': sum(e['feasible_distractors'] for e in entries) / len(entries)}, indent=1))


if __name__ == '__main__':
    main()
