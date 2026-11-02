#!/usr/bin/env python3
"""Check saved inputs and reconstruct the empirical tables used in the KBS draft."""
from collections import Counter, defaultdict, deque
from fractions import Fraction
import csv
import gzip
import json
from pathlib import Path
import re
import sys

from workspace import ROOT, EVIDENCE, read_json, sha256, write_json
sys.path[:0] = [str(ROOT / 'code/audit'), str(ROOT / 'code/revision_20261022'), str(ROOT / 'code')]
import ra_core as rc
import ra_calibration as calibration
import cpu_reporting
from language_cases import hidden_members
from language_interface import expand, universe


def require(condition, message):
    if not condition:
        raise ValueError(message)


def jsonlines(path):
    with gzip.open(path, 'rt') as stream:
        return [json.loads(line) for line in stream]


def tex_rows(name):
    rows = []
    for line in (ROOT / 'latex/tables' / name).read_text().splitlines():
        if '&' in line and line.rstrip().endswith('\\\\'):
            rows.append([part.strip() for part in line.rstrip()[:-2].split('&')])
    return rows


def s1(out):
    base = EVIDENCE / 's1'
    decisions = jsonlines(base / 'decisions.jsonl.gz')
    cases = {c['instance_id']: c for c in jsonlines(base / 'cases.jsonl.gz')}
    specs = jsonlines(base / 'requests_qwen_gemma.jsonl.gz') + jsonlines(base / 'requests_phi.jsonl.gz')
    specs = {rc.digest(s): s for s in specs}
    require(len(decisions) == 768 and len(cases) == 128 and len(specs) == 1152, 'S1 scope')
    used = {request for d in decisions for request in d['request_ids']}
    require(used == set(specs), 'S1 request-plan coverage')
    for rid, spec in specs.items():
        record = read_json(base / 'requests' / (rid + '.json.gz'))
        intent = read_json(base / 'requests' / (rid + '.intent.json.gz'))
        require(record['spec'] == spec == intent['spec'], 'S1 spec binding: ' + rid)
        require(record['task_id'] == intent['task_id'] == rid, 'S1 request identity')
        for value in (record, intent):
            if 'record_sha256' in value:
                require(rc.digest({k: v for k, v in value.items() if k != 'record_sha256'}) == value['record_sha256'], 'S1 sealed record')
        require(spec['runtime_sha256'] == rc.digest(cases[spec['instance_id']]['runtime']), 'S1 runtime binding')
    seen = set()
    summary = defaultdict(list)
    matrix = Counter()
    for d in decisions:
        cid, model, arm = d['instance_id'], d['model'], d['arm']
        key = (cid, model, arm)
        require(key not in seen and d['split'] == 'test' and d['budget'] == 4096 and d['scorable'], 'S1 duplicate or scope')
        seen.add(key)
        ref = read_json(base / 'reference' / (cid + '.json.gz'))
        payload = ref.get('payload', ref)
        feasible = payload['reference']['feasible_goals']
        expected = 'REJECT' if not feasible else 'EXECUTE' if len(feasible) == 1 else 'CLARIFY'
        require(d['n_feasible'] == len(feasible) and d['expected_decision'] == expected, 'S1 reference mismatch')
        execution = d['decision'] == 'EXECUTE'
        correct_execution = execution and len(feasible) == 1 and d['selected_goal'] in feasible
        require(d['correct_execution'] == correct_execution, 'S1 selected-goal correctness')
        require((d['execution_error'] is True) == (execution and not correct_execution), 'S1 execution error')
        policy = 'Audited' if arm == 'UNION3_AUDITED' else 'Union'
        summary[(model.capitalize(), policy)].append(d)
        matrix[(model, policy, expected, d['decision'])] += 1
    rows = []
    for (model, policy), ds in sorted(summary.items()):
        rows.append([model, policy, str(len(ds)), str(sum(d['correct_execution'] for d in ds)),
                     str(sum(d['execution_error'] is True for d in ds)),
                     str(sum(d['decision'] == 'REJECT' and d['n_feasible'] > 0 for d in ds)),
                     str(sum(d['decision'] == 'ABSTAIN' for d in ds))])
    require(rows == tex_rows('archive_summary.tex')[1:], 'S1 summary differs from manuscript')
    for model in ['qwen', 'gemma', 'phi']:
        rows = [[policy, ref.title()] + [str(matrix[model, policy, ref, d]) for d in rc.DECISIONS]
                for policy in ['Union', 'Audited'] for ref in rc.REFERENCE_DECISIONS]
        require(rows == tex_rows('matrix_' + model + '.tex')[1:], 'S1 matrix differs: ' + model)
    return {'decisions': len(decisions), 'worlds': len(cases), 'requests': len(specs), 'tables_matched': 4,
            'evidence': 'saved decision replay and request/reference binding; no new model or solver execution'}


def cpu(out):
    base = EVIDENCE / 'main/cpu'
    sources = list(base.glob('analysis_*/costs_and_decisions.csv'))
    require(len(sources) == 1, 'CPU analysis selection')
    with sources[0].open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    cpu_reporting.validate_rows(rows)
    cases = {p.name[:-8]: read_json(p) for p in (base / 'cases').glob('*.json.gz')}
    references = {p.name[:-8]: read_json(p) for p in (base / 'reference').glob('*.json.gz')}
    require(len(cases) == 2304 and len(references) == 576, 'CPU case/reference counts')
    from key_validation.cases import member
    for cid, case in cases.items():
        public = case['public']
        require(rc.digest(public['world']) == case['world_sha256'], 'CPU world binding')
        ref = references[case['cluster_id']]
        first = cases[case['cluster_id'] + '_n16']
        require(ref['public_sha256'] == rc.digest(first['public']), 'CPU reference binding')
        reached = {p['configuration_id']: {tuple(node[:2]) for node, _ in p['parents']} for p in ref['closures']}
        feasible = [member(f, i)['goal_id'] for f in public['families'] for i in range(f['count'])
                    if tuple(member(f, i)['target']) in reached[f['configuration_id']]]
        require(len(feasible) == case['designed_feasible'], 'CPU saved reference membership')
    for row in rows:
        case = cases[row['case_id']]
        raw = read_json(base / 'raw' / (row['case_id'] + '_' + row['method'] + '.json.gz'))
        require(raw['public_sha256'] == rc.digest(case['public']), 'CPU raw input binding')
        require(raw['audit']['status'] == 'PASS' and all(c['pass'] for c in raw['audit']['checks']), 'CPU audit failure')
        require(raw['result']['decision'] == row['decision'], 'CPU saved decision')
        require(all(int(row[k]) == v for k, v in raw['result']['cost'].items()), 'CPU counter export mismatch')
        require(all(str(case[k]) == row[k] for k in ['case_id','cluster_id','split','candidate_count','world_sha256','obstruction','designed_feasible']), 'CPU row descriptors')
        require(int(row['n_feasible']) == case['designed_feasible'], 'CPU feasible count')
    export = out / 'cpu_tables'
    cpu_reporting.run(sources[0], export)
    for name in ['cpu_ratios.tex', 'cpu_strata.tex']:
        def cells(text):
            return [[part.strip() for part in line.rstrip()[:-2].split('&')]
                    for line in text.splitlines() if '&' in line and line.rstrip().endswith('\\\\')]
        require(cells((export / name).read_text()) == tex_rows(name), 'CPU manuscript table mismatch: ' + name)
    points = read_json(ROOT / 'data/illustrations/cpu_plot_points.json')
    for item in points:
        values = sorted(int(r['cost_proxy']) for r in rows if r['split'] == 'test'
                        and int(r['candidate_count']) == item['candidate_count'] and r['method'] == item['method'])
        require(item['x'] == values and item['y'] == [(i + 1) / len(values) for i in range(len(values))], 'CPU plotting points')
    receipt = read_json(ROOT / 'data/illustrations/receipt.json')
    require(receipt['source_sha256'] == sha256(sources[0]), 'CPU figure source CSV')
    for name, digest in receipt['figures'].items():
        require(sha256(ROOT / 'latex/figures' / name) == digest, 'Current figure binding: ' + name)
    return {'raw_records': len(rows), 'independent_test_layouts': 512, 'scales': [16,64,256,1024],
            'tables_matched': 2, 'figure': '14336 plot points reconstructed from counters; current PDFs match their original export receipt',
            'evidence': 'raw counter/reference replay; no new solver execution or exclusive timing'}


def language_inputs(base, checks, is_h=False):
    entries = read_json(base / 'language/case_manifest.json')['entries']
    all_public = {}
    for entry in entries:
        cid = entry['case_id']
        lang = base / 'language'
        public = read_json(lang / 'public' / (cid + '.json'))
        lab = read_json(lang / 'labels' / (cid + '.json.gz'))
        ref = read_json(lang / 'reference' / (cid + '.json.gz'))
        ready = read_json(lang / 'ready' / (cid + '.json'))
        digest = rc.digest(public)
        require(public == lab['public'] and ref['public_sha256'] == ready['public_sha256'] == digest, 'Language input binding: ' + cid)
        require(lab['universe'] == universe(public), 'Language universe: ' + cid)
        require(lab['semantic'] == hidden_members(public, lab['hidden_spec']), 'Hidden label recomputation: ' + cid)
        require(ready['status'] == 'PASS' and ready['independent_reference_checked'], 'Ready status: ' + cid)
        by_goal = checks[cid]
        require(set(by_goal) == set(lab['universe']) and ready['all_goal_queries'] == len(by_goal), 'Goal check coverage: ' + cid)
        require(all(v[2] == digest for v in by_goal.values()), 'Goal check public binding: ' + cid)
        require(set(ref['feasible']) == {g for g, v in by_goal.items() if v[0]}, 'Goal/reference agreement: ' + cid)
        if is_h:
            require(lab['experiment_H']['pi'] == .25 and ready['independent_physics_checked'], 'H intervention scope')
        all_public[cid] = public
    return entries, all_public


def language(out):
    batch = rc.Batch(EVIDENCE / 'main')
    checks = batch.checks()
    entries, publics = language_inputs(batch.path, checks)
    require(len(entries) == 1872, 'Language case count')
    coverage, thresholds, models = [], [], {}
    for model in rc.MODELS:
        responses = batch.responses(model)
        rdir = batch.lang / model / 'responses'
        files = list(rdir.glob('*.json.gz'))
        require(len(files) == len(responses), 'Orphan/duplicate language responses: ' + model)
        invalid, truncated = 0, 0
        for key, response in responses.items():
            cid, rep, suite, draw, temp = key
            public = publics[cid]
            require(response['public_sha256'] == rc.digest(public), 'Response public binding')
            req = response['request']
            dispatch = read_json(batch.lang / model / 'dispatched' / (req['request_id'] + '.json'))
            require(dispatch['request'] == req and dispatch['public_sha256'] == response['public_sha256'], 'Dispatch binding')
            if response['valid']:
                require(json.loads(response['raw_output']) == response['parsed'], 'Parsed output mismatch')
                require(expand(public, rep, response['parsed']) == response['candidates'], 'Candidate expansion mismatch')
                require(response['error'] is None and not response['truncated'], 'Invalid response marked valid')
            invalid += not response['valid']
            truncated += bool(response['truncated'])
            require(response['output_tokens'] == len(response['output_token_ids']), 'Token count mismatch')
            require(re.fullmatch('[0-9a-f]{64}', response['prompt_sha256']) is not None, 'Malformed prompt hash')
        saved = read_json(batch.lang / ('calibration_' + model + '.json'))
        for domain in rc.DOMAINS:
            for rep in rc.REPS:
                recs = [calibration.world_record(batch, responses, e['case_id'], (rep,))
                        for e in batch.cases('calibration', domain, 'primary')]
                require(len(recs) == 128 and all(r is not None for r in recs), 'Incomplete calibration')
                qs = [calibration.conformal_threshold(recs, kind, Fraction(1,20)) for kind in ['semantic','decision']]
                thresholds.append([{'qwen3_8b':'Qwen3-8B','gemma2_9b':'Gemma-2-9B'}[model], domain.title(), rep] + [f'{float(q):.3f}' for q in qs])
                for split in ['calibration','test']:
                    es = batch.cases(split, domain, 'primary')
                    complete = sum(calibration.world_record(batch, responses, e['case_id'], (rep,)) is not None for e in es)
                    coverage.append(dict(model=model, domain=domain, representation=rep, split=split, required_worlds=len(es), complete_worlds=complete))
        # Exact response hashes bound to each saved calibration draw are checked separately from rounded tables.
        for key, digest in saved['response_hashes'].items():
            cid, rep, n = key.rsplit('_', 2)
            candidates = [responses[(cid, rep, 'primary', draw, .7)] for draw in range(int(n))]
            # The frozen pipeline hashes the list prefix used for this k, including invalid draws.
            require(rc.digest(candidates) == digest, 'Saved calibration response binding: ' + key)
        models[model] = {'responses': len(files), 'invalid': invalid, 'truncated': truncated,
                         'status_file_is_authoritative': False}
    require(thresholds == tex_rows('calibration.tex')[1:], 'Calibration manuscript table mismatch')
    write_json(out / 'language_coverage.json', coverage)
    write_json(out / 'calibration_recomputed.json', thresholds)
    return {'worlds': len(entries), 'goal_checks': sum(map(len, checks.values())), 'models': models,
            'calibration_cells': 8, 'tables_matched': 1, 'heldout_complete': all(r['complete_worlds'] == r['required_worlds'] for r in coverage),
            'prompt_hash_limit': 'syntax retained; tokenizer-rendered prompt hashes require the frozen tokenizer and were not reconstructed',
            'evidence': 'saved input/response/checker replay and exact-rational calibration, not new inference'}


def experiment_h(out):
    base = EVIDENCE / 'experiment_h'
    checks = defaultdict(dict)
    for path in (base / 'checks').glob('*.jsonl.gz'):
        for value in jsonlines(path):
            cid, goal = value['case_id'], value['goal']
            require(goal not in checks[cid], 'Duplicate H goal check')
            checks[cid][goal] = (value['feasible'], value['work_units'], value['public_sha256'])
    entries, _ = language_inputs(base, checks, True)
    require(Counter(e['split'] for e in entries) == {'calibration':128, 'test':512}, 'H split size')
    require(not list(base.rglob('responses')), 'Unexpected H responses; update the evidence assessment')
    return {'prepared_worlds': len(entries), 'goal_checks': sum(map(len, checks.values())), 'pi': .25,
            'model_responses': 0, 'evidence': 'prepared inputs only; H outcome pending'}


def cost_diagnostics(out):
    results = {}
    for name in ['E5_shared_certificates','E7_dead_end_savings']:
        base = ROOT / 'data/verification_cost' / name
        summary = read_json(base / 'summary.json')
        with (base / 'worlds.csv').open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        require(len(rows) == summary['rows'] == 3464, 'Verification diagnostic row count')
        require(len({(r['model'],r['domain'],r['representation'],r['case_id'],r['policy']) for r in rows}) == len(rows), 'Duplicate cost comparison')
        if name.startswith('E5'):
            require(all(r['sh_equal'] == r['ind_equal'] == 'True' and r['pg_decision'] == r['sh_decision'] for r in rows), 'E5 consistency')
        else:
            require(summary['decisions_identical'] is True, 'E7 saved decision assertion')
            require(all(int(r['queries_dead_end']) <= int(r['queries_paper']) and int(r['work_dead_end']) <= int(r['work_paper']) for r in rows), 'E7 cost monotonicity')
        results[name] = {'rows':len(rows), 'evidence':'saved diagnostic output consistency; not a complete held-out result or new execution'}
    return results


def schematic(out):
    scene = read_json(ROOT / 'data/illustrations/running_scene.json')
    w, h = scene['grid']; target = tuple(scene['target'])
    def reachable(source, opening):
        blocked = {(scene['wall_x'], y) for y in range(h) if not (opening and y == scene['optional_opening_y'])}
        seen = {tuple(source)}; queue = deque(seen)
        while queue:
            x, y = queue.popleft()
            for p in [(x-1,y),(x+1,y),(x,y-1),(x,y+1)]:
                if 0 <= p[0] < w and 0 <= p[1] < h and p not in blocked and p not in seen:
                    seen.add(p); queue.append(p)
        return target in seen
    require([reachable(scene[k], opening) for opening in [False,True] for k in ['a','b']] == [True,False,True,True], 'Running example reachability')
    return {'reachability':'PASS', 'evidence':'point-grid schematic only'}


def run(output):
    results = {}
    for name, function in [('s1',s1),('cpu',cpu),('language',language),('experiment_h',experiment_h),('verification_cost',cost_diagnostics),('schematic',schematic)]:
        print('Checking ' + name, flush=True)
        results[name] = function(output)
        write_json(output / 'data_checks.json', results)
    return results
