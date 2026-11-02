#!/usr/bin/env python3
"""Replay the two saved verification-cost diagnostics on the packaged language snapshot."""
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
import csv
from fractions import Fraction as F
import gzip
import json
import multiprocessing
import os
from pathlib import Path
import sys
from workspace import ROOT, EVIDENCE, io_resources, new_run, now, sha256, write_json
sys.path[:0] = [str(ROOT / 'code/audit'), str(ROOT / 'code/revision_20261022'), str(ROOT / 'code')]
import ra_core as rc
import ra_calibration as cal
import ra_lazy as rl
import e5_shared_certificates as e5


def write_csv(path, rows):
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def compare(saved, rows):
    with saved.open(newline='') as stream:
        original = list(csv.DictReader(stream))
    key = lambda row: tuple(str(row[k]) for k in ['model','domain','representation','case_id','policy'])
    got = {key(r): {k:str(v) for k,v in r.items()} for r in rows}
    want = {key(r):r for r in original}
    if got != want:
        mismatches = [k for k in set(got) | set(want) if got.get(k) != want.get(k)]
        raise ValueError(f'{saved.name}: {len(mismatches)} differing records; first: {mismatches[:3]}')


def main():
    out = new_run('cost_replays')
    plan = io_resources()
    write_json(out / 'resources.json', plan)
    state = {'status':'RUNNING', 'started_at':now(), 'new_model_requests':0,
             'scope':'replay of saved partial-snapshot diagnostics; no exclusive timing claim',
             'directory':str(out.relative_to(ROOT)), 'source_hashes':{str(p.relative_to(ROOT)):sha256(p) for p in (ROOT/'code/audit').glob('*.py')}}
    write_json(out / 'status.json', state)
    print(out, flush=True)
    try:
        batch = rc.Batch(EVIDENCE / 'main')
        checks = batch.checks()
        index = out / 'checks'; index.mkdir()
        with gzip.open(index / 'checks.jsonl.gz', 'wt') as stream:
            for cid, values in sorted(checks.items()):
                for goal, (feasible, work, digest) in sorted(values.items()):
                    stream.write(json.dumps(dict(case_id=cid,goal=goal,feasible=feasible,work_units=work,public_sha256=digest))+'\n')
        with (ROOT/'data/verification_cost/calibration_thresholds.csv').open(newline='') as stream:
            thresholds = list(csv.DictReader(stream))
        if len(thresholds) != 8:
            raise ValueError('Expected eight main calibration cells')
        responses = {model:batch.responses(model) for model in rc.MODELS}
        for row in thresholds:
            model,domain,rep = (row[k] for k in ['model','domain','representation'])
            records = [cal.world_record(batch,responses[model],e['case_id'],(rep,)) for e in batch.cases('calibration',domain,'primary')]
            if len(records) != 128 or any(r is None for r in records):
                raise ValueError('Incomplete calibration')
            q_sem = cal.conformal_threshold(records,'semantic',F(1,20))
            q_dec = cal.conformal_threshold(records,'decision',F(1,20))
            q_crc,_ = cal.crc_select(records,F(1,20),cal.q_grid(3),cal.env_q)
            if [q_sem,q_dec,q_crc] != [F(row[k]) for k in ['q_semantic','q_decision','q_crc']]:
                raise ValueError('Cost calibration mismatch')
        print('Calibration and compact check index PASS', flush=True)
        tasks = [(str(index),model,domain,rep,None,thresholds) for model in rc.MODELS for domain in rc.DOMAINS for rep in rc.REPS]
        # Only CPU search and saved responses are used. Fork retains the optional file-access guard.
        context = multiprocessing.get_context('fork') if 'fork' in multiprocessing.get_all_start_methods() else multiprocessing.get_context()
        with ProcessPoolExecutor(max_workers=plan['workers'], mp_context=context) as pool:
            rows = [row for cell in pool.map(e5.cell_rows,tasks) for row in cell]
        write_csv(out/'E5_worlds.csv',rows)
        compare(ROOT/'data/verification_cost/E5_shared_certificates/worlds.csv',rows)
        state['E5'] = {'rows':len(rows),'all_saved_fields_equal':True}
        write_json(out/'status.json',state)
        print('E5 exact record replay PASS', flush=True)
        rows = []
        for row in thresholds:
            model,domain,rep = (row[k] for k in ['model','domain','representation'])
            for entry in batch.cases('test',domain,'primary'):
                cid = entry['case_id'];rec = cal.world_record(batch,responses[model],cid,(rep,))
                if rec is None or not rec['valid_any']:
                    continue
                values = checks[cid]
                for policy,(lower,upper) in [('UNION',(rec['union'],rec['union'])),
                        ('SEMANTIC_CALIBRATED',cal.env_q(rec,F(row['q_semantic']))),
                        ('DECISION_CALIBRATED',cal.env_q(rec,F(row['q_decision']))),
                        ('DECISION_RISK_CRC',cal.env_q(rec,F(row['q_crc'])))]:
                    checker = lambda goal:values[goal][:2]
                    ordinary = rc.lazy(lower,upper,checker)
                    stopped = rl.lazy_dead_end(lower,upper,checker)
                    if (ordinary['decision'],ordinary['selected']) != (stopped['decision'],stopped['selected']):
                        raise ValueError('Stopping rule changed a decision')
                    rows.append(dict(model=model,domain=domain,representation=rep,policy=policy,case_id=cid,
                        decision=ordinary['decision'],queries_paper=ordinary['queries'],queries_dead_end=stopped['queries'],
                        work_paper=ordinary['work'],work_dead_end=stopped['work'],dead_end_stop=stopped['dead_end_stop']))
        write_csv(out/'E7_worlds.csv',rows)
        compare(ROOT/'data/verification_cost/E7_dead_end_savings/worlds.csv',rows)
        state['E7'] = {'rows':len(rows),'all_saved_fields_equal':True,'decisions_equal':True}
        state.update(status='PASS',finished_at=now())
        (out/'REPORT.md').write_text('# 核验成本复核\n\nE5、E7 各 3,464 条记录的全部字段与保存输出一致；校准只使用既定 128 个校准世界。\n\n本次复用原语言响应，重新运行 CPU 核验与停止逻辑，没有模型推理。数据仍是未完成的 held-out 快照；一致性不把诊断升级为完整实验结果。并行耗时不用于性能比较。\n')
    except BaseException as error:
        state.update(status='FAIL',finished_at=now(),error=repr(error))
        raise
    finally:
        write_json(out/'status.json',state)
        write_json(ROOT/'work/latest_cost_replay.json',state)
        print(json.dumps({'status':state['status'],'directory':str(out.relative_to(ROOT))}),flush=True)


if __name__ == '__main__':
    main()
