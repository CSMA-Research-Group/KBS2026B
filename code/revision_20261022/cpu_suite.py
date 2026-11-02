"""Fresh layouts, strong existing baselines, and audited cost-to-decision curves."""
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback
from common import HERE, ROOT, now, digest, read, save, save_rows, table, interval
from operations import freeze, snapshot
from experiment_resources import plan_resources
from key_validation.cases import validate_public, member
from key_validation.engine import Policy, METHODS, COST_KEYS, BudgetExceeded
from key_validation.proofs import reference, audit_reference
from key_validation.audit import audit_decision
from e3_cpu_correctness.core import SCHEMA
from innovation_stage3.storage import stable

SETTINGS={'quantum_expansions':32,'initial_witness_expansions':64,'near_goal_distance':8}
WORK_KEYS=[k for k in COST_KEYS if k!='geometry_cache_hits']
PROTOCOL={'schema':'ccfa-fresh-layout-cost-v1','seed':2026102207,'calibration_layouts':64,
          'test_layouts':512,'candidate_counts':[16,64,256,1024],
          'methods':list(METHODS),'settings':SETTINGS,'cost_keys_sum':WORK_KEYS,
          'feasible_strata':[0,1,2],'obstructions':['shared','unshared'],
          'budget_quantiles':[.1,.25,.5,.75,.9],
          'scope':'replication and stress test of existing solvers, not new algorithm or independent timing claim',
          'primary':'cost-to-checked-decision CDF, exact decisions and all cost components; layout is pairing/cluster unit'}

def construct(split,index,count):
    seed=f"{PROTOCOL['seed']}:{split}:{index}"
    rng=random.Random(seed)
    lw,gap,rw,height=rng.randint(14,20),rng.randint(12,18),rng.randint(24,30),rng.randint(36,46)
    right=lw+gap+1; width=right+rw+1
    free={(x,y) for x in list(range(1,lw+1))+list(range(right,width-1)) for y in range(1,height-1)}
    bend1=rng.randint(lw+4,right-4);bend2=rng.randint(lw+4,right-4)
    y1=rng.randint(5,9);y2=rng.randint(height-10,height-6)
    for b,y,d in [(bend1,y1,4),(bend2,y2,-4)]:
        free.update((x,y) for x in range(lw,b+1))
        free.update((b,z) for z in range(min(y,y+d),max(y,y+d)+1))
        free.update((x,y+d) for x in range(b,right+1))
    obstruction=rng.choice(PROTOCOL['obstructions']); feasible=rng.choice(PROTOCOL['feasible_strata'])
    if obstruction=='shared':
        free.remove((lw+2,y1));free.remove((lw+2,y2))
    configs=[{'id':f'c{i}','footprint':fp,'start':[rng.randint(4,lw-4),rng.randint(4,height-5),rng.randrange(2)]}
             for i,fp in enumerate([[3,1],[1,3],[5,1],[1,5]])]
    world={'schema':SCHEMA,'width':width,'height':height,'rotation':'square_sweep','configurations':configs,
           'blocked':[[x,y] for y in range(height) for x in range(width) if (x,y) not in free]}
    negatives=count-feasible; fs=[]
    for i,c in enumerate(configs):
        n=negatives//4+(i<negatives%4)
        fs.append({'configuration_id':c['id'],'origin':[right+3,4],'columns':min(16,n),'count':n})
    positives=rng.sample(range(4),feasible)
    for i in positives:
        c=configs[i];fs.append({'configuration_id':c['id'],'origin':[c['start'][0]+2,c['start'][1]],'columns':1,'count':1})
    rng.shuffle(fs)
    for i,f in enumerate(fs): f['id']=f'f{i}'
    public={'world':world,'families':fs,'contract':'unique_loaded_goal'}
    validate_public(public)
    return {'case_id':f'{split}_{index:04d}_n{count}','cluster_id':f'{split}_{index:04d}',
            'split':split,'index':index,'candidate_count':count,'designed_feasible':feasible,
            'obstruction':obstruction,'world_sha256':digest(world),'public':public}

def work(result):return sum(result['cost'][k] for k in WORK_KEYS)

class WorkPolicy(Policy):
    def __init__(self,public,method,limit):
        super().__init__(public,method,None,SETTINGS);self.work_limit=limit
    def check_budget(self):
        if self.work_limit is not None and sum(self.cost[k] for k in WORK_KEYS)>self.work_limit:
            raise BudgetExceeded()

def init_worker(cpus):
    os.sched_setaffinity(0,set(cpus))
    try:os.nice(10)
    except OSError:pass

def layout_job(args):
    batch,split,index=args;batch=Path(batch);out=batch/'cpu'
    first=construct(split,index,PROTOCOL['candidate_counts'][0])
    ref_path=out/'reference'/f'{split}_{index:04d}.json.gz'
    if ref_path.exists():ref=read(ref_path)
    else:
        ref=reference(first['public']);audit_reference(first['public'],ref);save(ref_path,ref)
    entries=[]
    for count in PROTOCOL['candidate_counts']:
        c=construct(split,index,count);save(out/'cases'/(c['case_id']+'.json.gz'),c)
        if digest(c['public']['world'])!=first['world_sha256']:raise AssertionError('Scale changed physical world')
        reached={p['configuration_id']:{tuple(node[:2]) for node,_ in p['parents']} for p in ref['closures']}
        feasible=sorted(member(f,i)['goal_id'] for f in c['public']['families'] for i in range(f['count'])
                        if tuple(member(f,i)['target']) in reached[f['configuration_id']])
        current={**ref,'public_sha256':stable(c['public']),'feasible_goal_ids':feasible,
                 'expected_decision':'REJECT' if not feasible else 'EXECUTE' if len(feasible)==1 else 'CLARIFY'}
        if len(feasible)!=c['designed_feasible']:raise AssertionError('Construction failed; no replacement permitted')
        for method in METHODS:
            path=out/'raw'/(c['case_id']+'_'+method+'.json.gz')
            if path.exists():record=read(path)
            else:
                started=time.perf_counter()
                result=WorkPolicy(c['public'],method,None).run()
                audit=audit_decision(c['public'],result,current)
                record={'case_id':c['case_id'],'public_sha256':digest(c['public']),'method':method,
                        'result':result,'audit':audit,'wall_seconds_with_contention':time.perf_counter()-started,
                        'finished_at':now()}
                save(path,record)
            if record['public_sha256']!=digest(c['public']) or record['audit']['status']!='PASS':
                raise AssertionError('Saved input binding/audit failure')
            r=record['result'];entries.append({k:v for k,v in c.items() if k!='public'}|{
                'method':method,'expected_decision':current['expected_decision'],'decision':r['decision'],
                'n_feasible':len(feasible),'cost_proxy':work(r),**r['cost'],
                'cpu_seconds_throughput':r['policy_cpu_seconds'],'audit':record['audit']['status']})
    return entries

def preflight(out):
    checks=[]
    for i in range(6):
        c=construct('development',i,64);ref=reference(c['public']);audit_reference(c['public'],ref)
        assert len(ref['feasible_goal_ids'])==c['designed_feasible']
        for method in METHODS:
            result=WorkPolicy(c['public'],method,None).run();assert audit_decision(c['public'],result,ref)['status']=='PASS'
            cost=work(result)
            for budget in [0,cost//2,cost,cost+1]:
                short=WorkPolicy(c['public'],method,budget).run()
                predicted=result['decision'] if cost<=budget else 'ABSTAIN'
                assert short['decision']==predicted,(method,budget,cost,short['decision'],predicted)
            checks.append({'case':c['case_id'],'method':method,'full_cost':cost,'budget_prefix_equivalence':True})
    save(out/'development_checks.json',checks)

def analyze(batch,records):
    import numpy as np
    out=batch/'cpu'/('analysis_'+now().replace(':','').replace('-',''));out.mkdir(parents=True)
    table(out/'costs_and_decisions.csv',records)
    cal=[r for r in records if r['split']=='calibration'];test=[r for r in records if r['split']=='test']
    thresholds={str(n):sorted(set(int(np.quantile([r['cost_proxy'] for r in cal if r['candidate_count']==n],q,method='higher'))
                                      for q in PROTOCOL['budget_quantiles'])) for n in PROTOCOL['candidate_counts']}
    save(out/'calibrated_budgets.json',{'calibration_only':True,'thresholds':thresholds})
    metrics=[];confusion=[]
    for count in PROTOCOL['candidate_counts']:
        for method in METHODS:
            rs=[r for r in test if r['candidate_count']==count and r['method']==method]
            for budget in thresholds[str(count)]+[None]:
                decisions=[r['decision'] if budget is None or r['cost_proxy']<=budget else 'ABSTAIN' for r in rs]
                n=len(rs);xe=sum(d=='EXECUTE' for d in decisions);passed=sum(d!='ABSTAIN' for d in decisions)
                counts=Counter((r['expected_decision'],d) for r,d in zip(rs,decisions))
                metrics.append({'candidate_count':count,'method':method,'budget_proxy':budget,'layouts':n,
                    'correct_decisions':passed,'correct_executions':xe,'abstentions':n-passed,'wrong_decisions':0,
                    'median_cost_proxy':float(np.median([r['cost_proxy'] for r in rs])),
                    'error_upper95':interval(0,n)['upper95_one_sided'],
                    'curve_definition':'prefix-capped full deterministic trace; not fresh finite wall-time runs'})
                for e in ['EXECUTE','CLARIFY','REJECT']:
                    for d in ['EXECUTE','CLARIFY','REJECT','ABSTAIN']:
                        confusion.append({'candidate_count':count,'method':method,'budget_proxy':budget,'expected':e,'decision':d,'count':counts[e,d]})
    table(out/'budget_curves.csv',metrics);table(out/'confusion_matrices.csv',confusion)
    paired=[]
    for count in PROTOCOL['candidate_counts']:
        ix={(r['cluster_id'],r['method']):r for r in test if r['candidate_count']==count}
        clusters=sorted({k[0] for k in ix})
        for method in METHODS:
            ratios=[ix[c,method]['cost_proxy']/max(1,ix[c,'LAZY_SHARED_GBFS']['cost_proxy']) for c in clusters]
            paired.append({'candidate_count':count,'method':method,'paired_layouts':len(clusters),
                           'median_work_ratio_vs_lazy':float(np.median(ratios)),
                           'fraction_less_work':sum(v<1 for v in ratios)/len(ratios)})
    table(out/'paired_work_ratios.csv',paired)
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for ax,n in zip(axes.flat,PROTOCOL['candidate_counts']):
        for method in METHODS:
            vals=sorted(r['cost_proxy'] for r in test if r['candidate_count']==n and r['method']==method)
            ax.step(vals,np.arange(1,len(vals)+1)/len(vals),where='post',label=method)
        ax.set(title=f'{n} candidates; {PROTOCOL["test_layouts"]} layouts',xlabel='Audited operation-count proxy',ylabel='Correct decision fraction',xscale='log')
    handles,labels=axes.flat[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=2,fontsize=7)
    fig.tight_layout(rect=(0,.12,1,1));fig.savefig(out/'cost_to_decision.pdf');plt.close(fig)
    report=('# 新布局 CPU 方法比较\n\n'+f"{PROTOCOL['test_layouts']} 个测试物理布局，每布局配对四个候选规模与七个既有算法；不是 2048 个独立物理布局。校准布局另为 {PROTOCOL['calibration_layouts']}。全部原始证书和独立参考均保存并审计。\n\n"
        +'主图为得到完整决策所需操作计数的累积分布；有限预算点来自确定性搜索前缀截断，开发集实际截断运行核对等价。计数是明确声明的代理，逐项向量另附，不等同于 CPU 秒。并发运行计时只供运维，不能据此宣称独占加速。\n\n'
        +'这是既有方法在新随机布局上的比较；抽样仍局限于双房间/弯通道构造，不能视作实际部署或新算法证明。若决策调度未优于强 lazy 基线，保留结果并收缩贡献。\n')
    (out/'REPORT.md').write_text(report)
    (ROOT/'docs'/('CCFA_CPU_'+batch.name+'.md')).write_text(report+'\n结果：`'+str(out.relative_to(ROOT))+'`。\n')
    return out

def run(batch):
    batch=Path(batch);out=batch/'cpu';out.mkdir(parents=True,exist_ok=True)
    sources=list((ROOT/'code').rglob('*.py'))+[ROOT/'code/configs/experiment_resources.json',HERE/'cpu_suite.py',HERE/'common.py',HERE/'operations.py']
    freeze(batch,'cpu',PROTOCOL,sources)
    save(batch/'cpu_status.json',{'status':'RUNNING','pid':os.getpid(),'at':now()},mutable=True)
    resources=plan_resources(16);save(out/('resources_'+str(os.getpid())+'.json'),resources)
    try:
        if not (out/'development_checks.json').exists():preflight(out)
        all_records=[]
        with ProcessPoolExecutor(max_workers=resources['workers'],initializer=init_worker,initargs=(resources['worker_cpus'],)) as pool:
            for split,n in [('calibration',PROTOCOL['calibration_layouts']),('test',PROTOCOL['test_layouts'])]:
                futures=[pool.submit(layout_job,(str(batch),split,i)) for i in range(n)]
                for index,f in enumerate(as_completed(futures)):
                    all_records.extend(f.result())
                    if index%8==0:
                        save(batch/'cpu_status.json',{'status':'RUNNING','pid':os.getpid(),'split':split,'completed_layouts_in_split':index+1,'total_layouts_in_split':n,'at':now()},mutable=True)
                        print('CPU',split,index+1,'/',n,flush=True)
        out_analysis=analyze(batch,all_records)
        save(batch/'cpu_status.json',{'status':'COMPLETE','output':str(out_analysis),'at':now(),'records':len(all_records)},mutable=True)
    except BaseException:
        save(batch/'cpu_status.json',{'status':'FAILED','at':now(),'traceback':traceback.format_exc()},mutable=True);raise

if __name__=='__main__':run(Path(sys.argv[1]).resolve())
