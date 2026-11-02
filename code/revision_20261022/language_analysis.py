"""Calibrate on calibration primaries; evaluate frozen decision rules on held-out worlds."""
from collections import Counter,defaultdict
import math
from pathlib import Path
import sys
from common import ROOT,now,read,save,save_rows,table,interval,digest
from language_interface import universe
from runtime_checks import decide

ANALYSIS_PROTOCOL={'schema':'decision-relevant-envelope-v1','alphas':[.01,.05,.1,.2],
    'main_alpha':.05,'k_primary':[1,2,3],'query_budgets':[0,1,2,4,8,16,32,64,128,None],
    'calibration':'128 independent primary worlds per domain/model/representation/k, max nonconformity per world',
    'semantic_score':'max(1-p(g) for g in S, p(g) for g in G minus S)',
    'decision_score':'same max restricted to physically feasible G; empty maximum is zero',
    'quantile':'ceil((n+1)*(1-alpha)) order statistic, infinity if rank > n; conservatively clamp q >= 0.5',
    'envelope':'L={g:p(g)>q}; U={g:p(g)>=1-q}; decision rules only inspect queried physical states',
    'invalid':'invalid draws contribute an empty proposal to scores; if every draw invalid, model policies ABSTAIN',
    'paired_primary':'decision calibrated versus semantic calibrated, alpha=.05,k=3, full query budget; domains/models/representations kept separate',
    'multiplicity':'Holm across 32 prespecified tests: 2 models x 2 domains x 2 representations x 4 endpoints',
    'scope':'marginal per-world error bound under exchangeability, not error conditional on execution; synthetic language only'}

def response_index(batch,model,required=False):
    jobs=read(batch/'language'/('jobs_'+model+'.json'))['jobs'];ix={}
    for j in jobs:
        p=batch/'language'/model/'responses'/(j['request_id']+'.json.gz')
        if p.exists():
            r=read(p)
            if r['request']!=j:raise ValueError('Request binding changed')
            ix[(j['case_id'],j['representation'],j['suite'],j['draw'],j['temperature'])]=r
        elif required:raise ValueError('Missing terminal request '+j['request_id'])
    return ix

def draws(ix,cid,rep,k=3,suite='primary',temp=.7):
    return [ix[cid,rep,('sample_sweep' if suite=='sweep' and i>=3 else 'primary' if suite=='sweep' else suite),i,temp] for i in range(k)]

def votes(gs,rs):
    count=Counter(g for r in rs if r['valid'] for g in set(r['candidates']))
    return {g:count[g]/len(rs) for g in gs}

def nonconformity(p,semantic,scope):
    semantic=set(semantic)
    return max([1-p[g] if g in semantic else p[g] for g in scope]+[0.])

def quantile(scores,alpha):
    rank=math.ceil((len(scores)+1)*(1-alpha))
    return max(.5,sorted(scores)[rank-1]) if rank<=len(scores) else 1. # scores <=1; q=1 is the full safe envelope

def calibrate(batch,model):
    path=batch/'language'/('calibration_'+model+'.json')
    if path.exists():return read(path)
    ix=response_index(batch,model);entries=read(batch/'language/case_manifest.json')['entries'];out=[];inputs={}
    for domain in ['carrying','workflow']:
        cases=[c for c in entries if c['split']=='calibration' and c['variant']=='primary' and c['domain']==domain]
        if len(cases)!=128:raise ValueError('Calibration size changed')
        for rep in ['LIST','RULE']:
            for k in [1,2,3]:
                scores={'semantic':[],'decision':[]}
                for meta in cases:
                    cid=meta['case_id'];c=read(batch/'language/labels'/(cid+'.json.gz'))
                    ref=read(batch/'language/reference'/(cid+'.json.gz'))
                    if not (batch/'language/ready'/(cid+'.json')).exists():raise ValueError('Unchecked calibration reference')
                    rs=draws(ix,cid,rep,k);p=votes(c['universe'],rs)
                    inputs[cid+'_'+rep+'_'+str(k)]=digest(rs)
                    scores['semantic'].append(nonconformity(p,c['semantic'],c['universe']))
                    scores['decision'].append(nonconformity(p,c['semantic'],ref['feasible']))
                for kind,values in scores.items():
                    for alpha in ANALYSIS_PROTOCOL['alphas']:
                        out.append({'domain':domain,'representation':rep,'k':k,'kind':kind,'alpha':alpha,
                                    'n':len(values),'q':quantile(values,alpha),'scores':values})
    result={'model':model,'at':now(),'protocol':ANALYSIS_PROTOCOL,'thresholds':out,'response_hashes':inputs,
            'uses_test_outcomes':False}
    save(path,result);return result

def score(view,feasible,semantic):
    feasible=set(feasible);d=view['decision'];contract=view['contract']
    expected=('EXECUTE' if feasible else 'REJECT') if contract=='any_valid' else ('REJECT' if not feasible else 'EXECUTE' if len(feasible)==1 else 'CLARIFY')
    correct_exec=d=='EXECUTE' and view['selected_goal'] in feasible and (contract=='any_valid' or len(feasible)==1)
    return {'expected_decision':expected,'n_feasible':len(feasible),'correct_execution':correct_exec,
        'execution_error':d=='EXECUTE' and not correct_exec,'false_infeasibility':d=='REJECT' and bool(feasible),
        'wrong_decision':d!='ABSTAIN' and (d!=expected or (d=='EXECUTE' and not correct_exec)),
        'unsupported_clarification':d=='CLARIFY' and len(set(view['checked_positive_goals'])&set(semantic))<2,
        'correct_decision':d==expected and (d!='EXECUTE' or correct_exec)}

def case_decisions(batch,c,model,rep,k,rs,cal):
    cid=c['case_id'];gs=c['universe'];ref=read(batch/'language/reference'/(cid+'.json.gz'))
    physical=set(ref['feasible']);semantic=set(c['semantic']);feasible=physical&semantic;p=votes(gs,rs)
    union={g for r in rs if r['valid'] for g in r['candidates']};valid=any(r['valid'] for r in rs)
    checks={}
    def query(g):
        if g not in checks:
            checks[g]=read(batch/'language/checks'/cid/(g.replace('->','_to_')+'.json.gz'))
            assert checks[g]['goal']==g and checks[g]['public_sha256']==digest(c['public'])
        return checks[g]
    policies=[('UNION',union,union,'lower_first',None,False),
              ('UNION_EAGER',union,union,'eager',None,False),
              ('AUDIT_ORACLE',union,union,'lower_first',None,True),
              ('CATALOG_ORACLE',semantic,semantic,'lower_first',None,False)]
    for kind in ['semantic','decision']:
        for alpha in ANALYSIS_PROTOCOL['alphas']:
            q=next(x['q'] for x in cal['thresholds'] if x['domain']==c['domain'] and x['representation']==rep and x['k']==k and x['kind']==kind and x['alpha']==alpha)
            lo={g for g in gs if p[g]>q+1e-12};up={g for g in gs if p[g]>=1-q-1e-12}
            policies.append((kind.upper()+'_CALIBRATED',lo,up,'lower_first',alpha,False))
            if alpha==.05 and k==3:
                policies.append((kind.upper()+'_CALIBRATED_LEX',lo,up,'lexical',alpha,False))
                policies.append((kind.upper()+'_CALIBRATED_EAGER',lo,up,'eager',alpha,False))
    if k==3:
        for q in [.5,2/3,1.]:
            policies.append(('AGREEMENT_HEURISTIC', {g for g in gs if p[g]>q+1e-12},{g for g in gs if p[g]>=1-q-1e-12},'lower_first',q,False))
    for contract in ['referential','any_valid']:
        for name,lo,up,order,alpha,audit in policies:
            # Full budget is evaluated for all k; the main k=3 comparison has a complete query-budget curve.
            budgets=ANALYSIS_PROTOCOL['query_budgets'] if k==3 else [None]
            for budget in budgets:
                view=decide(lo,up,query,contract,order,budget)
                if (not valid and name!='CATALOG_ORACLE') or (audit and union!=semantic):
                    view={**view,'decision':'ABSTAIN','selected_goal':None}
                yield {'case_id':cid,'cluster_id':c['cluster_id'],'split':c['split'],'variant':c['variant'],
                    'domain':c['domain'],'style':c['style'],'distractors':c['distractors'],'model':model,'representation':rep,'k':k,
                    'policy':name,'alpha_or_threshold':alpha,'semantic_size':len(semantic),'universe_size':len(gs),
                    'semantic_envelope_covered':lo<=semantic<=up,
                    'decision_envelope_covered':(lo&physical)<=feasible<=(up&physical),
                    'all_draws_invalid':not valid,**view,**score(view,feasible,semantic)}

def power_plan():
    from scipy.stats import binom
    result=[]
    n=512
    for discordance in [.1,.2,.4]:
        for difference in [.025,.05,.1]:
            if difference>discordance:continue
            power=0.
            for m in range(1,n+1):
                mass=binom.pmf(m,n,discordance)
                if mass<1e-12:continue
                lo=int(binom.ppf(.025,m,.5))
                while lo>=0 and 2*binom.cdf(lo,m,.5)>.05:lo-=1
                while lo+1<=m and 2*binom.cdf(lo+1,m,.5)<=.05:lo+=1
                alternative=(1+difference/discordance)/2
                power+=mass*(binom.cdf(lo,m,alternative)+binom.sf(m-lo-1,m,alternative))
            result.append({'worlds':n,'discordance':discordance,'net_difference':difference,'exact_two_sided_McNemar_power':float(power),'alpha':.05,'multiplicity':'unadjusted planning scenario, not guaranteed achieved power'})
    return result

def analyze(batch,models):
    import numpy as np
    entries=read(batch/'language/case_manifest.json')['entries']
    output=batch/'language'/('analysis_'+now().replace(':','').replace('-',''));output.mkdir(parents=True)
    all_rows=[];sampling=[];inference=[];format_summary=[]
    for model in models:
        cal=calibrate(batch,model);ix=response_index(batch,model,required=True)
        counts=defaultdict(list)
        for r in ix.values():
            j=r['request'];counts[(j['split'],j['suite'],j['representation'],j['temperature'])].append(r)
            inference.append({'request_id':j['request_id'],'case_id':j['case_id'],'model':model,'split':j['split'],
                'suite':j['suite'],'representation':j['representation'],'valid':r['valid'],'error':r.get('error'),
                **{f:r.get(f) for f in ['input_tokens','output_tokens','reasoning_tokens','truncated','reasoning_forced_close','batch_size','batch_wall_seconds','allocated_batch_wall_seconds','peak_cuda_allocated_bytes']}})
        for key,vals in counts.items():
            format_summary.append(dict(zip(['split','suite','representation','temperature'],key))|{'model':model,
                'requests':len(vals),'invalid':sum(not r['valid'] for r in vals),'truncated':sum(bool(r.get('truncated')) for r in vals),
                'output_tokens':sum(r.get('output_tokens',0) for r in vals),'reasoning_tokens':sum(r.get('reasoning_tokens',0) for r in vals)})
        for meta in entries:
            if meta['split']!='test':continue
            cid=meta['case_id'];c=read(batch/'language/labels'/(cid+'.json.gz'))
            if not (batch/'language/ready'/(cid+'.json')).exists():raise ValueError('Unchecked test case')
            ref=read(batch/'language/reference'/(cid+'.json.gz'));semantic=set(c['semantic']);feasible=semantic&set(ref['feasible'])
            for rep in ['LIST','RULE']:
                for k in [1,2,3]:
                    rs=draws(ix,cid,rep,k)
                    all_rows.extend(case_decisions(batch,c,model,rep,k,rs,cal))
                # Extra conditions are coverage diagnostics, not calibrated using the primary thresholds.
                suites=[('primary',.7,range(1,4))]
                if c['variant']=='primary' and c['index']<32:
                    if rep=='LIST':suites.append(('sweep',.7,range(1,21)))
                    for t in [0.,1.]:suites.append(('temperature',t,range(1,4)))
                    if model=='qwen3_8b':suites.append(('thinking',.6,range(1,4)))
                for suite,temp,ks in suites:
                    for k in ks:
                        rs=draws(ix,cid,rep,k,suite,temp);sets=[set(r['candidates']) if r['valid'] else set() for r in rs];u=set.union(*sets)
                        missing=semantic-u;missing_feasible=feasible-u
                        sampling.append({'case_id':cid,'cluster_id':c['cluster_id'],'variant':c['variant'],'domain':c['domain'],
                            'model':model,'representation':rep,'suite':suite,'temperature':temp,'k':k,'style':c['style'],
                            'semantic_size':len(semantic),'universe_size':len(c['universe']),'missing_semantic':len(missing),
                            'missing_feasible':len(missing_feasible),'infeasible_only_omission':bool(missing) and not missing_feasible,
                            'exact':u==semantic,'semantic_recall':len(u&semantic)/len(semantic) if semantic else None,
                            'false_members':len(u-semantic),'systematic_omissions':sorted(g for g in semantic if all(g not in s for s in sets)),
                            'invalid_draws':sum(not r['valid'] for r in rs),'truncated_draws':sum(bool(r.get('truncated')) for r in rs)})
    save_rows(output/'decisions.jsonl.gz',all_rows);table(output/'sampling.csv',sampling);table(output/'inference_cost.csv',inference);table(output/'format_summary.csv',format_summary)
    group=defaultdict(list)
    keys=['domain','model','representation','k','policy','alpha_or_threshold','contract','query_budget','variant']
    for r in all_rows:group[tuple(r[k] for k in keys)].append(r)
    metrics=[];matrices=[];utilities=[]
    for key,rs in group.items():
        meta=dict(zip(keys,key));n=len(rs);ex=sum(r['decision']=='EXECUTE' for r in rs);wrong=sum(r['execution_error'] for r in rs)
        row=meta|{'worlds':n,'executions':ex,'wrong_executions':wrong,'correct_executions':sum(r['correct_execution'] for r in rs),
            'correct_decisions':sum(r['correct_decision'] for r in rs),'false_rejections':sum(r['false_infeasibility'] for r in rs),
            'wrong_decisions':sum(r['wrong_decision'] for r in rs),'clarifications':sum(r['decision']=='CLARIFY' for r in rs),
            'abstentions':sum(r['decision']=='ABSTAIN' for r in rs),'mean_goal_queries':sum(r['queries'] for r in rs)/n,
            'mean_checker_work':sum(r['work_units'] for r in rs)/n,'mean_semantic_set_size':sum(r['semantic_size'] for r in rs)/n,
            'semantic_envelope_coverage':sum(r['semantic_envelope_covered'] for r in rs)/n,'decision_envelope_coverage':sum(r['decision_envelope_covered'] for r in rs)/n,
            **{'wrong_per_world_'+k:v for k,v in interval(wrong,n).items()},**{'wrong_per_execution_'+k:v for k,v in interval(wrong,ex).items()}}
        metrics.append(row);counts=Counter((r['expected_decision'],r['decision']) for r in rs)
        for e in ['EXECUTE','CLARIFY','REJECT']:
            for d in ['EXECUTE','CLARIFY','REJECT','ABSTAIN']:matrices.append(meta|{'expected_decision':e,'decision':d,'count':counts[e,d]})
        if meta['variant']=='primary' and meta['k']==3:
            for cw in [1,10,100,1000]:
                for cq in [0,.001,.01,.1]:
                    utilities.append(meta|{'wrong_execution_cost':cw,'false_reject_cost':10,'clarify_cost':.1,'abstain_cost':1,
                        'per_checker_call_cost':cq,'mean_loss':(cw*wrong+10*row['false_rejections']+.1*row['clarifications']+row['abstentions'])/n+cq*row['mean_goal_queries'],
                        'scope':'scenario utility in declared units; LLM token costs exported separately, no invented currency'})
    table(output/'metrics.csv',metrics);table(output/'confusion_matrices.csv',matrices);table(output/'utility_sensitivity.csv',utilities)
    from scipy.stats import binomtest
    comparisons=[]
    for model in models:
        for domain in ['carrying','workflow']:
            for rep in ['LIST','RULE']:
                subset=[r for r in all_rows if r['model']==model and r['domain']==domain and r['representation']==rep and r['k']==3 and r['variant']=='primary'
                        and r['alpha_or_threshold']==.05 and r['query_budget'] is None and r['contract']=='referential'
                        and r['policy'] in ['SEMANTIC_CALIBRATED','DECISION_CALIBRATED']]
                ix={(r['case_id'],r['policy']):r for r in subset};worlds=sorted({r['case_id'] for r in subset})
                for endpoint in ['correct_execution','wrong_decision','false_infeasibility','correct_decision']:
                    dif=[int(ix[c,'DECISION_CALIBRATED'][endpoint])-int(ix[c,'SEMANTIC_CALIBRATED'][endpoint]) for c in worlds]
                    a,b=dif.count(1),dif.count(-1)
                    p=float(binomtest(a,a+b,.5).pvalue) if a+b else 1.
                    comparisons.append({'model':model,'domain':domain,'representation':rep,'endpoint':endpoint,'n':len(worlds),'positive':a,'negative':b,'difference':sum(dif)/len(dif),'exact_p':p})
    rank=sorted(range(len(comparisons)),key=lambda i:comparisons[i]['exact_p']);previous=0
    for j,i in enumerate(rank):
        previous=max(previous,min(1.,(32-j)*comparisons[i]['exact_p']));comparisons[i]['holm_32_p']=previous
    table(output/'paired_exact_tests.csv',comparisons)
    # Plot primary-world operating points, keeping execution denominators explicit.
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,domain in zip(axes,['carrying','workflow']):
        for model in models:
            for rep in ['LIST','RULE']:
                for policy in ['SEMANTIC_CALIBRATED','DECISION_CALIBRATED']:
                    rs=sorted([r for r in metrics if r['domain']==domain and r['model']==model and r['representation']==rep and r['policy']==policy and r['k']==3 and r['variant']=='primary' and r['query_budget'] is None and r['contract']=='referential'],key=lambda r:r['alpha_or_threshold'])
                    ax.plot([r['correct_executions']/r['worlds'] for r in rs],[r['wrong_executions']/r['worlds'] for r in rs],'o-',label=model+'/'+rep+'/'+policy)
        ax.set(title=domain,xlabel='Correct executions / all worlds',ylabel='Wrong executions / all worlds');ax.grid(alpha=.2)
    axes[-1].legend(fontsize=5);fig.tight_layout();fig.savefig(output/'error_autonomy.pdf');plt.close(fig)
    report=('# 无解析器新模型实验\n\n全部预设请求已有终态，独立物理参考和逐目标检查已审计。原始无效、截断、异常输出保留；完成不表示方法优于基线。\n\n'
        +'主推断单元为每领域 512 个独立 primary 世界；同一世界的模型、表达、预算和采样前缀是配对测量，不能相加为样本量。额外 paired 场景只作配对诊断。校准使用另 128 个世界，阈值在测试请求前保存。\n\n'
        +'运行时没有隐藏语义过滤。可见实体构成有限宇宙，模型规则展开只读公共属性。CATALOG_ORACLE 与 AUDIT_ORACLE 使用离线语义标签，显式作为 oracle 对照。\n\n'
        +'主要比较 DECISION_CALIBRATED 与 SEMANTIC_CALIBRATED；若得到 q=1、大量弃权或未见优势，不能据此宣称新方法有效。概率边界是可交换条件下的每世界边际错误上界，不是执行条件错误率保证。\n\n'
        +'语言由程序模板生成，包含空间、操作化的 nearby、所有权和要求关系，不代表真实用户语用验证。物理检查成本通过实际已保存的单目标检查记录按策略调用顺序重放；每次调用计费，跨策略缓存不免费计入科学成本。预算单位为完整目标检查次数，原始工作向量与共享推理 tokens/wall time 单列。\n\n'
        +'列表/谓词、普通采样/推理、1–20 次采样与温度实验均已导出；后两类仅为预设小子集诊断，不能写作全测试集结论。没有前沿 API 模型或人工标签。\n')
    (output/'REPORT.md').write_text(report)
    (ROOT/'docs'/('CCFA_Language_'+batch.name+'.md')).write_text(report+'\n结果目录：`'+str(output.relative_to(ROOT))+'`。\n')
    save(output/'manifest.json',{'status':'COMPLETE','finished_at':now(),'decision_rows':len(all_rows),'model_requests':len(inference),'models':models,'protocol':ANALYSIS_PROTOCOL})
    save(batch/'language_analysis_status.json',{'status':'COMPLETE','at':now(),'output':str(output)},mutable=True)
    return output
