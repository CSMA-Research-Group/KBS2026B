"""Decision safety over all finite completions, plus independent interface checks."""
from itertools import product,combinations
from pathlib import Path
from common import save,now
from runtime_checks import decide,check
from language_cases import make_cluster,physical_reference,hidden_members
from language_interface import expand,universe
from language_analysis import score,quantile

def validate(out):
    gs=['a','b','c'];subsets=[set(c) for k in range(4) for c in combinations(gs,k)];n=0
    for physical in subsets:
        for lower in subsets:
            for upper in subsets:
                if not lower<=upper:continue
                possible=[s for s in subsets if (lower&physical)<=(s&physical)<=(upper&physical)]
                for contract in ['referential','any_valid']:
                    for budget in [0,1,2,3,None]:
                        for order in ['lower_first','lexical','eager']:
                            called=[]
                            def query(g):called.append(g);return {'feasible':g in physical,'work_units':1}
                            v=decide(lower,upper,query,contract,order,budget)
                            assert len(called)==len(set(called)) and set(called)<=upper
                            assert len(called)<=len(upper) and (budget is None or len(called)<=budget)
                            for semantic in possible:
                                assert not score(v,semantic&physical,semantic)['wrong_decision'],(physical,lower,upper,semantic,v)
                            n+=1
    pairs=0;roundtrip=[]
    for domain in ['carrying','workflow']:
        for i in range(8):
            c=make_cluster(domain,'development',i)[0]
            assert expand(c['public'],'RULE',c['hidden_spec'])==c['semantic']
            ref=physical_reference(c['public'])
            for g in c['universe'][::max(1,len(c['universe'])//4)]:
                r=check(c['public'],g);assert r['feasible']==(g in ref['feasible']);pairs+=1
            roundtrip.append({'case_id':c['case_id'],'physical_reference_complete':True,'denotation_match':True})
    # A paired intervention changes only the added target/endpoint, not distractor attributes.
    for domain in ['carrying','workflow']:
        cs=make_cluster(domain,'calibration',0);plus=[c for c in cs if c['variant'].startswith('add_')]
        a,b=plus[0]['public']['runtime'],plus[1]['public']['runtime']
        if domain=='carrying':assert a['objects']==b['objects'] and a['grid']==b['grid']
        else:assert a['actions']==b['actions'] and a['initial']==b['initial']
    assert quantile([0.]*128,.05)==.5 and quantile([1.]*128,.05)==1.
    result={'status':'PASS','at':now(),'finite_completion_policy_checks':n,'independent_physical_crosschecks':pairs,
            'development_roundtrip':roundtrip,'model_accuracy_not_a_gate':True}
    save(Path(out)/'preflight_validation.json',result);return result

if __name__=='__main__':
    import sys
    print(validate(Path(sys.argv[1])))
