"""Per-goal physical checker. Only public state and a requested goal enter here."""
from collections import deque
from counterfactual_pilot.solver import run_contract
from counterfactual_pilot.validation import validate_contract_prediction
from innovation_stage3.protocol import families
from reviewer_validation.domain import witness_valid
from language_interface import universe

def check(public,goal):
    if goal not in universe(public):raise ValueError('Unknown visible entity identity')
    r=public['runtime']
    if public['domain']=='carrying':
        s,t=goal.split('->');fs=families([{'action':'place','source_id':s,'target_id':t}])
        result=run_contract(r,fs,None,mode='referential')
        validate_contract_prediction(result,r,fs,None,'referential')
        if result['decision'] not in ['EXECUTE','REJECT']:raise ValueError('Full physical query unresolved')
        return {'feasible':result['decision']=='EXECUTE','work_units':result['cost']['actual_checker_work_units'],
                'cost_unit':'legacy_actual_checker_work_units','raw':result}
    bits={name:1<<i for i,name in enumerate(sorted(r['facts']))}
    mask=lambda names:sum(bits[name] for name in names)
    actions=[(a['id'],mask(a['requires']),mask(a['delete']),mask(a['add'])) for a in r['actions']]
    required=mask(next(g['requires'] for g in r['goals'] if g['id']==goal))
    start=mask(r['initial']);parents={start:None};todo=deque([start]);work=0;end=None
    while todo:
        node=todo.popleft()
        if node&required==required:end=node;break
        for aid,pre,delete,add in actions:
            work+=1
            if node&pre==pre:
                nxt=(node&~delete)|add
                if nxt not in parents:parents[nxt]=(node,aid);todo.append(nxt)
    path=[]
    if end is not None:
        node=end
        while parents[node] is not None:node,aid=parents[node];path.append(aid)
        path.reverse();assert witness_valid(public,goal,path)
    else:
        # Independent set-based closure check, included in reported checker work.
        states={frozenset(name for name,bit in bits.items() if value&bit) for value in parents}
        for state in states:
            for action in r['actions']:
                work+=1
                if set(action['requires'])<=state:
                    assert frozenset((state-set(action['delete']))|set(action['add'])) in states
        assert not any(set(next(g['requires'] for g in r['goals'] if g['id']==goal))<=s for s in states)
    return {'feasible':end is not None,'work_units':work,'cost_unit':'workflow_action_test_including_closure_audit',
            'raw':{'witness':path,'closed':end is None,'visited_integer_states':sorted(parents),'bit_order':sorted(bits)}}

def decide(lower,upper,query,contract='referential',order='lower_first',budget=None):
    """A checked lower witness and exclusion of every upper competitor authorize execution.

    query is an instrumented callback. The policy cannot inspect unqueried labels.
    Budgets count completed individual goal-checker calls, not CPU seconds.
    """
    lower,upper=set(lower),set(upper)
    if not lower<=upper:raise ValueError('Non-nested semantic envelope')
    candidates=sorted(upper) if order=='lexical' else sorted(lower)+sorted(upper-lower)
    positive=set();tested={};sequence=[];work=0
    def decision():
        certain=positive&lower
        if contract=='any_valid' and certain:return 'EXECUTE',min(certain)
        if contract=='referential' and len(certain)>=2:return 'CLARIFY',None
        if len(tested)==len(upper):
            if not positive:return 'REJECT',None
            if len(positive)==1 and certain:return 'EXECUTE',min(certain)
        return 'ABSTAIN',None
    for g in candidates:
        d,selected=decision()
        if d!='ABSTAIN' and order!='eager':break
        if budget is not None and len(tested)>=budget:break
        result=query(g);tested[g]=result['feasible'];sequence.append(g);work+=result['work_units']
        if result['feasible']:positive.add(g)
    d,selected=decision()
    return {'decision':d,'selected_goal':selected,'queries':len(sequence),'queried_goals':sequence,
            'checked_positive_goals':sorted(positive),'work_units':work,'query_budget':budget,
            'lower':sorted(lower),'upper':sorted(upper),'contract':contract,'order':order}
