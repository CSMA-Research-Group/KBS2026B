"""Offline controlled-language generator and labels, isolated from runtime inference."""
from collections import deque
import copy
import random
from common import digest
from counterfactual_pilot.core import make_cases as old_cases
from reviewer_validation.domain import workflow_world
from innovation_stage2.reference import reference_oracle
from innovation_stage2.domain import validate_witness
from language_interface import universe

DATA_PROTOCOL={'seed':2026102213,'splits':{'development':8,'calibration':128,'test':512},
               'variants':'one iid primary per cluster; test first 64 and calibration first 32 have paired +F/+I diagnostic scenes',
               'styles':['color_paraphrase','logic_reordering','spatial_or_requirement','operational_near_or_requirement','speaker_reference'],
               'primary_distribution':'independent RNG seeds; nuisance density, instruction style, and 0/1/2 intended feasible base types sampled per cluster',
               'no_human_annotation':True,'no_semantic_runtime_filter':True}

def hidden_members(public,spec):
    # Independent comprehension, not the model-rule evaluator.
    r=public['runtime'];ctx=public['context']
    if public['domain']=='workflow':
        return sorted(g['id'] for g in r['goals'] if g['tag'] in spec['tags'] and g['kind'] in spec['kinds']
            and (spec['owner']=='any' or ctx['owners'][g['id']]==ctx['speaker'])
            and (spec['requires_fact']=='any' or spec['requires_fact'] in g['requires']))
    def source(s):
        return (s['color'] in spec['source_colors'] and s['shape'] in spec['source_shapes']
            and (spec['owner']=='any' or ctx['owners'][s['id']]==ctx['speaker'])
            and (spec['source_region']=='any' or sum(abs(x-y) for x,y in zip(s['cell'],r['robot']['start']))<=ctx['near_distance']))
    def target(t):
        x,rx=t['cell'][0],r['robot']['start'][0]
        return (t['color'] in spec['target_colors'] and t['shape'] in spec['target_shapes']
            and (spec['target_side']=='any' or (x<rx if spec['target_side']=='west_of_robot' else x>rx)))
    return sorted(s['id']+'->'+t['id'] for s in r['objects'] if source(s) for t in r['targets'] if target(t)
                  and (spec['pairing']=='any' or s['color'] in t['accepts_colors']))

def words(values):return ' or '.join(values)

def render(domain,spec,style):
    if domain=='carrying':
        a,b=words(spec['source_colors']),words(spec['target_colors'])
        text=(f'The intended placement must use a {a} block and a {b} tray. The tray must accept the block\'s color.'
              if style%2==0 else f'I mean a tray colored {b}, with a block colored {a} placed into it. Only color-compatible pairs are permitted.')
        if spec['target_side']!='any':text+=' The tray must be strictly '+('west' if spec['target_side'].startswith('west') else 'east')+' of the robot\'s initial location.'
        if spec['source_region']!='any':text+=' Only nearby blocks qualify; nearby means within the public near_distance in Manhattan grid distance from the robot initially.'
        if spec['owner']=='speaker':text+=' I am referring to my blocks; the public ownership record identifies which ones are mine.'
    else:
        a=words(spec['tags'])
        text=(f'The endpoint I intend is a finished one with tag {a}.' if style%2==0
              else f'Use either the {a} tag, but only on endpoints marked finished.')
        if spec['requires_fact']!='any':text+=f" Its stated requirements must include {spec['requires_fact']}."
        if spec['owner']=='speaker':text+=' I mean my endpoints, as identified by the public ownership record.'
    return text+' These are semantic restrictions; physical feasibility will be checked separately.'

def make_cluster(domain,split,index):
    key=f"ccfa-language:{DATA_PROTOCOL['seed']}:{domain}:{split}:{index}"
    rng=random.Random(key);seed=int(digest(key)[:12],16);style=rng.randrange(5)
    base_type=rng.randrange(3);density=rng.choice([0,4,8]);speaker=rng.choice(['Mira','Noah','Jules'])
    paired=(split=='test' and index<64) or (split=='calibration' and index<32)
    if domain=='carrying':
        old_index=rng.randrange(24)
        originals=[c for c in old_cases({'base_worlds':24,'world_seed':seed}) if c['contract']=='referential'][old_index*3:old_index*3+3]
        runtimes={c['variant']:copy.deepcopy(c['runtime']) for c in originals}
        primary=originals[0]['primary_goal'][6:];source_id,target_id=primary.split('->')
        original=runtimes['base'];obj=next(o for o in original['objects'] if o['id']==source_id)
        source_colors=[obj['color'],next(o['color'] for o in original['objects'] if o['mass']>5)]
        target_colors=[next(t['color'] for t in original['targets'] if t['id']==target_id),
                       next(t['color'] for t in original['targets'] if t['accepts_colors']==[source_colors[1]])]
        names={o['id']:f'o{i}' for i,o in enumerate(sorted(original['objects'],key=lambda o:o['id']))}
        names.update({v:f't{i}' for i,v in enumerate(sorted({t['id'] for r in runtimes.values() for t in r['targets']}))})
        owners={names[o['id']]:rng.choice([speaker,'other']) for o in original['objects']}
        owners[names[source_id]]=speaker
        for r in runtimes.values():
            for collection in ['objects','targets']:
                for entity in r[collection]:entity['id']=names[entity['id']]
            r['objects'][0]['footprint']=[1,1] if density==8 else r['objects'][0]['footprint']
            # Extra visible distractors create size/density variation without rejection sampling.
            for j in range(density):
                o=copy.deepcopy(r['objects'][j%len(original['objects'])]);o['id']=f'd{j}'
                o['color']=source_colors[j%2];o['shape']='cone' if j%3==0 else 'block'
                o['mass']=9;o['movable']=False
                o['cell']=[2+j%3,2+j%5];r['objects'].append(o);owners[o['id']]='other'
        # The primary iid draw includes infeasible, unique, and competitive base scenes.
        primary_runtime=copy.deepcopy(runtimes['add_feasible'] if base_type==2 else runtimes['base'])
        if base_type==0:
            for o in primary_runtime['objects']:
                if o['color'] in source_colors:o['movable']=False
        spec={'source_colors':source_colors,'target_colors':target_colors,'source_shapes':['block'],
              'target_shapes':['tray'],'pairing':'compatible','target_side':'any','source_region':'any','owner':'any'}
        if style==2:spec['target_side']=rng.choice(['west_of_robot','east_of_robot'])
        if style==3:spec['source_region']='near_robot'
        if style==4:spec['owner']='speaker'
        context={'speaker':speaker,'owners':owners,'near_distance':rng.choice([3,5,8])}
    else:
        base,ast,primary,addition,impossible=workflow_world(seed,rng.randrange(100000))
        owners={g['id']:rng.choice([speaker,'other']) for g in base['goals']};owners[primary]=speaker;owners[addition['id']]=speaker
        for j in range(density):
            g=copy.deepcopy(base['goals'][j%5]);g['id']=f'd{j}';g['tag']=rng.choice(['amber','purple','white']);
            g['kind']='intermediate' if j%2 else 'finished';base['goals'].append(g);owners[g['id']]='other'
        runtimes={'base':base,'add_feasible':copy.deepcopy(base),'add_infeasible':copy.deepcopy(base)}
        runtimes['add_feasible']['goals'].append(addition)
        runtimes['add_infeasible']['goals'].append({**addition,'requires':[impossible]})
        primary_runtime=copy.deepcopy(runtimes['add_feasible'] if base_type==2 else base)
        if base_type==0:
            for g in primary_runtime['goals']:
                if g['tag'] in ast['tags'] and g['kind']==ast['kind']:g['requires']=[impossible]
        spec={'tags':ast['tags'],'kinds':[ast['kind']],'owner':'any','requires_fact':'any'}
        if style in [2,3]:spec['requires_fact']=rng.choice(next(g['requires'] for g in base['goals'] if g['id']==primary))
        if style==4:spec['owner']='speaker'
        context={'speaker':speaker,'owners':owners}
    scenes=[('primary',primary_runtime)]
    if paired:scenes.extend((v,runtimes[v]) for v in ['base','add_feasible','add_infeasible'])
    result=[]
    cluster=f'{domain}_{split}_{index:04d}'
    for variant,runtime in scenes:
        runtime['instruction']=render(domain,spec,style)
        public={'domain':domain,'runtime':runtime,'context':context}
        result.append({'case_id':cluster+'_'+variant,'cluster_id':cluster,'domain':domain,'split':split,
                       'index':index,'variant':variant,'style':DATA_PROTOCOL['styles'][style],
                       'base_type':base_type,'distractors':density,'public':public,'hidden_spec':spec,
                       'semantic':hidden_members(public,spec),'universe':universe(public)})
    return result

def physical_reference(public):
    r=public['runtime']
    if public['domain']=='carrying':
        def disj(field,values):return {'op':'or','args':[{'op':'eq','field':field,'value':v} for v in sorted(set(values))]}
        ast={'action':'place','pairing':'any','source':disj('color',[o['color'] for o in r['objects']]),
             'target':disj('color',[t['color'] for t in r['targets']])}
        value=reference_oracle(r,ast,1000000)
        if not value['complete']:raise RuntimeError('Independent reference incomplete; retain and stop')
        for gid,path in value['witnesses'].items():
            s,t=gid[6:].split('->')
            assert validate_witness(r,{'action':'place','source_id':s,'target_id':t},path)
        return {'feasible':sorted(g[6:] for g in value['feasible_goals']),'raw':value}
    initial=frozenset(r['initial']);parents={initial:None};todo=deque([initial])
    while todo:
        node=todo.popleft()
        for a in r['actions']:
            if set(a['requires'])<=node:
                nxt=frozenset((node-set(a['delete']))|set(a['add']))
                if nxt not in parents:parents[nxt]=(node,a['id']);todo.append(nxt)
                if len(parents)>1000000:raise RuntimeError('Workflow reference bound exceeded')
    witnesses={}
    for g in r['goals']:
        node=next((p for p in parents if set(g['requires'])<=p),None)
        if node is not None:
            seq=[]
            while parents[node] is not None:node,aid=parents[node];seq.append(aid)
            witnesses[g['id']]=list(reversed(seq))
    return {'feasible':sorted(witnesses),'raw':{'closed_states':sorted([sorted(n) for n in parents]),'witnesses':witnesses}}
