"""Runtime-only language interface. No hidden specification or membership oracle."""
import json

COLORS=['red','blue','green','yellow','orange','purple','amber','white']

def universe(public):
    r=public['runtime']
    return sorted(s['id']+'->'+t['id'] for s in r['objects'] for t in r['targets']) if public['domain']=='carrying' else sorted(g['id'] for g in r['goals'])

def schema(public,representation):
    def choices(values):return {'type':'string','enum':sorted(set(values))}
    def array(values):
        values=sorted(set(values))
        return {'type':'array','items':choices(values),'maxItems':len(values)}
    if representation=='LIST':props={'candidates':array(universe(public))}
    elif public['domain']=='carrying':
        r=public['runtime']
        props={'source_colors':array(o['color'] for o in r['objects']),
               'target_colors':array(t['color'] for t in r['targets']),
               'source_shapes':array(o['shape'] for o in r['objects']),
               'target_shapes':array(t['shape'] for t in r['targets']),
               'pairing':choices(['any','compatible']),
               'target_side':choices(['any','west_of_robot','east_of_robot']),
               'source_region':choices(['any','near_robot']),
               'owner':choices(['any','speaker'])}
    else:
        r=public['runtime']
        props={'tags':array(g['tag'] for g in r['goals']),
               'kinds':array(g['kind'] for g in r['goals']),
               'owner':choices(['any','speaker']),
               'requires_fact':choices(['any']+r['facts'])}
    return {'type':'object','properties':props,'required':list(props),'additionalProperties':False}

def expand(public,representation,value):
    """Evaluate a MODEL-authored rule against public attributes, without filtering its semantics."""
    import jsonschema
    jsonschema.validate(value,schema(public,representation))
    if representation=='LIST':return sorted(set(value['candidates']))
    r=public['runtime'];ctx=public['context'];out=[]
    if public['domain']=='workflow':
        for g in r['goals']:
            if g['tag'] not in value['tags'] or g['kind'] not in value['kinds']:continue
            if value['owner']=='speaker' and ctx['owners'][g['id']]!=ctx['speaker']:continue
            if value['requires_fact']!='any' and value['requires_fact'] not in g['requires']:continue
            out.append(g['id'])
    else:
        rx,ry=r['robot']['start']
        for s in r['objects']:
            if s['color'] not in value['source_colors'] or s['shape'] not in value['source_shapes']:continue
            if value['owner']=='speaker' and ctx['owners'][s['id']]!=ctx['speaker']:continue
            if value['source_region']=='near_robot' and abs(s['cell'][0]-rx)+abs(s['cell'][1]-ry)>ctx['near_distance']:continue
            for t in r['targets']:
                if t['color'] not in value['target_colors'] or t['shape'] not in value['target_shapes']:continue
                if value['pairing']=='compatible' and s['color'] not in t['accepts_colors']:continue
                if value['target_side']=='west_of_robot' and t['cell'][0]>=rx:continue
                if value['target_side']=='east_of_robot' and t['cell'][0]<=rx:continue
                out.append(s['id']+'->'+t['id'])
    return sorted(set(out))

def prompt(public,representation):
    # The same public scene and task are supplied to both representations.
    pre=('Interpret every goal that the user language permits, without deciding physical feasibility. '
         'Do not remove a permitted goal because an object is heavy, immovable, unreachable, or a workflow is impossible. '
         'A separate checker handles feasibility. The user explicitly requires this decision contract: execute only '
         'if exactly one permitted feasible goal exists; clarify if two or more exist; reject if none exists; '
         'abstain when the evidence cannot settle the decision. Your output here is only the semantic interpretation. '
         'Return one JSON object conforming to the schema. No prose.\n')
    if representation=='LIST':
        pre+='List all permitted goal IDs in candidates. Carrying IDs have the form objectID->targetID. Use [] for an empty set.\n'
    else:
        pre+=('Return the predicate, using every field. Values within an attribute array are alternatives (OR); '
              'different fields are conjoined (AND). An unrestricted attribute lists all its visible values. '
              'An empty attribute array matches no entities. any removes that optional restriction. '
              'compatible means the source color is included in the target accepts_colors. '
              'west/east use a strict x comparison to the robot; near uses the public Manhattan distance threshold. '
              'speaker selects entities whose recorded owner equals the public speaker. requires_fact refers to '
              'membership of a fact in an endpoint requirement list, not whether the fact is reachable.\n')
    pre+=('Example of interpretation: "red or blue blocks into green trays that accept their color" means both '
          'source colors red and blue, source shape block, target color green, target shape tray and compatible pairing. '
          'It does not mean red only. "finished endpoints tagged red or blue" permits both tags and excludes intermediate endpoints.\n')
    return pre+'Schema: '+json.dumps(schema(public,representation),sort_keys=True)+'\nPublic input: '+json.dumps(public,sort_keys=True)
