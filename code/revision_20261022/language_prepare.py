from concurrent.futures import ProcessPoolExecutor,as_completed
import os
from pathlib import Path
import sys
import traceback
from common import ROOT, HERE, now, read, save, digest
from experiment_resources import plan_resources
from language_cases import DATA_PROTOCOL, make_cluster, physical_reference
from language_interface import expand,universe
from runtime_checks import check

def job(args):
    batch,domain,split,index=args;batch=Path(batch)
    result=[]
    for c in make_cluster(domain,split,index):
        pub=c['public'];cid=c['case_id'];p=batch/'language'
        save(p/'labels'/(cid+'.json.gz'),c)
        save(p/'public'/(cid+'.json'),pub)
        assert expand(pub,'RULE',c['hidden_spec'])==c['semantic'],'Independent denotation mismatch'
        path=p/'reference'/(cid+'.json.gz')
        if path.exists():ref=read(path)
        else:
            ref=physical_reference(pub);ref['public_sha256']=digest(pub);save(path,ref)
        if ref['public_sha256']!=digest(pub):raise AssertionError('Reference binding changed')
        for g in universe(pub):
            path=p/'checks'/cid/(g.replace('->','_to_')+'.json.gz')
            if path.exists():value=read(path)
            else:
                value=check(pub,g);value['public_sha256']=digest(pub);value['goal']=g;save(path,value)
            if value['public_sha256']!=digest(pub) or value['feasible']!=(g in ref['feasible']):
                raise AssertionError('Independent physical checker mismatch; preserve and stop')
        save(p/'ready'/(cid+'.json'),{'status':'PASS','public_sha256':digest(pub),'all_goal_queries':len(universe(pub)),
                                      'independent_reference_checked':True})
        result.append(cid)
    return result

def prepare_public(batch):
    entries=[]
    for split,n in DATA_PROTOCOL['splits'].items():
        for domain in ['carrying','workflow']:
            for i in range(n):
                for c in make_cluster(domain,split,i):
                    cid=c['case_id'];save(batch/'language/labels'/(cid+'.json.gz'),c);save(batch/'language/public'/(cid+'.json'),c['public'])
                    entries.append({k:v for k,v in c.items() if k not in ['public','hidden_spec','semantic','universe']})
    save(batch/'language/case_manifest.json',{'entries':entries,'cases':len(entries),'protocol':DATA_PROTOCOL})
    return entries

def run(batch):
    batch=Path(batch)
    save(batch/'language_reference_status.json',{'status':'RUNNING','pid':os.getpid(),'at':now()},mutable=True)
    resources=plan_resources(12,mode='physical');save(batch/'language'/('reference_resources_'+str(os.getpid())+'.json'),resources)
    try:
        prepare_public(batch)
        for split,n in DATA_PROTOCOL['splits'].items():
            with ProcessPoolExecutor(max_workers=resources['workers']) as pool:
                fs=[pool.submit(job,(str(batch),d,split,i)) for d in ['carrying','workflow'] for i in range(n)]
                for j,f in enumerate(as_completed(fs)):
                    f.result()
                    if j%8==0:
                        print('REFERENCE',split,j+1,'/',2*n,flush=True)
                        save(batch/'language_reference_status.json',{'status':'RUNNING','pid':os.getpid(),'split':split,'finished_clusters':j+1,'total_clusters':2*n,'at':now()},mutable=True)
        save(batch/'language_reference_status.json',{'status':'COMPLETE','at':now()},mutable=True)
    except BaseException:
        save(batch/'language_reference_status.json',{'status':'FAILED','at':now(),'traceback':traceback.format_exc()},mutable=True);raise

if __name__=='__main__':run(Path(sys.argv[1]).resolve())
