"""Operational provenance; never rewrites completed scientific records."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from common import HERE, ROOT, now, read, save, file_sha

def freeze(batch, stage, protocol, sources):
    out = batch/'stages'/stage
    out.mkdir(parents=True, exist_ok=True)
    manifest = out/'freeze.json'
    if manifest.exists():
        frozen = read(manifest)
        if frozen['protocol'] != protocol:
            raise RuntimeError('Frozen protocol changed: '+stage)
        for name, sha in frozen['sources'].items():
            if file_sha(ROOT/name) != sha:
                raise RuntimeError('Frozen source changed: '+name)
        return frozen
    hashes = {}
    for path in sorted(set(map(Path, sources))):
        name = str(path.relative_to(ROOT))
        dest = out/'source'/name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        hashes[name] = file_sha(path)
    value = {'stage':stage,'frozen_at':now(),'protocol':protocol,'sources':hashes,
             'registration':'local timestamp and SHA-256 only; not public preregistration'}
    save(manifest,value)
    return value

def snapshot():
    from experiment_resources import plan_resources
    info=plan_resources(16)
    for key,args in [('gpus',['nvidia-smi','--query-gpu=index,uuid,name,memory.total,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader']),
                     ('gpu_processes',['nvidia-smi','--query-compute-apps=pid,gpu_uuid,process_name,used_memory','--format=csv,noheader'])]:
        p=subprocess.run(args,text=True,capture_output=True)
        info[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
    info['resource_override']='Both GPUs may be shared if memory allows; never stop foreign tasks. Temperature recorded, not a dispatch/stop condition (user 2026-09-26). Shared times are not isolated speed comparisons.'
    return info
