"""Local constrained inference, append-only receipts, no semantic label access."""
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import traceback
from common import HERE, ROOT, digest, file_sha, now, read, save
from language_interface import prompt, schema, expand

MODEL_PATHS={'qwen3_8b':os.environ.get('KBS_QWEN_MODEL',''),
             'gemma2_9b':os.environ.get('KBS_GEMMA_MODEL','')}
INFERENCE={'precision':'float16','attention':'sdpa','primary_temperature':.7,'top_p':.95,
           'top_k':20,'max_new_tokens':2048,'thinking_tokens':2048,'batch_size':4,
           'primary_samples':3,'sweep_samples':20,'sweep_test_clusters_per_domain':32,
           'thinking_test_clusters_per_domain':32,'temperature_probe_clusters_per_domain':32,
           'temperature_probe':[0.,1.0],
           'constraint':'lm-format-enforcer 0.11.3 JSON schema; no semantic filtering',
           'failure_policy':'retain invalid/truncated/error; do not automatically retry dispatched requests',
           'thinking':'Qwen3 native thinking prefix, up to 2048 reasoning tokens then constrained JSON answer; forced closing at cap recorded',
           'times':'shared resource wall time and token counts, not exclusive speed comparisons'}

def jobs_for(entries,model):
    jobs=[]
    for c in entries:
        for rep in ['LIST','RULE']:
            for draw in range(1 if c['split']=='development' else 3):
                jobs.append({**c,'representation':rep,'draw':draw,'temperature':.7,'thinking':False,'suite':'primary'})
        if c['split']=='test' and c['variant']=='primary' and c['index']<32:
            for draw in range(3,20):jobs.append({**c,'representation':'LIST','draw':draw,'temperature':.7,'thinking':False,'suite':'sample_sweep'})
            for temperature in INFERENCE['temperature_probe']:
                for rep in ['LIST','RULE']:
                    for draw in range(3):jobs.append({**c,'representation':rep,'draw':draw,'temperature':temperature,'thinking':False,'suite':'temperature'})
            if model=='qwen3_8b':
                for rep in ['LIST','RULE']:
                    for draw in range(3):jobs.append({**c,'representation':rep,'draw':draw,'temperature':.6,'thinking':True,'suite':'thinking'})
    for j in jobs:
        j['model']=model;j['request_id']=digest(j)[:24]
    rank={'development':0,'calibration':1,'test':2}
    suite={'primary':0,'sample_sweep':1,'temperature':2,'thinking':3}
    return sorted(jobs,key=lambda j:(rank[j['split']],suite[j['suite']],j['thinking'],j['temperature'],j['case_id'],j['representation'],j['draw']))

def resource_state():
    r=subprocess.run(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True)
    return {'at':now(),'gpu_csv':r.stdout,'returncode':r.returncode}

class LocalModel:
    def __init__(self,model,out):
        if not MODEL_PATHS.get(model):raise ValueError('Set KBS_QWEN_MODEL or KBS_GEMMA_MODEL to the verified local model directory')
        import torch
        from transformers import AutoTokenizer,AutoModelForCausalLM
        from lmformatenforcer.integrations.transformers import build_token_enforcer_tokenizer_data
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable; launch via authorized host wrapper')
        self.torch=torch;self.model_name=model
        torch.set_num_threads(2)
        model_path=Path(MODEL_PATHS[model]).resolve()
        print('MODEL LOAD',model,model_path,flush=True)
        manifest={str(p.relative_to(model_path)):{'bytes':p.stat().st_size,'sha256':file_sha(p)}
                  for p in sorted(model_path.iterdir()) if p.is_file() and p.suffix in ['.json','.safetensors','.model']}
        save(out/'model_files.json',{'model':model,'resolved_path':str(model_path),'files':manifest})
        self.tokenizer=AutoTokenizer.from_pretrained(str(model_path),local_files_only=True,padding_side='left')
        if self.tokenizer.pad_token_id is None:self.tokenizer.pad_token=self.tokenizer.eos_token
        self.model=AutoModelForCausalLM.from_pretrained(str(model_path),local_files_only=True,
            torch_dtype=torch.float16,device_map={'':'cuda:0'},attn_implementation='sdpa',low_cpu_mem_usage=True)
        self.model.eval()
        self.token_data=build_token_enforcer_tokenizer_data(self.tokenizer)
        save(out/('loaded_'+str(os.getpid())+'.json'),{'at':now(),'pid':os.getpid(),'resource':resource_state(),
             'torch':torch.__version__,'cuda_visible_devices':os.environ.get('CUDA_VISIBLE_DEVICES'),
             'device_name':torch.cuda.get_device_name(0),'generation_config':self.model.generation_config.to_dict()})
        print('MODEL READY',model,flush=True)

    def infer(self,requests,publics):
        from lmformatenforcer import JsonSchemaParser
        from lmformatenforcer.integrations.transformers import build_transformers_prefix_allowed_tokens_fn
        torch=self.torch;tok=self.tokenizer
        temp=requests[0]['temperature'];thinking=requests[0]['thinking']
        assert all(j['temperature']==temp and j['thinking']==thinking for j in requests)
        seed=int(digest([j['request_id'] for j in requests])[:8],16)%(2**31)
        torch.manual_seed(seed);torch.cuda.manual_seed_all(seed);random.seed(seed)
        texts=[];parsers=[]
        for req,pub in zip(requests,publics):
            txt=prompt(pub,req['representation'])
            texts.append(tok.apply_chat_template([{'role':'user','content':txt}],tokenize=False,add_generation_prompt=True,enable_thinking=thinking))
            parsers.append(build_transformers_prefix_allowed_tokens_fn(self.token_data,JsonSchemaParser(schema(pub,req['representation']))))
        inputs=tok(texts,return_tensors='pt',padding=True).to('cuda:0')
        input_lengths=inputs.attention_mask.sum(dim=1).tolist();padded_length=inputs.input_ids.shape[1]
        kwargs={'do_sample':temp>0,'pad_token_id':tok.pad_token_id,'use_cache':True}
        if temp>0:kwargs.update(temperature=temp,top_p=.95,top_k=20)
        start=time.perf_counter();torch.cuda.reset_peak_memory_stats()
        reasoning=['']*len(requests);reason_tokens=[0]*len(requests);forced=[False]*len(requests)
        with torch.inference_mode():
            if thinking:
                assert len(requests)==1,'Reasoning probe uses single request to preserve stopping semantics'
                close=tok.convert_tokens_to_ids('</think>')
                if close==tok.unk_token_id:raise RuntimeError('Missing native Qwen3 thinking token')
                first=self.model.generate(**inputs,**kwargs,max_new_tokens=INFERENCE['thinking_tokens'],eos_token_id=[close,tok.eos_token_id])
                answer=first[0,padded_length:];reasoning=[tok.decode(answer,skip_special_tokens=False)];reason_tokens=[len(answer)]
                forced=[not len(answer) or answer[-1].item()!=close]
                if forced[0]:first=torch.cat([first,torch.tensor([[close]],device=first.device)],dim=1)
                newline=tok.encode('\n',add_special_tokens=False,return_tensors='pt').to(first.device)
                first=torch.cat([first,newline],dim=1)
                inputs={'input_ids':first,'attention_mask':torch.ones_like(first)};padded_length=first.shape[1]
            output=self.model.generate(**inputs,**kwargs,max_new_tokens=INFERENCE['max_new_tokens'],
                prefix_allowed_tokens_fn=lambda batch_id,tokens:parsers[batch_id](batch_id,tokens))
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        result=[]
        for i,(req,pub) in enumerate(zip(requests,publics)):
            ids=output[i,padded_length:].tolist()
            # Strip trailing pad tokens while retaining the first EOS for finish accounting.
            eos=self.model.generation_config.eos_token_id
            ends=set(eos if isinstance(eos,list) else [eos])|{tok.eos_token_id}
            stop=next((k+1 for k,x in enumerate(ids) if x in ends),len(ids));ids=ids[:stop]
            text=tok.decode(ids,skip_special_tokens=True);truncated=len(ids)>=INFERENCE['max_new_tokens'] and (not ids or ids[-1] not in ends)
            error=None;value=None;candidates=None
            try:
                value=json.loads(text);candidates=expand(pub,req['representation'],value)
                if truncated:error='output_token_limit'
            except Exception as e:error=type(e).__name__+': '+str(e)
            result.append({'request':req,'public_sha256':digest(pub),'prompt_sha256':digest(texts[i]),
                'raw_output':text,'parsed':value,'candidates':candidates,'valid':error is None,'error':error,
                'input_tokens':input_lengths[i],'output_tokens':len(ids),'output_token_ids':ids,
                'truncated':truncated,'reasoning_text':reasoning[i],'reasoning_tokens':reason_tokens[i],
                'reasoning_forced_close':forced[i],'batch_seed':seed,'batch_size':len(requests),
                'batch_wall_seconds':elapsed,'allocated_batch_wall_seconds':elapsed/len(requests),
                'peak_cuda_allocated_bytes':torch.cuda.max_memory_allocated(),'finished_at':now()})
        return result

def run(batch,model,gpu,smoke=False):
    batch=Path(batch);out=batch/('smoke' if smoke else 'language')/model;out.mkdir(parents=True,exist_ok=True)
    status=batch/((model+'_smoke_status.json') if smoke else model+'_status.json')
    save(status,{'status':'LOADING','pid':os.getpid(),'gpu':gpu,'at':now()},mutable=True)
    try:
        if smoke:
            from language_cases import make_cluster
            cases=[make_cluster(d,'development',0)[0] for d in ['carrying','workflow']]
            jobs=[];public_by={c['case_id']:c['public'] for c in cases}
            for c in cases:
                for rep in ['LIST','RULE']:
                    j={'case_id':c['case_id'],'representation':rep,'model':model,'temperature':.7,'thinking':False,'suite':'interface_smoke','draw':0}
                    j['request_id']=digest(j)[:24];jobs.append(j)
        else:
            jobs=read(batch/'language'/('jobs_'+model+'.json'))['jobs'];public_by=None
        pending=[]
        for j in jobs:
            rid=j['request_id'];path=out/'responses'/(rid+'.json.gz')
            if path.exists():continue
            started=out/'dispatched'/(rid+'.json')
            if started.exists():
                save(path,{'request':j,'valid':False,'candidates':None,'error':'INTERRUPTED_AFTER_DISPATCH; not retried','finished_at':now()})
            else:pending.append(j)
        if not pending:
            save(status,{'status':'COMPLETE','at':now(),'requests':len(jobs),'no_new_inference':True},mutable=True);return
        local=LocalModel(model,out)
        while pending:
            first=pending[0];n=1 if first['thinking'] else INFERENCE['batch_size'];group=[]
            if not smoke and first.get('split')=='test':
                calibration=batch/'language'/('calibration_'+model+'.json')
                while not calibration.exists():
                    save(status,{'status':'WAITING_CALIBRATION','pid':os.getpid(),'gpu':gpu,'completed':len(jobs)-len(pending),'total':len(jobs),'at':now()},mutable=True)
                    time.sleep(10)
            for j in pending[:n]:
                if (j['temperature'],j['thinking'],j.get('split'),j['suite'])!=(first['temperature'],first['thinking'],first.get('split'),first['suite']):break
                group.append(j)
            pubs=[public_by[j['case_id']] if public_by else read(batch/'language/public'/(j['case_id']+'.json')) for j in group]
            for j,pub in zip(group,pubs):save(out/'dispatched'/(j['request_id']+'.json'),{'at':now(),'request':j,'public_sha256':digest(pub)})
            results=local.infer(group,pubs)
            for j,result in zip(group,results):save(out/'responses'/(j['request_id']+'.json.gz'),result)
            pending=pending[len(group):]
            completed=len(jobs)-len(pending)
            save(status,{'status':'RUNNING','pid':os.getpid(),'gpu':gpu,'completed':completed,'total':len(jobs),
                         'current_split':first.get('split','smoke'),'suite':first['suite'],'at':now()},mutable=True)
            if completed%32==0 or not pending:
                save(out/'resources'/(str(completed)+'.json'),resource_state())
                print('INFERENCE',model,completed,'/',len(jobs),first['suite'],flush=True)
        save(status,{'status':'COMPLETE','at':now(),'requests':len(jobs)},mutable=True)
    except BaseException:
        save(status,{'status':'FAILED','at':now(),'pid':os.getpid(),'traceback':traceback.format_exc()},mutable=True);raise

if __name__=='__main__':run(sys.argv[1],sys.argv[2],int(sys.argv[3]),len(sys.argv)>4 and sys.argv[4]=='smoke')
