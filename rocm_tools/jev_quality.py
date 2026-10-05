"""Small frozen JEV decision-probability screen; BF16 body / FP32 exact head oracle.

Reference runs unquantized HF+PEFT on the V620 pair while EXL3 conversion uses
R9700. Candidate runs are sequential after conversion. This is a regression
screen, not a general calibration/capability benchmark.
"""
from __future__ import annotations
import argparse
import base64
import io
import json
from pathlib import Path
import time
import resource
from unittest.mock import patch


def cases_file(path):
    from PIL import Image,ImageDraw
    cases=[
        {'id':'noul-ja-true','kind':'noul','state':'東京は日本の首都です。','question':'この記述は正しいですか？'},
        {'id':'noul-en-false','kind':'noul','state':'Seven times eight equals fifty-four.','question':'Is the statement mathematically correct?'},
        {'id':'choice-fruit','kind':'choice','state':'A ripe yellow banana.','question':'Which category describes this object?',
         'options':['Vehicle','Fruit','Animal','Building']},
        {'id':'score-quality','kind':'score','state':'Question: What is 2 + 2? Answer: 4.','question':'Rate the correctness of the answer, from 0 (entirely wrong) to 5 (entirely correct).'},
        {'id':'choice-16','kind':'choice','state':'17 + 25','question':'Choose the correct sum.',
         'options':[str(i) for i in range(40,56)]},
        {'id':'choice-256','kind':'choice','state':'The target identifier is 173.','question':'Select the option matching the target identifier.',
         'options':[str(i) for i in range(256)]},
    ]
    for name,color,shape in [('red-square','red','square'),('blue-circle','blue','circle')]:
        image=Image.new('RGB',(256,256),'white');draw=ImageDraw.Draw(image)
        if shape=='square':draw.rectangle((48,48,208,208),fill=color)
        else:draw.ellipse((48,48,208,208),fill=color)
        buffer=io.BytesIO();image.save(buffer,format='PNG');image.close()
        url='data:image/png;base64,'+base64.b64encode(buffer.getvalue()).decode()
        cases.append({'id':'vision-'+name,'kind':'choice','state':[{'image':url}],
                      'question':'What color and shape is shown?',
                      'options':['A red square','A blue circle','A green triangle','A black star']})
    Path(path).write_text(json.dumps(cases,ensure_ascii=False,indent=2)+'\n')


def hf_reference(args,cases):
    import os
    import inspect
    os.environ['USE_HUB_KERNELS']='0'
    import torch
    from transformers import AutoModelForImageTextToText,AutoTokenizer,AutoProcessor
    from peft import PeftModel
    from safetensors import safe_open
    from safetensors.torch import load_file
    from exllamav3.model.decision import DecisionProfile
    from rocm_tools.exl3_server.vision import image_bytes,decode_image
    import transformers.models.qwen3_5.modeling_qwen3_5 as hf_qwen
    # The installed FLA bf16 dot path cannot compile for gfx1030. Use the
    # upstream Torch reference equations rather than alter source precision.
    for name in ('torch_chunk_gated_delta_rule','torch_recurrent_gated_delta_rule',
                 'causal_conv1d_fn','causal_conv1d_update'):
        setattr(hf_qwen,name,inspect.unwrap(getattr(hf_qwen,name)))
    torch.set_num_threads(8)
    tok=AutoTokenizer.from_pretrained(args.model)
    profile=DecisionProfile.from_directory(args.model,lambda s:tok.encode(s,add_special_tokens=False))
    if args.reference_placement=='pair':
        placement={'model.visual':0,'model.language_model.embed_tokens':0,
                   'model.language_model.rotary_emb':0,'lm_head':1,'model.language_model.norm':1}
        placement.update({f'model.language_model.layers.{i}':0 if i<32 else 1 for i in range(64)})
    else:
        # Bound persistent GPU residency; remaining original BF16 modules are
        # staged by Accelerate from CPU. This leaves room for the source LoRA.
        placement={'model.visual':'cpu','model.language_model.embed_tokens':'cpu',
                   'model.language_model.rotary_emb':'cpu','lm_head':'cpu','model.language_model.norm':'cpu'}
        placement.update({f'model.language_model.layers.{i}':0 if i<20 else 1 if i<40 else 'cpu'
                          for i in range(64)})
    print('Loading unquantized BF16 oracle over V620 pair',flush=True)
    # V620 rejects the loader's single ~26GiB warmup allocation despite
    # sufficient free VRAM. Skip that optional allocator optimization; weights
    # still load in their original BF16 dtype and forwards are unmodified.
    with patch('transformers.modeling_utils.caching_allocator_warmup'):
        model=AutoModelForImageTextToText.from_pretrained(args.model,dtype=torch.bfloat16,
                                                        attn_implementation='sdpa',device_map=placement).eval()
    model=PeftModel.from_pretrained(model,str(Path(args.model)/'adapter_vllm')).eval()
    side=load_file(str(Path(args.assets)/'decision_rows.safetensors'))
    with safe_open(str(Path(args.model)/'adapter_vllm/adapter_model.safetensors'),framework='pt') as f:
        a=f.get_tensor('base_model.model.lm_head.lora_A.weight').float()
        bs=f.get_slice('base_model.model.lm_head.lora_B.weight')
        b=torch.cat([bs[i:i+1] for i in profile.all_ids]).float()
    # Share the declared exact-head policy with EXL3: quantization and backbone
    # numerics are compared without BF16 output-head cancellation/rounding noise.
    weight=(side['base_rows'].float()+b@a*2.0).T.contiguous().to('cuda:1')
    class Readout(torch.nn.Module):
        def __init__(self):super().__init__();self.register_buffer('weight',weight)
        def forward(self,x):return x.to(device=self.weight.device,dtype=torch.float32)@self.weight
    model.base_model.model.lm_head=Readout()
    processor=AutoProcessor.from_pretrained(args.model)
    index={i:j for j,i in enumerate(profile.all_ids)}
    result=[]
    from torch.nn.attention import sdpa_kernel,SDPBackend
    with torch.inference_mode(),sdpa_kernel(SDPBackend.MATH):
        for case in cases:
            q=profile.question(case['kind'],case['question'],case.get('options'))
            state=case['state'];images=[]
            if isinstance(state,list):
                pieces=[]
                for part in state:
                    if isinstance(part,str):pieces.append(part)
                    elif 'image' in part:
                        data,_=image_bytes({'image_url':{'url':part['image']}},20*1024**2,False)
                        images.append(decode_image(data,16777216))
                        pieces.append('<|vision_start|><|image_pad|><|vision_end|>')
                text=''.join(pieces)
            else:text=state if isinstance(state,str) else json.dumps(state,ensure_ascii=False)
            prompt='[kind] '+case['kind']+'\n[state] '+text+q['suffix']
            inputs=processor(text=prompt,images=images or None,return_tensors='pt')
            ids=inputs['input_ids'].flatten().tolist()
            inputs={k:v.to('cuda:0') if isinstance(v,torch.Tensor) else v for k,v in inputs.items()}
            start=time.monotonic()
            logits=model(**inputs,use_cache=False,logits_to_keep=1).logits[0,-1].float()
            raw=[logits[index[i]].item() for i in q['ids']]
            p=profile.probabilities(raw,q)
            result.append({'id':case['id'],'probabilities':p,'raw_logits':raw,'input_ids':ids,
                           'seconds':time.monotonic()-start,
                           'image_grid_thw':inputs.get('image_grid_thw').cpu().tolist() if images else None})
            for image in images:image.close()
            print(case['id'],max(range(len(p)),key=p.__getitem__),max(p),flush=True)
    return result,{'torch':torch.__version__,'transformers':__import__('transformers').__version__,
                   'devices':[torch.cuda.get_device_properties(i).name for i in range(torch.cuda.device_count())],
                   'body_dtype':'BF16','head_dtype':'FP32 exact source + LoRA rows','device_map':placement,
                   'loader_allocator_warmup':False,'gdn_implementation':'upstream Torch reference',
                   'sdpa_backend':'MATH'}


def candidate(args,cases):
    import os
    os.environ['EXL3_ROCM_MLP_RANGE_BALANCE']='0'
    from rocm_tools.jev_runtime import JEVRuntime
    split=[float(v) for v in args.gpu_split.split(',')] if args.gpu_split else None
    runtime=JEVRuntime(args.model,context=16384,chunk_size=1024,vision=True,gpu_split=split)
    result=[]
    try:
        for case in cases:
            out=runtime.decide(case['kind'],case['state'],case['question'],case.get('options'))
            result.append({'id':case['id'],'probabilities':out['probabilities'],'seconds':out['elapsed_seconds'],
                           'usage':out['usage']})
            print(case['id'],out['choice_index'],max(out['probabilities']),flush=True)
        # Explicit base→S1→base determinism check: no stale adapter or KV/SSM state.
        messages=[{'role':'user','content':'日本の首都を一語で答えてください。'}]
        before=runtime.generate(messages,max_tokens=16)
        runtime.decide('noul','Tokyo is in Japan.','Is this correct?')
        after=runtime.generate(messages,max_tokens=16)
        if before['text']!=after['text']:raise RuntimeError('System 1 contaminated System 2 generation')
        image_chat=runtime.generate([{'role':'user','content':cases[-2]['state']+[
            'Name the color and shape. Answer briefly.']}],max_tokens=32)
        props=runtime.torch.cuda.get_device_properties(0)
        metadata={'torch':runtime.torch.__version__,'device':props.name,'arch':props.gcnArchName,
                  'device_indices':runtime.device_indices,'gpu_split':split,
                  'quantization_config':runtime.config.config_dict.get('quantization_config'),
                  'base_generation':before,'image_generation':image_chat,'adapter_isolation_pass':True,
                  'peak_vram_bytes':runtime.torch.cuda.max_memory_allocated(0)}
        return result,metadata
    finally:runtime.close()


def compare(reference,candidate_path,output):
    import math
    ref=json.loads(Path(reference).read_text());cand=json.loads(Path(candidate_path).read_text())
    if ref['cases_sha256']!=cand['cases_sha256']:raise ValueError('Case files differ')
    if [r['id'] for r in ref['results']]!=[r['id'] for r in cand['results']]:raise ValueError('Case IDs differ')
    results=[]
    for a,b in zip(ref['results'],cand['results']):
        p,q=a['probabilities'],b['probabilities']
        if len(p)!=len(q) or not all(math.isfinite(v) and v>=0 for v in p+q):raise ValueError('Invalid distribution')
        if abs(sum(p)-1)>1e-6 or abs(sum(q)-1)>1e-6:raise ValueError('Non-normalized distribution')
        kl=sum(x*math.log(x/max(y,1e-300)) for x,y in zip(p,q) if x>0)
        results.append({'id':a['id'],'kl_ref_to_candidate':kl,'max_probability_difference':max(abs(x-y) for x,y in zip(p,q)),
                        'top1_match':max(range(len(p)),key=p.__getitem__)==max(range(len(q)),key=q.__getitem__)})
    report={'reference':str(reference),'candidate':str(candidate_path),'cases':results,
            'mean_kl':sum(r['kl_ref_to_candidate'] for r in results)/len(results)}
    Path(output).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--backend',choices=['cases','hf','exl3','compare'],required=True)
    ap.add_argument('--cases',required=True)
    ap.add_argument('--model')
    ap.add_argument('--assets')
    ap.add_argument('--output',required=True)
    ap.add_argument('--reference')
    ap.add_argument('--candidate')
    ap.add_argument('--reference-placement',choices=['pair','hybrid'],default='pair')
    ap.add_argument('--gpu-split')
    args=ap.parse_args()
    soft,hard=resource.getrlimit(resource.RLIMIT_NOFILE)
    resource.setrlimit(resource.RLIMIT_NOFILE,(max(soft,min(65536,hard)),hard))
    if args.backend=='cases':cases_file(args.cases);return
    if args.backend=='compare':compare(args.reference,args.candidate,args.output);return
    import hashlib
    raw=Path(args.cases).read_bytes();cases=json.loads(raw)
    result,metadata=(hf_reference if args.backend=='hf' else candidate)(args,cases)
    Path(args.output).write_text(json.dumps({'cases_sha256':hashlib.sha256(raw).hexdigest(),
        'backend':args.backend,'model':args.model,'results':result,'metadata':metadata},indent=2,ensure_ascii=False)+'\n')


if __name__=='__main__':main()
