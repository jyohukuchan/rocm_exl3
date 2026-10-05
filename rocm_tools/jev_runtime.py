"""Serialized native EXL3 runtime for JEV System 1 and System 2.

Single GPU (or single-process layer split) only: per-request adapter selection,
separate fresh recurrent state, exact decision head, bounded cached prefill.
"""
from __future__ import annotations
from contextlib import contextmanager
import json
from pathlib import Path
import time


class JEVRuntime:
    def __init__(self, directory, *, context=16384, chunk_size=1024, vision=True):
        import torch
        from exllamav3 import Config, Model, Cache, Tokenizer
        from exllamav3.model.lora import LoRA
        from exllamav3.model.decision import DecisionProfile
        from safetensors.torch import load_file
        from transformers import AutoTokenizer
        self.torch = torch
        self.directory = Path(directory)
        self.context, self.chunk_size = context, chunk_size
        self.config = Config.from_directory(directory)
        if not 256 <= context <= self.config.max_position_embeddings or context % 256:
            raise ValueError('Context must be a multiple of 256 within the model limit')
        self.tokenizer = Tokenizer.from_config(self.config)
        self.hf_tokenizer = AutoTokenizer.from_pretrained(directory)
        generation = json.loads((self.directory/'generation_config.json').read_text())
        eos = generation.get('eos_token_id',self.tokenizer.eos_token_id)
        self.eos_ids = eos if isinstance(eos,list) else [eos]
        self.profile = DecisionProfile.from_directory(directory,
            lambda s: self.tokenizer.encode(s).flatten().tolist())
        self.model = Model.from_config(self.config)
        self.cache = Cache(self.model, max_num_tokens=context, max_batch_size=1)
        self.model.load(device='cuda:0', max_chunk_size=chunk_size, max_output_size=1)
        self.lora = LoRA.from_directory(self.model, str(self.directory/'adapter_vllm'), strict=True,
                                       dtype=torch.float32)
        self.lora.enabled = False
        sidecar = load_file(str(self.directory/'decision_rows.safetensors'))
        if sidecar['token_ids'].tolist() != self.profile.all_ids:
            raise ValueError('Decision rows and tokenizer differ')
        head = self.model.modules[-1]
        a, b = head.lora_a_tensors[self.lora], head.lora_b_tensors[self.lora]
        row_ids = sidecar['token_ids'].to(head.device)
        self.head_weights = (sidecar['base_rows'].float().T.to(head.device)
                             + a.float() @ b.float()[:,row_ids]).contiguous()
        if not bool(torch.isfinite(self.head_weights).all()):
            raise ValueError('Non-finite decision head')
        self.head_lookup = {v:i for i,v in enumerate(self.profile.all_ids)}
        self.vision_model = None
        if vision:
            self.vision_model = Model.from_config(self.config, component='vision')
            self.vision_model.load(device='cuda:0', max_chunk_size=1024)
            self.config.vision_pp.max_pixels = min(self.config.vision_pp.max_pixels,262144)

    def state_parts(self, value):
        """Preserve image/text order using the same MM embeddings as ordinary VL."""
        if isinstance(value,str): return value,[]
        if isinstance(value,dict): return json.dumps(value,ensure_ascii=False),[]
        if not isinstance(value,list): raise ValueError('State must be text, JSON object or a list')
        from rocm_tools.exl3_server.vision import image_bytes,decode_image
        text,embeddings = [],[]
        for part in value:
            if isinstance(part,str): text.append(part)
            elif isinstance(part,dict) and (part.get('type')=='image_url' or 'image' in part):
                if self.vision_model is None: raise ValueError('Vision is disabled')
                if len(embeddings)>=16: raise ValueError('At most 16 images are supported')
                image_part = part if part.get('type')=='image_url' else {'image_url':{'url':part['image']}}
                data,_ = image_bytes(image_part,20*1024**2,False)
                image = decode_image(data,16777216)
                try:
                    emb = self.vision_model.get_image_embeddings(self.tokenizer,image)
                finally: image.close()
                if not bool(self.torch.isfinite(emb.embeddings).all()):
                    raise RuntimeError('Non-finite image embeddings')
                embeddings.append(emb)
                text.append(emb.text_alias)
            elif isinstance(part,dict) and part.get('type')=='text': text.append(part['text'])
            else: text.append(json.dumps(part,ensure_ascii=False))
        return ''.join(text),embeddings

    @contextmanager
    def session(self,prompt,embeddings=(),*,adapter=False,decision=False,reserve=0):
        ids = self.tokenizer.encode(prompt,encode_special_tokens=True,embeddings=list(embeddings))
        n = ids.shape[-1]
        if n<1 or n+reserve>self.context: raise ValueError('Request exceeds the configured context')
        freqs = self.model.g_rope.get_mrope_freqs(ids,list(embeddings),n+reserve)[0] if embeddings else None
        state = self.cache.get_new_state()
        def params(s):
            return {'attn_mode':'flash_attn','cache':self.cache,'past_len':s,
                    'batch_shape':(1,self.context),'recurrent_states':[state],
                    'loras':(self.lora,) if adapter else (), 'last_tokens_only':1,
                    'head_override':self.head_weights if decision else None,
                    'indexed_embeddings':list(embeddings),'inv_freq':freqs}
        try:
            logits = None
            for s in range(0,n,self.chunk_size):
                e = min(n,s+self.chunk_size)
                if e<n: self.model.prefill(ids[:,s:e],params(s))
                else: logits = self.model.forward(ids[:,s:e],params(s))[0,-1].float()
            if not bool(self.torch.isfinite(logits).all()): raise RuntimeError('Non-finite logits')
            yield logits,n,params
        finally:
            self.torch.cuda.synchronize()
            state.free()

    def decide(self,kind,state,question,options=None,*,strategy='single'):
        start = time.monotonic()
        text,embeddings = self.state_parts(state)
        prepared = self.profile.question(kind,question,options)
        if strategy not in ('single','tournament','permute'):raise ValueError('Invalid choice strategy')
        requests,prompt_tokens = 0,0
        def one(opts):
            nonlocal requests,prompt_tokens
            q = self.profile.question(kind,question,opts)
            with self.session('[kind] '+kind+'\n[state] '+text+q['suffix'],embeddings,
                              adapter=True,decision=True) as (logits,n,_):
                raw = [logits[self.head_lookup[i]].item() for i in q['ids']]
                prompt_tokens+=n;requests+=1
                return self.profile.probabilities(raw,q)
        opts = prepared['options']
        if kind!='choice' or len(opts)<=16 or strategy=='single':p=one(opts)
        elif strategy=='permute':
            import random,zlib
            rng=random.Random(zlib.crc32(question.encode()))
            orders=[list(range(len(opts)))]+[rng.sample(range(len(opts)),len(opts)) for _ in range(3)]
            p=[0.0]*len(opts)
            for order in orders:
                for i,v in zip(order,one([opts[j] for j in order])):p[i]+=v/4
        else:
            import math
            groups=math.ceil(len(opts)/16);width,extra=divmod(len(opts),groups)
            grouped=[];in_group={};start_i=0
            for j in range(groups):
                end=start_i+width+(j<extra)
                indices=list(range(start_i,end));grouped.append(indices)
                in_group.update(zip(indices,one(opts[start_i:end])))
                start_i=end
            chosen=[max(g,key=lambda i:(in_group[i],-i)) for g in grouped]
            rest=sorted((i for i in range(len(opts)) if i not in chosen),key=lambda i:(-in_group[i],i))
            finalists=sorted(chosen+rest[:max(0,16-len(chosen))])
            final=dict(zip(finalists,one([opts[i] for i in finalists])))
            group_of={i:j for j,g in enumerate(grouped) for i in g}
            share=[0.0]*groups;cap=[0.0]*groups
            for i in finalists:share[group_of[i]]+=final[i];cap[group_of[i]]+=in_group[i]
            among=sum(a*b for a,b in zip(share,cap))
            p=[final[i]*among if i in final else share[group_of[i]]*in_group[i] for i in range(len(opts))]
            total=sum(p);p=[v/total for v in p]
        best=max(range(len(p)),key=p.__getitem__)
        out={'kind':kind,'effective_kind':kind,'options':opts,'probabilities':p,'choice_index':best,
             'choice':opts[best],'protocol':'jev27-bare-v1','model':str(self.directory.name),
             'adaptation':'native' if kind!='choice' or len(opts)<=16 else strategy,
             'num_model_requests':requests,'elapsed_seconds':time.monotonic()-start,
             'usage':{'prompt_tokens':prompt_tokens,'completion_tokens':0,'total_tokens':prompt_tokens}}
        if kind=='score':out['score']=sum(i*v for i,v in enumerate(p))
        return out

    def generate(self,messages,*,max_tokens=128,temperature=0.0,enable_thinking=False,reasoning_effort=None):
        if not isinstance(max_tokens,int) or not 1<=max_tokens<=self.context:
            raise ValueError('max_tokens is outside the configured context')
        if not isinstance(temperature,(int,float)) or not 0<=temperature<=2:
            raise ValueError('temperature must be between 0 and 2')
        plain,embeddings=[],[]
        for message in messages:
            content,media=self.state_parts(message['content'])
            plain.append({'role':message['role'],'content':content});embeddings.extend(media)
        kwargs={'enable_thinking':enable_thinking}
        if reasoning_effort is not None:kwargs['reasoning_effort']=reasoning_effort
        prompt=self.hf_tokenizer.apply_chat_template(plain,tokenize=False,add_generation_prompt=True,**kwargs)
        return self.generate_prompt(prompt,embeddings,max_tokens,temperature)

    def generate_prompt(self,prompt,embeddings,max_tokens,temperature=0.0,stop_ids=()):
        tokens=[];reason='length'
        with self.session(prompt,embeddings,reserve=max_tokens) as (logits,n,params):
            for i in range(max_tokens):
                logits=logits[:self.tokenizer.actual_vocab_size]
                if not bool(self.torch.isfinite(logits).all()):raise RuntimeError('Non-finite generation logits')
                token=int(logits.argmax()) if temperature==0 else int(self.torch.multinomial(
                    self.torch.softmax(logits/temperature,dim=-1),1))
                if token in self.eos_ids or token in stop_ids:reason='stop';break
                tokens.append(token)
                if i+1<max_tokens:
                    ids=self.torch.tensor([[token]],dtype=self.torch.long)
                    logits=self.model.forward(ids,params(n+i))[0,-1].float()
        text=self.tokenizer.decode(self.torch.tensor(tokens,dtype=self.torch.long),decode_special_tokens=True) if tokens else ''
        return {'text':text,'finish_reason':reason,'usage':{'prompt_tokens':n,'completion_tokens':len(tokens),
                                                          'total_tokens':n+len(tokens)}}

    def adaptive_decide(self,kind,state,question,options=None,*,strategy='single',thinking='off',
                        threshold=0.8,think_budget=1024,return_reasoning=False,debug=False,
                        reasoning_effort=None):
        """Confidence-gated System 2, using the official 50:50 distribution mix."""
        import math
        start=time.monotonic()
        if thinking=='default':thinking='off'
        if thinking not in ('off','auto','on'):raise ValueError('Invalid thinking mode')
        if kind=='score' and thinking!='off':raise ValueError('Thinking supports noul and choice')
        if not isinstance(threshold,(int,float)) or not math.isfinite(threshold) or not 0<=threshold<=1:
            raise ValueError('threshold must be between 0 and 1')
        if not isinstance(think_budget,int) or not 1<=think_budget<self.context:
            raise ValueError('think_budget must be positive and smaller than context')
        if reasoning_effort not in (None,'low','medium','xhigh'):raise ValueError('Invalid reasoning effort')
        result=self.decide(kind,state,question,options,strategy=strategy)
        if thinking=='off':return result
        p1=result['probabilities']
        info={'mode':thinking,'threshold':threshold,'budget':think_budget,'used':False}
        if thinking=='on' or max(p1)<threshold:
            text,embeddings=self.state_parts(state)
            shown=['Yes (true)','No (false)'] if kind=='noul' else result['options']
            # Contextual answer labels differ from bare-v1 option-line labels.
            import string
            prefix=self.tokenizer.encode('Answer: (').flatten().tolist()
            labels=[]
            for lab in list(string.ascii_uppercase)+[a+b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
                ids=self.tokenizer.encode(f'Answer: ({lab})').flatten().tolist()
                if ids[:len(prefix)]==prefix and len(ids)==len(prefix)+2:labels.append((lab,ids[len(prefix)]))
                if len(labels)==len(shown):break
            if len(labels)!=len(shown):raise ValueError('Not enough contextual System 2 answer labels')
            body='\n'.join(f'({lab}) {opt}' for (lab,_),opt in zip(labels,shown))
            user=text+f'\n\nQuestion: {question}\n\nOptions:\n{body}\n\n'+(
                'Think it through carefully, then give your final answer on the last line in the form: Answer: (X)')
            kwargs={'enable_thinking':True}
            if reasoning_effort is not None:kwargs['reasoning_effort']=reasoning_effort
            prompt=self.hf_tokenizer.apply_chat_template([{'role':'user','content':user}],tokenize=False,
                                                        add_generation_prompt=True,**kwargs)
            end=self.tokenizer.encode('</think>',encode_special_tokens=True).flatten().tolist()
            if len(end)!=1:raise ValueError('Model has no single-token thinking delimiter')
            thought=self.generate_prompt(prompt,embeddings,think_budget,stop_ids=end)
            read_prompt=prompt+thought['text']+'</think>\n\nAnswer: ('
            with self.session(read_prompt,embeddings) as (logits,n,_):
                z=[logits[i].item() for _,i in labels];mx=max(z);e=[math.exp(v-mx) for v in z]
                total=sum(e);p2=[v/total for v in e]
            if kind=='noul':p2=[p2[1],p2[0]]
            mixed=[0.5*a+0.5*b for a,b in zip(p1,p2)]
            best=max(range(len(mixed)),key=mixed.__getitem__)
            result.update(probabilities=mixed,choice_index=best,choice=result['options'][best])
            usage=result['usage'];usage['prompt_tokens']+=thought['usage']['prompt_tokens']+n
            usage['completion_tokens']+=thought['usage']['completion_tokens']
            usage['total_tokens']=usage['prompt_tokens']+usage['completion_tokens']
            result['num_model_requests']+=2
            info.update(used=True,think_tokens=thought['usage']['completion_tokens'],
                        finished_within_budget=thought['finish_reason']=='stop',mix=0.5)
            if return_reasoning:info['reasoning']=thought['text'].strip()
            if debug:info.update(system1=p1,system2=p2)
        elif debug:info['system1']=p1
        result['thinking']=info
        result['elapsed_seconds']=time.monotonic()-start
        return result

    def close(self):
        self.lora.unload()
        if self.vision_model is not None:self.vision_model.unload()
        self.model.unload()
