"""Native JEV EXL3 HTTP server: calibrated decisions, TypeSafe batches and VL chat.

GPU work is serialized; fresh KV/recurrent state and explicit LoRA selection
prevent System 1 prefixes or adapters from contaminating System 2 requests.
"""
from __future__ import annotations
import argparse
import asyncio
from contextlib import asynccontextmanager
import hmac
import json
import math
import os
import time
import uuid


def systemone_questions(body):
    if 'state' not in body or not isinstance(body.get('questions'),dict) or not body['questions']:
        raise ValueError('Provide state and a non-empty questions object')
    if len(body['questions'])>128:raise ValueError('At most 128 questions per batch')
    result=[]
    for qid,q in body['questions'].items():
        if not isinstance(q,dict) or q.get('instructions') is None:
            raise ValueError(f'Question {qid} requires instructions')
        kind=q.get('type');criteria=q.get('criteria')
        question=q['instructions'] if isinstance(q['instructions'],str) else json.dumps(q['instructions'],ensure_ascii=False)
        if kind=='choice':
            if not isinstance(criteria,dict) or not 2<=len(criteria)<=256:
                raise ValueError('Choice criteria must have 2..256 keys')
            keys=list(criteria)
            options=[k if criteria[k] is None else f'{k}: {criteria[k]}' for k in keys]
        elif kind=='score':
            if not isinstance(criteria,list) or len(criteria)!=6:
                raise ValueError('JEV score criteria must contain its six trained levels (0..5)')
            keys=[str(i) for i in range(6)]
            options=[str(v) for v in criteria]
        elif kind=='noul':
            if criteria is not None and not isinstance(criteria,dict):raise ValueError('Noul criteria must be an object')
            keys=['false','true']
            options=[k if not criteria or criteria.get(k) is None else f'{k}: {criteria[k]}' for k in keys]
        else:raise ValueError('Question type must be noul, score or choice')
        result.append((qid,kind,question,options,keys,criteria))
    return result


def format_answer(kind,p,keys,criteria):
    best=max(range(len(p)),key=p.__getitem__)
    if kind=='noul':return {'type':kind,'noul':p[1]}
    answer={'type':kind,'probabilities':dict(zip(keys,p))}
    if kind=='choice':
        answer.update(choice=keys[best],confidence=max(0,(max(p)-1/len(p))/(1-1/len(p))))
    else:
        dist=sum(v*abs(i-best) for i,v in enumerate(p))
        uniform=sum(abs(i-(len(p)-1)/2)/len(p) for i in range(len(p)))
        answer.update(score=sum(i*v for i,v in enumerate(p)),legend=dict(zip(keys,criteria)),
                      confidence=max(0,1-dist/uniform))
    return answer


def create_app(runtime,*,model_name='jev27-local',api_key=None):
    from fastapi import FastAPI,HTTPException,Request
    # Make the annotation available to FastAPI when postponed annotations resolve.
    globals()['Request']=Request
    lock=asyncio.Lock()
    @asynccontextmanager
    async def lifespan(app):
        yield
        runtime.close()
    app=FastAPI(lifespan=lifespan)
    async def run(fn,*a,**kw):
        async with lock:
            # Await completion even if the HTTP client disconnects; a cancelled
            # request must not release serialization while its GPU thread runs.
            task=asyncio.create_task(asyncio.to_thread(fn,*a,**kw))
            try:return await asyncio.shield(task)
            except asyncio.CancelledError:
                await task
                raise
            except ValueError as exc:raise HTTPException(400,str(exc)) from exc
    async def body_of(request):
        if api_key and not hmac.compare_digest(request.headers.get('authorization',''),'Bearer '+api_key):
            raise HTTPException(401,'Invalid API key')
        body=await request.json()
        if not isinstance(body,dict):raise HTTPException(400,'Request must be a JSON object')
        if body.get('model',model_name)!=model_name:raise HTTPException(404,'Unknown model')
        return body
    @app.get('/health')
    async def health():return {'status':'ok'}
    @app.get('/v1/models')
    async def models():return {'object':'list','data':[{'id':model_name,'object':'model','owned_by':'local'}]}
    @app.get('/v1/decide/info')
    async def info():
        return {'model':model_name,'protocol':'jev27-bare-v1','max_options':len(runtime.profile.labels),
                'temperatures':runtime.profile.temperatures,'context':runtime.context,
                'thinking':['off','auto','on'],'strategies':['single','permute','tournament']}
    @app.post('/v1/decide')
    async def decide(request:Request):
        body=await body_of(request)
        unknown=set(body)-{'model','kind','state','question','options','strategy','thinking','threshold',
                          'think_budget','return_reasoning','debug','reasoning_effort'}
        if unknown:raise HTTPException(400,'Unsupported decide fields: '+', '.join(sorted(unknown)))
        strategy=body.get('strategy','single')
        if strategy=='auto':strategy='single'
        out=await run(runtime.adaptive_decide,body.get('kind'),body.get('state',''),body.get('question'),body.get('options'),
                      strategy=strategy,thinking=body.get('thinking','off'),threshold=body.get('threshold',0.8),
                      think_budget=body.get('think_budget',1024),return_reasoning=body.get('return_reasoning',False),
                      debug=body.get('debug',False),reasoning_effort=body.get('reasoning_effort'))
        out['model']=model_name
        return out
    @app.post('/v1/systemone')
    async def systemone(request:Request):
        body=await body_of(request)
        try:questions=systemone_questions(body)
        except ValueError as exc:raise HTTPException(400,str(exc)) from exc
        def batch():
            answers={};prompt=0
            for qid,kind,q,opts,keys,criteria in questions:
                out=runtime.decide(kind,body['state'],q,opts)
                answers[qid]=format_answer(kind,out['probabilities'],keys,criteria)
                prompt+=out['usage']['prompt_tokens']
            return {'model':model_name,'answers':answers,'usage':{'prompt_tokens':prompt,'completion_tokens':0,
                                                                 'total_tokens':prompt}}
        return await run(batch)
    @app.post('/v1/chat/completions')
    async def chat(request:Request):
        body=await body_of(request)
        if body.get('stream'):raise HTTPException(400,'This JEV server currently returns non-streaming chat')
        if set(body)&{'tools','tool_choice','response_format','top_p','top_k'}:
            raise HTTPException(400,'Unsupported generation control in the JEV server')
        messages=body.get('messages')
        if not isinstance(messages,list) or not messages:raise HTTPException(400,'Provide messages')
        if any(not isinstance(m,dict) or m.get('role') not in ('system','user','assistant') or 'content' not in m
               for m in messages):raise HTTPException(400,'Invalid chat message')
        template=body.get('chat_template_kwargs') or {}
        if set(template)-{'enable_thinking','reasoning_effort'}:raise HTTPException(400,'Unsupported chat template option')
        out=await run(runtime.generate,messages,max_tokens=body.get('max_tokens',128),
                      temperature=body.get('temperature',0),enable_thinking=template.get('enable_thinking',False),
                      reasoning_effort=template.get('reasoning_effort'))
        return {'id':'chatcmpl-'+uuid.uuid4().hex,'object':'chat.completion','created':int(time.time()),
                'model':model_name,'choices':[{'index':0,'message':{'role':'assistant','content':out['text']},
                                             'finish_reason':out['finish_reason']}],'usage':out['usage']}
    return app


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('-m','--model',required=True)
    ap.add_argument('--context',type=int,default=16384)
    ap.add_argument('--chunk-size',type=int,default=1024)
    ap.add_argument('--host',default='127.0.0.1')
    ap.add_argument('--port',type=int,default=3960)
    ap.add_argument('--model-name',default='jev27-local')
    ap.add_argument('--api-key')
    ap.add_argument('--no-vision',action='store_true')
    args=ap.parse_args()
    # The existing metadata-rescale adapter rejects LoRA. JEV keeps original
    # scales and takes the unfused projection path for both systems.
    os.environ['EXL3_ROCM_MLP_RANGE_BALANCE']='0'
    from rocm_tools.jev_runtime import JEVRuntime
    import uvicorn
    runtime=JEVRuntime(args.model,context=args.context,chunk_size=args.chunk_size,vision=not args.no_vision)
    uvicorn.run(create_app(runtime,model_name=args.model_name,api_key=args.api_key),host=args.host,port=args.port)


if __name__=='__main__':main()
