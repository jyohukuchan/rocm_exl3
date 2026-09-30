"""Verify actual HTTP dynamic batching and independent per-request constraints."""
import argparse
import asyncio
import json
import os
from pathlib import Path
import time
import httpx


async def verify(a):
    headers = {'Authorization': 'Bearer ' + os.environ.get('EXL3_API_KEY', '')}
    evidence = {'started_unix_s': time.time(), 'batch': a.batch, 'observations': [],
                'requests': [], 'complete': False}
    def save():
        Path(a.output).write_text(json.dumps(evidence,ensure_ascii=False,indent=2)+'\n')
    async with httpx.AsyncClient(base_url=a.base_url, headers=headers, timeout=180) as c:
        evidence['startup'] = (await c.get('/props')).json()['runtime']
        save()
        configured = evidence['startup']['generator_runtime']['configured_max_batch_size']
        assert configured >= a.batch, configured
        async def call(i, constrained=False):
            body = {'model': a.model, 'enable_thinking': False, 'max_tokens': 256,
                    'messages': [{'role': 'user', 'content':
                        f'Job {i}. Write a Python implementation and tests for a trie. '
                        'Include the complete code and detailed examples. '
                        + ('Use strings as keys. ' * 150)}]}
            if constrained:
                value = f'job-{i}'
                body.update(messages=[{'role': 'user', 'content': 'Call mark with the permitted value.'}],
                    tools=[{'type':'function','function':{'name':'mark','parameters':{
                        'type':'object','properties':{'value':{'type':'string','enum':[value]}},
                        'required':['value'],'additionalProperties':False}}}],
                    tool_choice='required', parallel_tool_calls=False)
            start = time.monotonic()
            r = await c.post('/v1/chat/completions', json=body)
            result = {'job': i, 'status': r.status_code, 'wall_s': time.monotonic()-start,
                      'response': r.json()}
            evidence['requests'].append(result)
            save()
            assert r.status_code == 200, result
            if constrained:
                calls = result['response']['choices'][0]['message']['tool_calls']
                assert json.loads(calls[0]['function']['arguments']) == {'value':value}, calls
            return result
        jobs = [asyncio.create_task(call(i)) for i in range(a.batch)]
        while not all(j.done() for j in jobs):
            props = (await c.get('/props')).json()['runtime']
            evidence['observations'].append({'unix_s':time.time(),
                'generator':props['generator_runtime'], 'power':props.get('power_policy')})
            save()
            await asyncio.sleep(.2)
        evidence['text_requests'] = await asyncio.gather(*jobs)
        peak = max(x['generator']['active_jobs'] for x in evidence['observations'])
        assert peak == a.batch, peak
        evidence['constrained_requests'] = await asyncio.gather(*(call(i, True) for i in range(a.batch)))
        evidence['final_runtime'] = (await c.get('/props')).json()['runtime']
        evidence['complete'] = True
    save()
    print(f'PASS dynamic batch {peak}; {a.batch} independent constrained jobs')


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--base-url',default='http://127.0.0.1:3953')
    p.add_argument('--model',default='qwen38-local')
    p.add_argument('--batch',type=int,default=4)
    p.add_argument('--output',required=True)
    asyncio.run(verify(p.parse_args()))
