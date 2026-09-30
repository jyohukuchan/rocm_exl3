"""Exercise a real exl3 HTTP server; writes request/response evidence as JSON."""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import time
import httpx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:3953')
    ap.add_argument('--model', default='qwen38-local')
    ap.add_argument('--output', required=True)
    ap.add_argument('--case', action='append', default=None,
                    help='run only named checks (repeatable); default runs all')
    a = ap.parse_args()
    allowed_cases = {
        'named_tool', 'tool_history_and_none', 'auto_tool', 'parallel_tools',
        'json_schema', 'thinking_json_schema', 'json_schema_adversarial_enum',
        'json_object', 'sse_tool_arguments', 'prefix_cache',
        'text_completion_sse',
        'invalid_tool_choice', 'unknown_model', 'oversized_context',
        'stream_cancellation_cleanup',
    }
    unknown_cases = set(a.case or ()) - allowed_cases
    if unknown_cases:
        ap.error('unknown --case value(s): ' + ', '.join(sorted(unknown_cases)))
    evidence = {'model': a.model, 'tests': {}, 'requests': []}
    headers = {'Authorization': 'Bearer ' + os.environ.get('EXL3_API_KEY', '')}
    client = httpx.Client(base_url=a.base_url, headers=headers, timeout=180)
    selected = set(a.case or ())

    def save():
        Path(a.output).write_text(json.dumps(evidence, ensure_ascii=False, indent=2) + '\n')

    def call(body, expected=200):
        body = {'model': a.model, 'max_tokens': 256, 'enable_thinking': False, **body}
        t = time.monotonic()
        response = client.post('/v1/chat/completions', json=body)
        data = response.json()
        evidence['requests'].append({'request': body, 'status': response.status_code,
                                     'response': data, 'wall_s': time.monotonic() - t})
        save()
        assert response.status_code == expected, data
        return data

    def check(name, fn):
        if selected and name not in selected:
            evidence['tests'][name] = {'pass': True, 'skipped': True}
            return
        try:
            detail = fn()
            evidence['tests'][name] = {'pass': True, 'detail': detail}
            print('PASS', name, flush=True)
        except Exception as e:
            evidence['tests'][name] = {'pass': False, 'error': repr(e)}
            save()
            raise
        save()

    props = client.get('/props').json()
    evidence['startup'] = props
    assert client.get('/health').status_code == 200
    assert props['runtime']['ok']
    echo = {'type': 'function', 'function': {
        'name': 'echo', 'description': 'Return text unchanged',
        'parameters': {'type': 'object', 'properties': {'text': {'type': 'string'}},
                       'required': ['text'], 'additionalProperties': False}}}
    user = {'role': 'user', 'content': 'Call echo with text Tokyo.'}
    first = {}

    def named():
        data = call({'messages': [user], 'tools': [echo], 'parallel_tool_calls': False,
                     'tool_choice': {'type': 'function', 'function': {'name': 'echo'}}})
        msg = data['choices'][0]['message']
        assert data['choices'][0]['finish_reason'] == 'tool_calls'
        assert json.loads(msg['tool_calls'][0]['function']['arguments']) == {'text': 'Tokyo'}
        first.update(msg)
        return msg['tool_calls'][0]['id']
    check('named_tool', named)

    def history():
        tool = first['tool_calls'][0]
        messages = [user, first, {'role': 'tool', 'name': 'echo', 'tool_call_id': tool['id'],
                    'content': 'Tokyo'}, {'role': 'user', 'content': 'Reply exactly Received: Tokyo'}]
        data = call({'messages': messages, 'tools': [echo], 'tool_choice': 'none'})
        assert not data['choices'][0]['message'].get('tool_calls')
        assert 'Tokyo' in data['choices'][0]['message']['content']
        return data['choices'][0]['message']
    check('tool_history_and_none', history)

    def automatic():
        data = call({'messages': [user], 'tools': [echo], 'tool_choice': 'auto'})
        assert data['choices'][0]['message'].get('tool_calls')
        return data['choices'][0]['message']
    check('auto_tool', automatic)

    def parallel():
        data = call({'messages': [{'role': 'user', 'content':
            'Make two separate echo tool calls, first with text A, then with text B. Return both calls in this response.'}],
            'tools': [echo], 'tool_choice': 'required', 'parallel_tool_calls': True})
        calls = data['choices'][0]['message']['tool_calls']
        assert len(calls) == 2, calls
        assert {json.loads(c['function']['arguments'])['text'] for c in calls} == {'A', 'B'}
        assert len({c['id'] for c in calls}) == 2
        return calls
    check('parallel_tools', parallel)

    response_format = {'type': 'json_schema', 'json_schema': {'name': 'math', 'strict': True,
        'schema': {'type': 'object', 'properties': {'answer': {'type': 'integer', 'enum': [4]}},
                   'required': ['answer'], 'additionalProperties': False}}}
    def structured(thinking):
        data = call({'messages': [{'role': 'user', 'content': 'What is 2+2?'}],
                     'response_format': response_format, 'enable_thinking': thinking,
                     'reasoning_effort': 'low'})
        msg = data['choices'][0]['message']
        assert json.loads(msg['content']) == {'answer': 4}
        assert '<think>' not in msg['content']
        if thinking:
            assert msg.get('reasoning_content')
        return msg
    check('json_schema', lambda: structured(False))
    check('thinking_json_schema', lambda: structured(True))
    def adversarial_schema():
        schema = {'type': 'json_schema', 'json_schema': {'name': 'answer', 'strict': True,
            'schema': {'type': 'object', 'properties': {'answer': {'type': 'integer', 'enum': [4, 9, 13]}},
                       'required': ['answer'], 'additionalProperties': False}}}
        data = call({'messages': [{'role': 'user', 'content': 'Answer with the number 9 only.'}],
                     'response_format': schema})
        value = json.loads(data['choices'][0]['message']['content'])
        assert isinstance(value, dict) and value['answer'] in {4, 9, 13}
        return value
    check('json_schema_adversarial_enum', adversarial_schema)

    def json_object():
        data = call({'messages': [{'role': 'user', 'content': 'Reply with a JSON object with answer 4.'}],
                     'response_format': {'type': 'json_object'}})
        value = json.loads(data['choices'][0]['message']['content'])
        assert isinstance(value, dict), value
        return value
    check('json_object', json_object)

    def streaming():
        value = 'line1\n東京\n</parameter>'
        tool = json.loads(json.dumps(echo))
        tool['function']['parameters']['properties']['text']['enum'] = [value]
        body = {'model': a.model, 'messages': [{'role': 'user', 'content': 'Echo the only permitted text.'}],
                'tools': [tool], 'tool_choice': 'required', 'parallel_tool_calls': False,
                'enable_thinking': False, 'max_tokens': 256, 'stream': True,
                'stream_options': {'include_usage': True}}
        chunks, args, ids, names, reasons = [], '', [], [], []
        with client.stream('POST', '/v1/chat/completions', json=body) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if not line.startswith('data: '): continue
                raw = line[6:]
                if raw == '[DONE]': break
                chunk = json.loads(raw); chunks.append(chunk)
                assert 'error' not in chunk, chunk
                for choice in chunk.get('choices', []):
                    if choice.get('finish_reason'): reasons.append(choice['finish_reason'])
                    for call_data in choice['delta'].get('tool_calls', []):
                        if call_data.get('id'): ids.append(call_data['id'])
                        if call_data.get('function', {}).get('name'): names.append(call_data['function']['name'])
                        args += call_data.get('function', {}).get('arguments', '')
        evidence['requests'].append({'request': body, 'chunks': chunks}); save()
        assert json.loads(args) == {'text': value}, args
        assert ids and names == ['echo'] and reasons == ['tool_calls']
        assert len([c for c in chunks if c.get('choices', [{}])[0].get('delta', {}).get('tool_calls')]) > 2
        return {'chunks': len(chunks), 'arguments': args}
    check('sse_tool_arguments', streaming)

    def text_completion_sse():
        body = {'model': a.model, 'prompt': 'Reply with one short sentence.',
                'enable_thinking': False, 'max_tokens': 32, 'stream': True}
        chunks = []
        with client.stream('POST', '/v1/completions', json=body) as response:
            assert response.status_code == 200
            for line in response.iter_lines():
                if not line.startswith('data: '):
                    continue
                raw = line[6:]
                if raw == '[DONE]':
                    break
                chunks.append(json.loads(raw))
        assert chunks, 'text completion SSE returned no chunks'
        assert any(c.get('choices', [{}])[0].get('text') for c in chunks)
        assert chunks[-1].get('choices', [{}])[0].get('finish_reason') in {'stop', 'length'}
        return {'chunks': len(chunks)}
    check('text_completion_sse', text_completion_sse)

    def prefix():
        body = {'messages': [{'role': 'user', 'content': 'Background: '+('alpha beta gamma delta. '*400)+' Reply exactly OK.'}], 'max_tokens': 16}
        one, two = call(body), call(body)
        cached = two['usage']['prompt_tokens_details']['cached_tokens']
        assert cached > 0
        return {'cached_tokens': cached, 'prompt_tokens': two['usage']['prompt_tokens']}
    check('prefix_cache', prefix)
    check('invalid_tool_choice', lambda: call({'messages': [user], 'tools': [echo],
        'tool_choice': {'type':'function','function':{'name':'missing'}}}, expected=400))
    check('unknown_model', lambda: call({'model': 'missing', 'messages': [user]}, expected=404))

    def context_error():
        oversized = 'context-boundary ' * 100000
        data = call({'messages': [{'role': 'user', 'content': oversized}]}, expected=400)
        assert data.get('error', {}).get('code') == 'context_length_exceeded', data
        return data.get('error')
    check('oversized_context', context_error)

    def cancellation_cleanup():
        body = {'model': a.model, 'messages': [{'role': 'user', 'content': 'stream a long answer'}],
                'enable_thinking': False, 'max_tokens': 1024, 'stream': True}
        with client.stream('POST', '/v1/chat/completions', json=body) as response:
            assert response.status_code == 200
            saw_content = False
            for line in response.iter_lines():
                if not line.startswith('data: ') or line[6:] == '[DONE]':
                    continue
                payload = json.loads(line[6:])
                choices = payload.get('choices') or []
                delta = choices[0].get('delta', {}) if choices else {}
                if delta.get('content') or delta.get('tool_calls'):
                    saw_content = True
                    break
            assert saw_content, 'stream disconnected before a content/tool delta'
        deadline = time.monotonic() + 15.0
        active = None
        props_now = {}
        while time.monotonic() < deadline:
            props_now = client.get('/props').json()
            active = ((props_now.get('runtime') or {}).get('generator_runtime') or {}).get('active_jobs')
            if active == 0:
                break
            time.sleep(0.2)
        assert active == 0, props_now
        assert client.get('/health').status_code == 200
        return {'active_jobs': active}
    check('stream_cancellation_cleanup', cancellation_cleanup)
    evidence['final_runtime'] = client.get('/props').json()['runtime']
    evidence['complete'] = all(t['pass'] for t in evidence['tests'].values())
    save()
    client.close()


if __name__ == '__main__':
    main()
