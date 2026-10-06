import argparse
import json
import pathlib
import time
import urllib.request

parser = argparse.ArgumentParser()
parser.add_argument('--output', required=True)
args = parser.parse_args()
base = 'http://127.0.0.1:3960'
results = []

def call(path, body=None):
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(base + path, data=data, headers={'Content-Type': 'application/json'})
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=180) as response:
        result = json.load(response)
        results.append({'path': path, 'status': response.status, 'seconds': time.monotonic()-started, 'response': result})
    return result

def normalized(probabilities):
    assert all(0 <= p <= 1 for p in probabilities)
    assert abs(sum(probabilities) - 1) < 1e-6

assert call('/health')['status'] == 'ok'
assert call('/v1/decide/info')['max_options'] == 256
out = call('/v1/decide', {'kind': 'noul', 'state': 'Tokyo is the capital of Japan.', 'question': 'Is this correct?'})
assert out['choice'] == 'true'
normalized(out['probabilities'])
out = call('/v1/systemone', {'state': '2 + 2 = 4.', 'questions': {
    'correct': {'type': 'noul', 'instructions': 'Is this mathematically correct?'},
    'grade': {'type': 'score', 'instructions': 'Rate the correctness of the answer.', 'criteria': ['entirely wrong', 'poor', 'weak', 'acceptable', 'good', 'entirely correct']},
    'sum': {'type': 'choice', 'instructions': 'What is two plus two?', 'criteria': {'three': None, 'four': None, 'five': None}},
}})
assert out['answers']['correct']['noul'] > 0.5
assert out['answers']['sum']['choice'] == 'four'
grades = out['answers']['grade']['probabilities']
assert max(grades, key=grades.get) == '5'
normalized(list(grades.values()))
cases = json.loads(pathlib.Path('/work/runs/jev-20261006/cases.json').read_text())
image_case = next(c for c in cases['cases'] if c['id'] == 'vision-red-square') if isinstance(cases, dict) else next(c for c in cases if c['id'] == 'vision-red-square')
state = image_case['state']
image_part = next(p for p in state if isinstance(p, dict))
url = image_part.get('image') or image_part['image_url']['url']
out = call('/v1/chat/completions', {'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': 'Name the color and shape. Answer briefly.'}, {'type': 'image_url', 'image_url': {'url': url}}]}], 'max_tokens': 24, 'temperature': 0})
text = out['choices'][0]['message']['content'].lower()
assert 'red' in text and 'square' in text, text
out = call('/v1/decide', {'kind': 'noul', 'state': '2 + 2 = 4.', 'question': 'Is this correct?', 'thinking': 'on', 'think_budget': 8, 'reasoning_effort': 'low', 'return_reasoning': True, 'debug': True})
assert out['thinking']['used'] and out['thinking']['think_tokens'] <= 8
normalized(out['probabilities'])
s1, s2 = out['thinking']['system1'], out['thinking']['system2']
assert all(abs(p-(a+b)*0.5) < 1e-6 for p,a,b in zip(out['probabilities'],s1,s2))
pathlib.Path(args.output).write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n')
print('HTTP_ALL_PASS', len(results))
