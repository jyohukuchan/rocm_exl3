"""Measure serial System 1 HTTP latency, including the official computer-use demo.

Model loading, browser rendering and PNG encoding are outside request timing.
Image decode/processing, vision inference, language prefill and calibrated
readout are inside it. No prefix/image cache is introduced by this collector.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import math
import os
from pathlib import Path
import statistics
import time
from types import SimpleNamespace


def summarize(values):
    ordered = sorted(values)
    if not ordered:
        raise ValueError('Cannot summarize an empty measurement set')
    return {
        'n': len(ordered), 'mean_ms': statistics.mean(ordered),
        'median_ms': statistics.median(ordered),
        'p95_ms': ordered[max(0, math.ceil(0.95*len(ordered))-1)],
        'min_ms': ordered[0], 'max_ms': ordered[-1],
    }


def text_cases():
    return [
        ('short_noul', {'kind': 'noul', 'state': 'Tokyo is the capital of Japan.',
                        'question': 'Is this correct?', 'thinking': 'off'}),
        ('choice16', {'kind': 'choice', 'state': 'The target identifier is 13.',
                      'question': 'Select the option matching the target identifier.',
                      'options': [str(i) for i in range(16)], 'thinking': 'off'}),
        ('long_text', {'kind': 'noul',
                       'state': ('Routine record: the warehouse received a shipment and recorded its contents.\n'*200)
                                + 'The refund for order 731 was approved.',
                       'question': 'Was the refund for order 731 approved?', 'thinking': 'off'}),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--url', default='http://127.0.0.1:3960')
    ap.add_argument('--label', required=True)
    ap.add_argument('--demo-dir', required=True, help='Official computer_use.py and webapp.html')
    ap.add_argument('--output', required=True)
    ap.add_argument('--text-repeats', type=int, default=30)
    ap.add_argument('--episodes-per-app', type=int, default=20)
    ap.add_argument('--warmup', type=int, default=3)
    args = ap.parse_args()
    if min(args.text_repeats, args.episodes_per_app, args.warmup) < 1:
        ap.error('Repetitions, episodes and warmup must be positive')
    import requests
    url = args.url.rstrip('/')
    info_response = requests.get(url+'/v1/decide/info', timeout=10)
    info_response.raise_for_status()
    report = {
        'label': args.label, 'info': info_response.json(),
        'measurement': {'concurrency': 1, 'thinking': 'off', 'strategy': 'single',
                        'warmup_per_text_case': args.warmup,
                        'browser_warmup_episodes': args.warmup,
                        'model_load_excluded': True, 'prefix_image_reuse': False,
                        'browser_render_and_png_encoding_excluded': True,
                        'p95_method': 'nearest rank', 'clock': 'perf_counter_ns'},
        'client_packages': {name: importlib.metadata.version(name) for name in ('requests', 'Pillow', 'playwright')},
        'environment': {key: os.environ.get(key) for key in ('PYTHONPATH', 'ROCR_VISIBLE_DEVICES', 'HSA_ENABLE_SDMA', 'AMD_SERIALIZE_KERNEL')},
        'text': {}, 'browser': {},
    }
    all_samples = []
    collecting = False
    group = ''

    def post(api, **kwargs):
        nonlocal all_samples
        started = time.perf_counter_ns()
        response = requests.post(api, **kwargs)
        response.raise_for_status()
        out = response.json()
        elapsed_ms = (time.perf_counter_ns()-started)/1e6
        p = out['probabilities']
        if (not all(math.isfinite(v) and 0 <= v <= 1 for v in p)
                or abs(sum(p)-1) > 1e-6):
            raise RuntimeError('Invalid decision probabilities')
        if out['num_model_requests'] != 1 or out['usage']['completion_tokens'] != 0:
            raise RuntimeError('Expected a single System 1 request without generation')
        if collecting:
            body = kwargs['json']
            all_samples.append({'group': group, 'http_ms': elapsed_ms,
                                'server_ms': out['elapsed_seconds']*1000,
                                'prompt_tokens': out['usage']['prompt_tokens'],
                                'options': len(p), 'choice_index': out['choice_index'],
                                'choice': out['choice'], 'max_probability': max(p),
                                'request_sha256': hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()})
        return response

    for name, body in text_cases():
        group = name
        for _ in range(args.warmup):
            post(url+'/v1/decide', json=body, timeout=120)
        collecting = True
        first = len(all_samples)
        for _ in range(args.text_repeats):
            out = post(url+'/v1/decide', json=body, timeout=120).json()
            expected = '13' if name == 'choice16' else 'true'
            if out['choice'] != expected:
                raise RuntimeError(f'{name} selected {out["choice"]}, expected {expected}')
        collecting = False
        samples = all_samples[first:]
        report['text'][name] = {'http': summarize([s['http_ms'] for s in samples]),
                               'server': summarize([s['server_ms'] for s in samples]),
                               'prompt_tokens': samples[0]['prompt_tokens'],
                               'request': body, 'samples': samples}
        print(name, json.dumps(report['text'][name]['http']), flush=True)

    demo_root = Path(args.demo_dir).resolve()
    spec = importlib.util.spec_from_file_location('jev_official_computer_use', demo_root/'computer_use.py')
    demo = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(demo)
    demo.API = url+'/v1/decide'
    # Preserve the official episode, task generation, marks, options and clicks.
    demo.requests = SimpleNamespace(post=post)
    report['demo_source_sha256'] = {name: hashlib.sha256((demo_root/name).read_bytes()).hexdigest()
                                    for name in ('computer_use.py', 'webapp.html')}
    report['browser'].update(viewport=[demo.VW, demo.VH], variant='marks+text', max_steps=demo.MAX_STEPS)
    from playwright.sync_api import sync_playwright
    episodes = []
    first = len(all_samples)
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch()
        report['browser']['chromium_version'] = browser.version
        page = browser.new_page(viewport={'width': demo.VW, 'height': demo.VH})
        apps = ('mail', 'shop', 'settings')
        for i in range(args.warmup):
            group = 'browser_warmup'
            demo.episode(page, apps[i % len(apps)], 100+i, 'marks+text')
        collecting = True
        for app in apps:
            for seed in range(1, args.episodes_per_app+1):
                group = f'{app}/{seed}'
                episode = demo.episode(page, app, seed, 'marks+text')
                episodes.append(episode)
                print('browser', app, seed, episode['success'], episode['steps'], flush=True)
        collecting = False
        browser.close()
    samples = all_samples[first:]
    report['browser'].update(
        episodes=episodes, samples=samples, episode_count=len(episodes),
        success_count=sum(e['success'] for e in episodes),
        http=summarize([s['http_ms'] for s in samples]),
        server=summarize([s['server_ms'] for s in samples]),
        prompt_tokens={'min': min(s['prompt_tokens'] for s in samples),
                       'max': max(s['prompt_tokens'] for s in samples),
                       'median': statistics.median(s['prompt_tokens'] for s in samples)},
    )
    for app in apps:
        subset = [s for s in samples if s['group'].startswith(app+'/')]
        report['browser'][app] = summarize([s['http_ms'] for s in subset])
    report['finished_epoch'] = time.time()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print('BROWSER_SUMMARY', json.dumps(report['browser']['http']),
          'success', report['browser']['success_count'], '/', len(episodes), flush=True)
    print('SAVED', output, flush=True)


if __name__ == '__main__':
    main()
