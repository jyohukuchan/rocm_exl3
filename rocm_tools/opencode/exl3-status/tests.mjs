import { test } from 'node:test';
import assert from 'node:assert/strict';
import { observeResponse, formatMetrics, stateFile } from './shared.mjs';
import plugin from './index.mjs';
import { mkdtemp, readFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
test('SSE framing and UTF8 splits preserve native metrics', async () => {
  const rows = [], payload = 'data: {"choices":[{"delta":{"content":"東京"}}]}\r\n\r\ndata: {"exl3_metrics":{"version":1,"cached_tokens":8}}\n\ndata: [DONE]\n\n';
  const bytes = new TextEncoder().encode(payload);
  const response = new Response(new ReadableStream({ start(c) { for (const b of bytes) c.enqueue(Uint8Array.of(b)); c.close(); } }), { headers: { 'content-type': 'text/event-stream' } });
  await observeResponse(response, event => rows.push(event));
  assert.equal(rows[0].choices[0].delta.content, '東京');
  assert.equal(rows[1].exl3_metrics.cached_tokens, 8);
});
test('missing metrics never falls back to input divided by TTFT', () => {
  assert.equal(formatMetrics({ phase: 'complete' }), 'EXL3 timing unavailable');
  assert.match(formatMetrics({ phase: 'complete', metrics: { prefill_tokens_per_second: 100, output_tokens_per_second: 20, cached_tokens: 90 } }), /PP 100.0.*TG 20.0.*cache 90/);
  assert.throws(() => stateFile('../secret'));
});
test('provider hook leaves streaming intact and stores only primary session metrics', async () => {
  const dir = await mkdtemp(join(tmpdir(), 'exl3-status-'));
  const previous = process.env.XDG_STATE_HOME; process.env.XDG_STATE_HOME = dir;
  const hooks = {};
  const dispose = await plugin.setup({ session: { hook: async (name, fn, filter) => {
    assert.equal(filter.providerID, 'rocm-exl3'); hooks[name] = fn;
  } } });
  try {
    const request = new Request('http://localhost/v1/chat/completions');
    const event = { sessionID: 'ses_test', kind: 'primary', request };
    await hooks['http.request'](event);
    const payload = 'data: {"id":"r1","exl3_metrics":{"version":1,"prefill_tokens_per_second":12,"cached_tokens":5}}\n\ndata: [DONE]\n\n';
    const response = new Response(payload, { headers: { 'content-type': 'text/event-stream' } });
    await hooks['http.response']({ ...event, response });
    assert.equal(await response.text(), payload);
    let record;
    for (let i = 0; i < 100; i++) {
      try { record = JSON.parse(await readFile(stateFile('ses_test'), 'utf8')); } catch {}
      if (record?.phase === 'complete') break;
      await new Promise(resolve => setTimeout(resolve, 10));
    }
    assert.equal(record.metrics.prefill_tokens_per_second, 12);
    const auxiliary = { ...event, kind: 'title', request: new Request(request.url) };
    await hooks['http.request'](auxiliary);
    assert.equal(auxiliary.request.headers.get('x-exl3-display-request'), null);
  } finally {
    dispose(); if (previous === undefined) delete process.env.XDG_STATE_HOME;
    else process.env.XDG_STATE_HOME = previous;
    await rm(dir, { recursive: true, force: true });
  }
});
