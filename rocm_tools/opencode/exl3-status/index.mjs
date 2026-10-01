import { mkdir, writeFile, rename, unlink } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';
import { stateDir, stateFile, observeResponse } from './shared.mjs';

export default {
  id: 'rocm-exl3.metrics',
  async setup(ctx) {
    const latest = new Map();
    let closed = false;
    const save = async (sessionID, nonce, record) => {
      if (closed || latest.get(sessionID) !== nonce) return;
      await mkdir(stateDir(), { recursive: true, mode: 0o700 });
      const file = stateFile(sessionID), temp = `${file}.${nonce}.tmp`;
      await writeFile(temp, JSON.stringify({ ...record, nonce, updated: Date.now() }), { mode: 0o600 });
      if (!closed && latest.get(sessionID) === nonce) await rename(temp, file);
      else await unlink(temp).catch(() => {});
    };
    await ctx.session.hook('http.request', event => {
      if (event.kind !== 'primary') return;
      const nonce = randomUUID();
      latest.set(event.sessionID, nonce);
      event.request.headers.set('x-exl3-display-request', nonce);
      void save(event.sessionID, nonce, { phase: 'prefill' }).catch(() => {});
    }, { providerID: 'rocm-exl3' });
    await ctx.session.hook('http.response', event => {
      if (event.kind !== 'primary') return;
      const nonce = event.request.headers.get('x-exl3-display-request');
      if (!nonce) return;
      if (!event.response.ok) {
        void save(event.sessionID, nonce, { phase: 'unavailable' }).catch(() => {});
        return;
      }
      const copy = event.response.clone();
      // Drain the cloned stream in parallel: do not delay the provider's stream.
      void (async () => {
        let generating = false, measured = false;
        await observeResponse(copy, async data => {
          if (data.exl3_metrics?.version === 1) {
            measured = true;
            await save(event.sessionID, nonce, { phase: 'complete', responseID: data.id, metrics: data.exl3_metrics });
          } else if (!generating && data.choices?.some(c => c.delta?.content || c.delta?.reasoning_content || c.delta?.tool_calls)) {
            generating = true;
            await save(event.sessionID, nonce, { phase: 'generating' });
          }
        });
        if (!measured) await save(event.sessionID, nonce, { phase: 'unavailable' });
      })().catch(() => save(event.sessionID, nonce, { phase: 'unavailable' }).catch(() => {}));
    }, { providerID: 'rocm-exl3' });
    return () => { closed = true; latest.clear(); };
  },
};
