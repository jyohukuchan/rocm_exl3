import { homedir } from 'node:os';
import { join } from 'node:path';
export const stateDir = () => join(process.env.XDG_STATE_HOME || join(homedir(), '.local', 'state'), 'opencode', 'exl3-metrics');
export function stateFile(sessionID) {
  if (!/^ses_[a-zA-Z0-9]+$/.test(sessionID)) throw new Error('Invalid session ID');
  return join(stateDir(), `${sessionID}.json`);
}
export function formatMetrics(record) {
  if (!record) return 'EXL3 PP —';
  const m = record.metrics;
  const rate = value => typeof value === 'number' && Number.isFinite(value) ? value.toFixed(1) : '—';
  if (record.phase === 'prefill') return 'EXL3 prefill…';
  if (record.phase === 'generating') return 'EXL3 generating…';
  if (!m) return 'EXL3 timing unavailable';
  return `EXL3 PP ${rate(m.prefill_tokens_per_second)} tok/s · TG ${rate(m.output_tokens_per_second)} tok/s · cache ${m.cached_tokens}`;
}
export async function observeResponse(response, onEvent) {
  if (!response.headers.get('content-type')?.includes('text/event-stream')) {
    await onEvent(await response.json());
    return;
  }
  const reader = response.body.getReader(), decoder = new TextDecoder();
  let buffer = '', lines = [];
  const line = async value => {
    value = value.replace(/\r$/, '');
    if (value === '') {
      const data = lines.filter(x => x.startsWith('data:')).map(x => x.slice(5).trimStart()).join('\n');
      lines = [];
      if (data && data !== '[DONE]') await onEvent(JSON.parse(data));
    } else lines.push(value);
  };
  try {
    while (true) {
      const { value, done } = await reader.read();
      buffer += decoder.decode(value, { stream: !done });
      let pos;
      while ((pos = buffer.indexOf('\n')) >= 0) {
        await line(buffer.slice(0, pos)); buffer = buffer.slice(pos + 1);
      }
      if (buffer.length > 4 * 1024 * 1024) throw new Error('Oversized SSE frame');
      if (done) { if (buffer) await line(buffer); await line(''); break; }
    }
  } finally { reader.releaseLock(); }
}
