import { readFile } from 'node:fs/promises';
import { createSignal, onCleanup } from 'solid-js';
import { createElement, insert } from '@opentui/solid';
import { stateFile, formatMetrics } from './shared.mjs';

export default {
  id: 'rocm-exl3.metrics',
  setup(ctx) {
    return ctx.ui.slot({
      append: 'prompt.footer.status',
      render: props => {
        const [record, setRecord] = createSignal(null);
        let stopped = false, busy = false;
        const refresh = async () => {
          if (busy) return;
          busy = true;
          try {
            const sessionID = props.sessionID;
            const data = sessionID ? JSON.parse(await readFile(stateFile(sessionID), 'utf8')) : null;
            if (!stopped && props.sessionID === sessionID) setRecord(data);
          } catch { if (!stopped) setRecord(null); }
          finally { busy = false; }
        };
        const timer = setInterval(refresh, 250);
        void refresh();
        onCleanup(() => { stopped = true; clearInterval(timer); });
        const text = createElement('text');
        insert(text, () => formatMetrics(record()));
        return text;
      },
    });
  },
};
