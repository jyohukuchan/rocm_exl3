import { readFile } from 'node:fs/promises';
import { createSignal, onCleanup } from 'solid-js';
import { createElement, insert } from '@opentui/solid';
import { stateFile, formatMetrics } from './shared.mjs';

export default {
  id: 'rocm-exl3.metrics',
  async setup(ctx) {
    const poll = (props, read, format, initial) => {
        const [record, setRecord] = createSignal(initial);
        let stopped = false, busy = false;
        const refresh = async () => {
          if (busy) return;
          busy = true;
          try {
            const sessionID = props.sessionID;
            const data = sessionID ? await read(sessionID) : initial;
            if (!stopped && props.sessionID === sessionID) setRecord(data);
          } catch { if (!stopped) setRecord(initial); }
          finally { busy = false; }
        };
        const timer = setInterval(refresh, 250);
        void refresh();
        onCleanup(() => { stopped = true; clearInterval(timer); });
        const text = createElement('text');
        insert(text, () => {
          const model = ctx.data.session.get(props.sessionID)?.model;
          return model?.providerID === 'rocm-exl3' ? format(record()) : '';
        });
        return text;
    };
    const disposers = [ctx.ui.slot({ append: 'prompt.footer.status', render: props =>
      poll(props, async id => JSON.parse(await readFile(stateFile(id), 'utf8')), formatMetrics, null) })];
    if (ctx.options.goalFormatter) {
      const { formatGoalSidebar } = await import(ctx.options.goalFormatter);
      disposers.push(ctx.ui.slot({ append: 'sidebar.content', render: props => poll(props,
        async id => {
          const root = ctx.data.session.get(id)?.location?.directory ?? ctx.location?.directory;
          return root ? formatGoalSidebar(root, id) : '';
        }, value => value, '') }));
    }
    return () => disposers.forEach(dispose => dispose());
  },
};
