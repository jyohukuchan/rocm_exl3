const resultTool = 'opencode_goal_verifier_result';

export function parseAudit(input) {
  if (input?.metadata?.opencode_goal_v2_verifier !== true || typeof input.text !== 'string') return null;
  const start = input.text.lastIndexOf('Verification request:\n');
  const end = input.text.indexOf('\n\nCall opencode_goal_verifier_result', start);
  if (start < 0 || end < 0) return null;
  try {
    const audit = JSON.parse(input.text.slice(start + 'Verification request:\n'.length, end));
    if (typeof audit.auditToken !== 'string' || !audit.auditToken || !Array.isArray(audit.requirements)) return null;
    const ids = audit.requirements.map(item => item.id);
    if (!ids.length || ids.some(id => typeof id !== 'string' || !id) || new Set(ids).size !== ids.length) return null;
    return { token: audit.auditToken, ids };
  } catch { return null; }
}

export function constrainAudit(event, audit) {
  const tool = event.tools?.[resultTool];
  const props = tool?.input?.properties;
  const requirement = props?.results?.items?.properties?.requirementID;
  if (!audit || !props?.auditToken || !requirement) return;
  // Clone this session's schema: concurrent audits retain separate IDs. Only
  // restrict values; the Goal plugin still validates authority and evidence.
  const input = structuredClone(tool.input);
  input.properties.auditToken.enum = [audit.token];
  input.properties.results.items.properties.requirementID.enum = [...audit.ids];
  event.tools[resultTool] = { ...tool, input };
}

export function adaptGoal(plugin) {
  return { ...plugin, async setup(ctx) {
    const audits = new Map();
    const session = {
      ...ctx.session,
      async prompt(input) {
        const audit = parseAudit(input);
        if (audit && input.sessionID) {
          audits.delete(input.sessionID);
          audits.set(input.sessionID, audit);
          // Completed verifier sessions are retained by some hosts.
          while (audits.size > 256) audits.delete(audits.keys().next().value);
        }
        return ctx.session.prompt(input);
      },
      hook(name, callback, ...options) {
        if (name !== 'context') return ctx.session.hook(name, callback, ...options);
        return ctx.session.hook(name, async event => {
          await callback(event);
          constrainAudit(event, audits.get(event.sessionID));
        }, ...options);
      },
    };
    const cleanup = await plugin.setup({ ...ctx, session });
    return async () => { audits.clear(); if (typeof cleanup === 'function') await cleanup(); };
  }};
}
