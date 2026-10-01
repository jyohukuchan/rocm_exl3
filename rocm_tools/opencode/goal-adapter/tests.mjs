import test from 'node:test';
import assert from 'node:assert/strict';
import { adaptGoal, parseAudit, constrainAudit } from './index.mjs';

const prompt = (token, id) => ({ sessionID: id, metadata: { opencode_goal_v2_verifier: true },
  text: `Verification request:\n${JSON.stringify({auditToken:token,requirements:[{id}]})}\n\nCall opencode_goal_verifier_result exactly once.` });
const schema = {type:'object',properties:{auditToken:{type:'string'},results:{type:'array',items:{type:'object',properties:{requirementID:{type:'string'},reason:{type:'string'}}}}}};

test('only native verifier prompts supply exact audit IDs', () => {
  assert.deepEqual(parseAudit(prompt('a','b')), {token:'a',ids:['b']});
  assert.equal(parseAudit({...prompt('a','b'),metadata:{}}), null);
  assert.equal(parseAudit({...prompt('a','b'),text:'broken'}), null);
});
test('audit schemas constrain IDs without mutating shared schemas or widening tools', () => {
  const a = {tools:{opencode_goal_verifier_result:{input:schema}}};
  const b = {tools:{opencode_goal_verifier_result:{input:schema}}};
  constrainAudit(a, {token:'a',ids:['1']});
  constrainAudit(b, {token:'b',ids:['2']});
  assert.deepEqual(a.tools.opencode_goal_verifier_result.input.properties.results.items.properties.requirementID.enum,['1']);
  assert.deepEqual(b.tools.opencode_goal_verifier_result.input.properties.auditToken.enum,['b']);
  assert.equal(schema.properties.auditToken.enum, undefined);
  const hidden={tools:{read:{input:{}}}};constrainAudit(hidden,{token:'a',ids:['1']});
  assert.deepEqual(Object.keys(hidden.tools),['read']);
});
test('native prompt dispatch and context filtering remain in control', async () => {
  let context, cleaned=false;
  const ctx={session:{prompt:async input=>({id:input.sessionID}),hook:async(name,fn)=>{context=fn;}}};
  const plugin={id:'goal',setup:async ctx=>{
    await ctx.session.prompt(prompt('audit','session'));
    await ctx.session.hook('context',event=>{delete event.tools.shell;});
    return ()=>{cleaned=true;};
  }};
  const cleanup=await adaptGoal(plugin).setup(ctx);
  const event={sessionID:'session',tools:{shell:{},opencode_goal_verifier_result:{input:schema}}};
  await context(event);
  assert.equal(event.tools.shell,undefined);
  assert.deepEqual(event.tools.opencode_goal_verifier_result.input.properties.auditToken.enum,['audit']);
  await cleanup();assert.equal(cleaned,true);
});
