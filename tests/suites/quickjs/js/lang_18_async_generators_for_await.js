async function* ag() { for (let i = 0; i < 3; i++) { await new Promise(r => setTimeout(r, 1)); yield i; } }
(async () => {
  for await (const v of ag()) console.log('v', v);
  for await (const v of [Promise.resolve('p1'), 'plain']) console.log(v);
  const it = ag(); console.log(await it.next(), await it.return('early'), await it.next());
  async function thrower() { throw new RangeError('async fail'); } try { await thrower(); } catch (e) { console.log(e.name); }
  const r = await Promise.resolve(1).finally(() => console.log('finally')); console.log(r);
})();
