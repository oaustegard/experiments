const d = (v, ms, fail) => new Promise((res, rej) => setTimeout(() => fail ? rej(new Error(v)) : res(v), ms));
(async () => {
  console.log(JSON.stringify(await Promise.all([d('a', 3), d('b', 1), 5])));
  console.log(await Promise.race([d('slow', 5), Promise.resolve('fast')]));
  console.log(await Promise.allSettled([d('ok', 1), d('bad', 1, true)]).then(r => r.map(x => x.status + ':' + (x.value || x.reason.message))));
  console.log(await Promise.any([d('x', 1, true), d('y', 2)]));
  try { await Promise.any([d('x', 1, true)]); } catch (e) { console.log(e.constructor.name, e.errors.length); }
  try { await Promise.all([d('z', 1, true), d('w', 2)]); } catch (e) { console.log('caught', e.message); }
  const { promise, resolve } = Promise.withResolvers ? Promise.withResolvers() : { promise: Promise.resolve('wr'), resolve() {} }; resolve('wr'); console.log(await promise);
})();
