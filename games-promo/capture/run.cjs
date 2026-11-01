const h = require('./lib.cjs');
(async () => {
  const b = await h.launch();
  for (const g of process.argv.slice(2)) {
    const t0 = Date.now();
    try { await require(`./games/${g}.cjs`)(b, h); } catch (e) { console.log(g, 'FAILED', e.message); }
    console.log(g, ((Date.now() - t0) / 1000).toFixed(0) + 's');
  }
  await b.close();
})();
