const results = []; const sleep = (ms, v) => new Promise(r => setTimeout(() => r(v), ms));
async function worker(id, jobs) { for (const j of jobs) { await sleep(j % 3, null); results.push(`w${id}:${j}`); } }
(async () => { await Promise.all([worker(1, [1, 2, 3]), worker(2, [4, 5]), worker(3, [6])]); console.log(results.length, results.sort().join()); const r = await Promise.race([sleep(20, 'slow'), Promise.resolve('fast')]); console.log(r); })();
