// node poker-hand-weights/run.mjs [--site PATH] [--jobs 4] [--ref]
// Head-to-head of the hand-wired network against the other computer players, on deals no development run
// used (seed offset 100000). Each pairing is one call of the site's duplicate-match harness,
// tests/poker-bots.mjs: every deal is played twice from the same deck with the seats swapped, so card luck
// cancels, and it reports the network's chips in big blinds per 100 hands with a standard error.
// Results are checkpointed to results/headtohead.json after each pairing; a rerun skips finished ones.
// --ref runs only the reference pairings (the existing players against the same baselines) into
// results/reference.json.
import { spawn } from 'node:child_process';
import { existsSync, readFileSync, writeFileSync } from 'node:fs';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const HERE = dirname(fileURLToPath(import.meta.url));
const arg = (k, d) => { const i = process.argv.indexOf(k); return i > 0 ? process.argv[i + 1] : d; };
// The site checkout: --site, then $POKER_SITE, then $EXPERIMENTS_SPOKES_ROOT/oaustegard.github.io, then a sibling of this repo.
const SITE = resolve(arg('--site', process.env.POKER_SITE
  || [process.env.EXPERIMENTS_SPOKES_ROOT && join(process.env.EXPERIMENTS_SPOKES_ROOT, 'oaustegard.github.io'), join(HERE, '..', '..', 'oaustegard.github.io')]
    .find(p => p && existsSync(join(p, 'tests/poker-bots.mjs')))));
const JOBS = +arg('--jobs', 4), OFFSET = 100000;
const REF = process.argv.includes('--ref');
const OUT = join(HERE, 'results', REF ? 'reference.json' : 'headtohead.json');

const OPP = ['tag', 'v1', 'caller', 'random'];
const PLAN = [];
const add = (game, structure, seats, opps, deals) => opps.forEach(B => PLAN.push({ game, structure, seats, B, deals: typeof deals === 'function' ? deals(B) : deals }));
add('holdem', 'nl', 6, OPP, 3000);
add('holdem', 'nl', 2, ['tag', 'v1', 'caller', 'random'], 3000);
add('holdem', 'fl', 6, OPP, 2000);
add('omaha', 'pl', 6, OPP, B => (B === 'tag' || B === 'v1' ? 600 : 2000));
add('omaha8', 'fl', 6, OPP, B => (B === 'tag' || B === 'v1' ? 600 : 2000));
for (const g of ['stud7', 'stud8', 'razz', 'stud5']) add(g, 'fl', 6, OPP, 2000);
add('draw5', 'fl', 6, OPP, 2000);

// Reference rows: how the existing players do against the same baselines (and v1), for scale.
for (const [g, st, seats, deals] of [['holdem', 'nl', 6, 3000], ['holdem', 'fl', 6, 2000], ['stud7', 'fl', 6, 2000], ['stud8', 'fl', 6, 2000]])
  for (const A of ['tag', 'v1']) for (const B of (A === 'tag' ? ['v1', 'caller', 'random'] : ['caller', 'random'])) PLAN.push({ game: g, structure: st, seats, A, B, deals });
for (const B of ['caller', 'random']) PLAN.push({ game: 'omaha8', structure: 'fl', seats: 6, A: 'tag', B, deals: 600 });
const key = p => `${p.game}-${p.structure}-${p.seats}-${p.A || 'net'}-vs-${p.B}`;
const done = existsSync(OUT) ? JSON.parse(readFileSync(OUT, 'utf8')) : {};
const todo = PLAN.filter(p => !done[key(p)] && !!p.A === REF);
console.log(`site ${SITE}; ${todo.length} of ${PLAN.length} pairings to run`);

function run(p) {
  return new Promise((ok, fail) => {
    const ch = spawn('node', ['tests/poker-bots.mjs', p.deals, p.game, p.structure, p.A || 'net', p.B, p.seats, OFFSET], { cwd: SITE, env: { ...process.env, BOTS_JSON: '1' } });
    let out = '';
    ch.stdout.on('data', d => { out += d; });
    ch.on('close', code => {
      const line = out.split('\n').find(l => l.startsWith('{'));
      if (code || !line) return fail(new Error(`${key(p)} exited ${code}: ${out.slice(-300)}`));
      ok(JSON.parse(line));
    });
  });
}
let next = 0;
async function worker() {
  while (next < todo.length) {
    const p = todo[next++];
    const r = await run(p);
    r.ci95 = +(1.96 * r.se).toFixed(2);
    done[key(p)] = r;
    writeFileSync(OUT, JSON.stringify(done, null, 1) + '\n');
    console.log(`${key(p)}: ${r.bb100} ± ${r.se} bb/100 (${r.deals} deals, ${r.seconds} s)`);
  }
}
await Promise.all(Array.from({ length: JOBS }, worker));
console.log('all pairings done');
