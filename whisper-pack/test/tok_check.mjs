// Runs every case in test/fixtures/tok.json against web/tokenizer.js. Exit 1 on any mismatch.
import { readFileSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';
import { Tokenizer } from '../web/tokenizer.js';

const here = dirname(fileURLToPath(import.meta.url));
const fx = JSON.parse(readFileSync(join(here, 'fixtures', 'tok.json'), 'utf8'));
let bad = 0;
for (const [model, spec] of Object.entries(fx)) {
  const tok = Tokenizer.fromJSON(JSON.parse(readFileSync(spec.tokenizer_json, 'utf8')));
  let pass = 0;
  const fails = [];
  for (const c of spec.cases) {
    const got = tok.decode(c.ids, { skipSpecial: c.skip_special });
    if (got === c.text) pass++; else fails.push({ got, c });
  }
  // timestamp helper sanity
  const t0 = spec.t0;
  if (tok.timestampSeconds(t0) !== 0 || tok.timestampSeconds(t0 + 1500) !== 30 ||
      tok.timestampSeconds(t0 + 1) !== 0.02 || tok.timestampSeconds(t0 - 1) !== null ||
      tok.timestampSeconds(t0 + 1501) !== null) {
    fails.push({ got: 'timestampSeconds', c: { kind: 'timestamps' } });
  }
  console.log(`${model}: ${pass}/${spec.cases.length} passed`);
  for (const f of fails.slice(0, 3)) {
    console.log('MISMATCH', JSON.stringify({ kind: f.c.kind, skip: f.c.skip_special, ids: f.c.ids,
      expected: f.c.text, got: f.got }, null, 1));
  }
  bad += fails.length;
}
process.exit(bad ? 1 : 0);
