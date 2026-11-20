// Recomputes the layout in JavaScript (web/layout.js) from the probability maps written by
// run/check_layout.py, and compares every field with the Python result in results/layout_check.json.
// Exit 0 when all pages are identical. Adds js_equal per page and js_all_equal to the results file.
//
// Usage: node run/check_layout.mjs

import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const TL = require('../web/layout.js');

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const RESULTS = join(ROOT, 'results', 'layout_check.json');
const CASES = join(ROOT, 'data', 'layout_cases');

const report = JSON.parse(readFileSync(RESULTS, 'utf8'));

// First place where two parsed values differ, as a path, or null when equal.
function firstDiff(a, b, path = '') {
  if (typeof a !== typeof b) return path || '(root)';
  if (a === null || b === null || typeof a !== 'object') return a === b ? null : (path || '(root)');
  if (Array.isArray(a) !== Array.isArray(b)) return path;
  const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
  for (const k of keys) {
    const d = firstDiff(a[k], b[k], `${path}.${k}`);
    if (d !== null) return d;
  }
  return null;
}

let allEqual = true;
for (const [name, page] of Object.entries(report.pages)) {
  const meta = JSON.parse(readFileSync(join(CASES, `${name}.json`), 'utf8'));
  const raw = readFileSync(join(CASES, `${name}.f32`));
  const prob = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  if (prob.length !== meta.H * meta.W) throw new Error(`${name}: map size ${prob.length} != ${meta.H}x${meta.W}`);

  const js = TL.layout(prob, meta.H, meta.W, meta.scale_x, meta.scale_y, meta.page_w, meta.page_h, report.params);
  // Compare the parsed JSON so both sides go through the same number formatting.
  const jsParsed = JSON.parse(JSON.stringify(js));
  const diff = firstDiff(jsParsed, page.result);
  page.js_equal = diff === null;
  page.js_first_diff = diff;
  if (diff !== null) allEqual = false;
  const boxes = js.lines.length;
  console.log(`${name.padEnd(14)} lines ${String(boxes).padStart(3)}  order ${js.order.length}  ` +
    `blocks ${js.blocks.length}  ${diff === null ? 'JS == Python' : 'DIFFERS at ' + diff}`);
}

report.js_all_equal = allEqual;
writeFileSync(RESULTS, JSON.stringify(report, null, 1) + '\n');
console.log(allEqual ? 'all pages identical' : 'MISMATCH');
process.exit(allEqual ? 0 : 1);
