// Recomputes both ordering modes in JavaScript (web/layout.js) from the probability maps written by
// run/check_order.py, and compares every field with the Python results in results/order_check.json.
// Exit 0 when every page and mode is identical. Adds js_equal per page and mode, and js_all_equal.
//
// Usage: node run/check_order.mjs

import { readFileSync, writeFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const TL = require('../web/layout.js');

const ROOT = join(dirname(fileURLToPath(import.meta.url)), '..');
const RESULTS = join(ROOT, 'results', 'order_check.json');

const report = JSON.parse(readFileSync(RESULTS, 'utf8'));
const MODES = report.modes ? Object.keys(report.modes) : ['xycut', 'bands'];

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
  const dir = join(ROOT, 'data', page.case_dir);
  const meta = JSON.parse(readFileSync(join(dir, `${name}.json`), 'utf8'));
  const raw = readFileSync(join(dir, `${name}.f32`));
  const prob = new Float32Array(raw.buffer.slice(raw.byteOffset, raw.byteOffset + raw.byteLength));
  if (prob.length !== meta.H * meta.W) throw new Error(`${name}: map size ${prob.length} != ${meta.H}x${meta.W}`);

  page.js_equal = {};
  const parts = [];
  for (const mode of MODES) {
    const js = TL.layout(prob, meta.H, meta.W, meta.scale_x, meta.scale_y, meta.page_w, meta.page_h, { order: mode });
    // Compare parsed JSON so both sides go through the same number formatting.
    const diff = firstDiff(JSON.parse(JSON.stringify(js)), page.layouts[mode]);
    page.js_equal[mode] = diff === null;
    if (diff !== null) {
      page.js_first_diff = page.js_first_diff || {};
      page.js_first_diff[mode] = diff;
      allEqual = false;
    }
    parts.push(`${mode} ${diff === null ? 'JS==Py' : 'DIFFERS at ' + diff}`);
  }
  console.log(`${name.padEnd(15)} ${parts.join('   ')}`);
}

report.js_all_equal = allEqual;
writeFileSync(RESULTS, JSON.stringify(report, null, 1) + '\n');
console.log(allEqual ? 'all pages and modes identical' : 'MISMATCH');
process.exit(allEqual ? 0 : 1);
