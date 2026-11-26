// The per-page table of README.md, from test/out/summary.json (run.mjs) and test/out/refdiff.json
// (refdiff.mjs --json test/out/refdiff.json).
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

const out = path.join(path.dirname(fileURLToPath(import.meta.url)), "out");
const { summary, rows } = JSON.parse(fs.readFileSync(path.join(out, "summary.json"), "utf8"));
const rd = fs.existsSync(path.join(out, "refdiff.json")) ? JSON.parse(fs.readFileSync(path.join(out, "refdiff.json"), "utf8")) : null;
const rdBy = new Map((rd?.rows || []).map((r) => [r.id, r]));
const p = (a, b) => (b ? ((100 * a) / b).toFixed(1) : "100.0");
console.log("| page | items (ref/ours) | wasm: found | +merged | ours found | item CER | box | native: found | +merged | native vs wasm: found |");
console.log("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|");
for (const r of rows) {
  const w = r.wasm, n = r.native, d = rdBy.get(r.id);
  const short = r.id.length > 58 ? r.id.slice(0, 55) + "..." : r.id;
  console.log(`| ${short} | ${w.ref_items}/${w.our_items} | ${p(w.ref_found, w.ref_items)} | ${p(w.ref_found_merged, w.ref_items)} | ` +
    `${p(w.ours_found, w.our_items)} | ${((100 * w.item_edits) / Math.max(w.item_chars, 1)).toFixed(2)} | ${p(w.box_ok, w.ref_items)} | ` +
    `${n ? p(n.ref_found, n.ref_items) : "-"} | ${n ? p(n.ref_found_merged, n.ref_items) : "-"} | ${d ? p(d.ref_found, d.ref_items) : "-"} |`);
}
const W = summary.wasm, N = summary.native;
const t = rd?.tot;
console.log(`| **all ${rows.length} pages** | ${W.ref_items}/${W.our_items} | **${W.ref_found_pct.toFixed(2)}** | ${W.ref_found_merged_pct.toFixed(2)} | ` +
  `${W.ours_found_pct.toFixed(2)} | ${W.item_cer_pct.toFixed(3)} | ${W.box_pct.toFixed(2)} | **${N ? N.ref_found_pct.toFixed(2) : "-"}** | ` +
  `${N ? N.ref_found_merged_pct.toFixed(2) : "-"} | ${t ? ((100 * t.ref_found) / t.ref_items).toFixed(2) : "-"} |`);
