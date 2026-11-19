// Check web/traineddata.js against run/pack.py: for each model, pack it with the Python
// reference into a temp dir, parse the same file in JS, and compare the manifest (every field
// except source/variant) and the weights bytes. Exit 1 on any difference.
//
//   node run/check_traineddata.mjs

import { execFileSync } from "node:child_process";
import { mkdtempSync, readFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { isDeepStrictEqual } from "node:util";
import { parseTraineddata } from "../web/traineddata.js";

const ROOT = join(dirname(fileURLToPath(import.meta.url)), "..");
const MODELS = ["best/eng", "best/fra", "best/grc", "fast/eng", "fast/fra", "fast/grc"];
const SKIP = new Set(["source", "variant"]);

// first differing path between two values, or null
function firstDiff(a, b, path = "") {
  if (path === "") {
    a = Object.fromEntries(Object.entries(a).filter(([k]) => !SKIP.has(k)));
    b = Object.fromEntries(Object.entries(b).filter(([k]) => !SKIP.has(k)));
  }
  if (isDeepStrictEqual(a, b)) return null;
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return `${path}: length ${a.length} vs ${b.length}`;
    for (let i = 0; i < a.length; i++) {
      const d = firstDiff(a[i], b[i], `${path}[${i}]`);
      if (d) return d;
    }
    return `${path}: arrays differ`;
  }
  if (a && b && typeof a === "object" && typeof b === "object") {
    const keys = new Set([...Object.keys(a), ...Object.keys(b)]);
    for (const k of keys) {
      const d = firstDiff(a[k], b[k], path ? `${path}.${k}` : k);
      if (d) return d;
    }
    return `${path}: objects differ`;
  }
  return `${path}: ${JSON.stringify(a)} vs ${JSON.stringify(b)}`;
}

const tmp = mkdtempSync(join(tmpdir(), "tessgpu-check-"));
let failed = false;
try {
  for (const rel of MODELS) {
    const model = join(ROOT, "models", rel + ".traineddata");
    const outDir = join(tmp, rel.replace("/", "-"));
    execFileSync("python3", [join(ROOT, "run", "pack.py"), model, outDir], { stdio: "pipe" });
    const refMan = JSON.parse(readFileSync(join(outDir, "manifest.json"), "utf8"));
    const refBin = readFileSync(join(outDir, "weights.bin"));

    const bytes = readFileSync(model);
    const ab = bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength);
    const t0 = performance.now();
    const { man, bin } = parseTraineddata(ab, { name: rel });
    const ms = performance.now() - t0;

    const manDiff = firstDiff(man, refMan);
    const got = Buffer.from(bin);
    const binOk = got.length === refBin.length && got.equals(refBin);
    const ok = !manDiff && binOk;
    if (!ok) failed = true;
    const detail = manDiff ? ` manifest ${manDiff}` : "";
    const binDetail = binOk ? "" : ` weights.bin ${got.length} vs ${refBin.length} bytes or content differs`;
    console.log(`${ok ? "OK  " : "FAIL"} ${rel.padEnd(10)} ${refMan.weights.padEnd(11)} ` +
      `${refBin.length} bytes  parse ${ms.toFixed(1)} ms${detail}${binDetail}`);
  }
} finally {
  rmSync(tmp, { recursive: true, force: true });
}
process.exit(failed ? 1 : 0);
