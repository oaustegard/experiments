// Build artifact/dist/: the Leaf Encoder Lab page for a claude.ai Artifact. Artifact pages cannot
// load scripts from their own files, so the engine (tokenizer.js, leaf.js, leaf-gpu.js) and both
// kernels are inlined into one module script. Weights ship as two files under the 15 MB per-file
// limit, named .wasm only because artifacts serve no generic binary type (the page reads them with
// fetch().arrayBuffer(); they are not WebAssembly); vocab.txt and fixtures.json (queries + edge cases only, no post text) ship beside them.
import { copyFileSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";

const root = new URL("../", import.meta.url);
const read = (p) => readFileSync(new URL(p, root));
// Each file gets its own scope (leaf.js and leaf-gpu.js both declare MAX_TOKENS and parse) and
// hands its exports to the next through the enclosing module scope, in import order.
function wrap(path) {
  const src = read(path).toString();
  const names = [...src.matchAll(/^export (?:async function|function|const|class) (\w+)/gm)].map((m) => m[1]);
  const body = src.replace(/^import .*$/gm, "").replace(/^export (async function|function|const|class) /gm, "$1 ");
  return `// ---- ${path}\nconst { ${names.join(", ")} } = (() => {\n${body}\nreturn { ${names.join(", ")} };\n})();`;
}
const engine = ["web/tokenizer.js", "web/leaf.js", "web/leaf-gpu.js"].map(wrap).join("\n");
const dist = new URL("artifact/dist/", root);
mkdirSync(dist, { recursive: true });
const page = read("artifact/app.html").toString()
  .replace("/*__ENGINE__*/", () => engine)
  .replace("/*__KERNEL_FMA__*/", read("web/kernel-fma.wasm").toString("base64"))
  .replace("/*__KERNEL_PLAIN__*/", read("web/kernel.wasm").toString("base64"))
  .replace("/*__QUERIES__*/", JSON.stringify(JSON.parse(read("web/queries.json")).slice(0, 5)));
writeFileSync(new URL("index.html", dist), page);
const w = read("out/leaf-ir-i8.bin"), half = Math.ceil(w.length / 2);
writeFileSync(new URL("leaf-ir-i8.part0.wasm", dist), w.subarray(0, half));
writeFileSync(new URL("leaf-ir-i8.part1.wasm", dist), w.subarray(half));
copyFileSync(new URL("out/hf/vocab.txt", root), new URL("vocab.txt", dist));
// One edge case contains U+FFFD on purpose (the tokenizer must drop it); the artifact publisher
// rejects that raw character, so write it as the JSON escape �, which parses to the same text.
writeFileSync(new URL("fixtures.json", dist), read("web/fixtures.json").toString().replaceAll("�", "\\ufffd"));
console.log(`index.html ${(page.length / 1024).toFixed(1)} KB, weights ${w.length} bytes in 2 parts`);
