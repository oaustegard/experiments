// node run_wasi_node.mjs MODULE "HOST::GUEST\n..." "K=V\n..." ARGS...
import { readFile } from "node:fs/promises";
import { WASI } from "node:wasi";
import { basename } from "node:path";

const [mod, mapsArg, envArg, ...args] = process.argv.slice(2);
const preopens = {};
for (const m of mapsArg.split("\n").filter(Boolean)) {
  const [host, guest] = m.split("::");
  preopens[guest ?? host] = host;
}
const env = {};
for (const e of envArg.split("\n").filter(Boolean)) {
  const i = e.indexOf("=");
  env[e.slice(0, i)] = e.slice(i + 1);
}
const wasi = new WASI({ version: "preview1", args: [basename(mod), ...args], env, preopens, returnOnExit: true });
const module = await WebAssembly.compile(await readFile(mod));
const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
process.exitCode = wasi.start(instance);
