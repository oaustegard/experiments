#!/usr/bin/env node
// Bundles the built package and @ui-sdk/core into code-split ES modules for
// pages with no build step: one file per component (web/esm/dialog.js), one per
// core entry point (web/esm/core/menu.js), shared chunks, and Preact left to the
// page's import map.
//
//   node esm.mjs <node_modules with ui-sdk-preact, @ui-sdk/core, rolldown> web/esm
//
// Minified, so a page loads what it uses and little else.

import { readFileSync, rmSync } from "node:fs"
import { join, dirname } from "node:path"
import { pathToFileURL } from "node:url"
import { createRequire } from "node:module"

const [modules, out] = process.argv.slice(2)
if (!modules || !out) {
  console.error("usage: node esm.mjs <node_modules> <out-dir>")
  process.exit(2)
}
const exportsOf = (name) =>
  Object.keys(JSON.parse(readFileSync(join(modules, name, "package.json"), "utf8")).exports)
    .filter((key) => key !== "./package.json" && !key.includes("*") && !key.endsWith(".css"))
    .map((key) => key.slice(2) || "index")

const input = Object.fromEntries([
  ...exportsOf("ui-sdk-preact").map((sub) => [sub, `ui-sdk-preact/${sub}`]),
  ...exportsOf("@ui-sdk/core").map((sub) => [
    `core/${sub}`,
    sub === "index" ? "@ui-sdk/core" : `@ui-sdk/core/${sub}`,
  ]),
])

// Rollup-style chunking: a component's chunks are the modules it uses, not
// every module a core barrel re-exports (esbuild's file-level split pulled
// 248 KB into a button).
const { rolldown } = await import(pathToFileURL(createRequire(join(modules, "x.js")).resolve("rolldown")).href)
rmSync(out, { recursive: true, force: true })
const bundle = await rolldown({
  input,
  cwd: dirname(modules),
  platform: "browser",
  // The packages may be linked in; resolve their imports from `modules`.
  resolve: { symlinks: false },
  external: [/^preact(\/|$)/],
  transform: { define: { "process.env.NODE_ENV": '"production"' } },
  logLevel: "warn",
  onLog: (level, log) => {
    // An import rolldown can't find would ship as a bare specifier no page resolves.
    if (log.code === "UNRESOLVED_IMPORT") {
      throw new Error(log.message)
    }
    if (log.code !== "MODULE_LEVEL_DIRECTIVE") {
      console.warn(level, log.code, log.message.split("\n")[0])
    }
  },
})
const result = await bundle.write({
  dir: out,
  format: "es",
  minify: true,
  entryFileNames: "[name].js",
  chunkFileNames: "chunks/[hash].js",
})
await bundle.close()

const files = result.output.filter((o) => o.type === "chunk")
const total = files.reduce((sum, o) => sum + o.code.length, 0)
console.log(`${Object.keys(input).length} entry points, ${files.length} files, ${(total / 1024).toFixed(0)} KB`)
