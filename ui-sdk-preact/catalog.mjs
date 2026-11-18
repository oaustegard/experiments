#!/usr/bin/env node
// Writes the component catalog for the developing-preact skill: one entry per
// module of web/esm, with the component's one-line purpose from the npm
// package's docs, its parts as exported by the build, and the core
// controller, if any.
//
//   node catalog.mjs <unpacked @ui-sdk/react package> web/esm > catalog.md

import { readFileSync, readdirSync, existsSync } from "node:fs"
import { join } from "node:path"

const [pkg, esm] = process.argv.slice(2)
if (!pkg || !esm) {
  console.error("usage: node catalog.mjs <@ui-sdk/react package dir> <web/esm dir>")
  process.exit(2)
}

const exportsOf = (file) => {
  const code = readFileSync(file, "utf8")
  const names = []
  for (const block of code.matchAll(/export\s*\{([^}]*)\}/g)) {
    for (const item of block[1].split(",")) {
      const name = item.trim().split(/\s+as\s+/).pop().trim()
      if (name) names.push(name)
    }
  }
  return names
}

// The first paragraph after the title: the docs open every page with one sentence on what it is for.
const purposeOf = (doc) => {
  const lines = readFileSync(doc, "utf8").split("\n")
  const start = lines.findIndex((l) => l.startsWith("# ")) + 1
  return lines.slice(start).find((l) => l.trim() && !l.startsWith("<") && !l.startsWith("```"))?.trim() ?? ""
}

const docs = join(pkg, "docs")
const componentDocs = new Set(readdirSync(join(docs, "components")).map((f) => f.replace(/\.md$/, "")))
const modules = readdirSync(esm).filter((f) => f.endsWith(".js")).map((f) => f.replace(/\.js$/, "")).sort()

const out = []
const others = []
for (const name of modules) {
  const parts = exportsOf(join(esm, `${name}.js`))
  if (!componentDocs.has(name)) {
    others.push(`| \`${name}.js\` | ${parts.map((p) => `\`${p}\``).join(", ")} |`)
    continue
  }
  const controller = parts.filter((p) => p.startsWith("create"))
  const components = parts.filter((p) => !p.startsWith("create") && /^[A-Z]/.test(p))
  const helpers = parts.filter((p) => !controller.includes(p) && !components.includes(p))
  out.push(`### ${name}`)
  out.push("")
  out.push(purposeOf(join(docs, "components", `${name}.md`)))
  out.push("")
  out.push(`\`ui-sdk/${name}.js\`: ${components.map((p) => `\`${p}\``).join(", ")}`)
  if (controller.length) out.push(`Controller: ${controller.map((p) => `\`${p}\``).join(", ")}`)
  if (helpers.length) out.push(`Also: ${helpers.map((p) => `\`${p}\``).join(", ")}`)
  out.push("")
}

const core = existsSync(join(esm, "core"))
  ? readdirSync(join(esm, "core")).filter((f) => f.endsWith(".js")).length
  : 0

console.log(`## Components (${out.filter((l) => l.startsWith("### ")).length})

Each entry: what it is for (the first line of its docs page), the module, and its parts. Fetch the
full page with \`scripts/ui_sdk_docs.sh <name>\` before building with a component for the first time.

${out.join("\n")}
## Other modules

Hooks, value helpers and the primitive kit, for building your own parts. ${core} \`@ui-sdk/core\` entry
points are under \`ui-sdk/core/\` (for example \`ui-sdk/core/menu.js\`, \`ui-sdk/core/agent.js\`).

| Module | Exports |
|---|---|
${others.join("\n")}`)
