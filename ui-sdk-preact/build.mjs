#!/usr/bin/env node
// Builds a Preact package from a published @ui-sdk/react tarball by pointing
// its React imports at preact/compat. The adapter's code is unchanged; the
// Playwright suite in RESULTS.md is what says that is enough.
//
//   node build.mjs ui-sdk-react-0.1.0-beta.10.tgz out/
//
// Reads the tarball with the system tar (no npm, no install scripts) and
// fails if any React specifier survives the rewrite.

import { execFileSync } from "node:child_process"
import { mkdtempSync, readFileSync, writeFileSync, readdirSync, statSync, mkdirSync, cpSync, rmSync } from "node:fs"
import { join } from "node:path"
import { tmpdir } from "node:os"

const [tarball, out] = process.argv.slice(2)
if (!tarball || !out) {
  console.error("usage: node build.mjs <ui-sdk-react-X.tgz> <out-dir>")
  process.exit(2)
}

// Every React module the adapter imports, and where it goes in Preact. The
// code goes through src/compat.js and src/jsx-runtime.js, which load
// preact/compat (className, onChange as input, onFocus as focusin and so on)
// plus src/fixes.js. The types point at preact/compat directly.
const codeTargets = {
  react: "./_preact/compat.js",
  "react-dom": "./_preact/compat.js",
  "react/jsx-runtime": "./_preact/jsx-runtime.js",
}
const typeTargets = {
  react: "preact/compat",
  "react-dom": "preact/compat",
  "react/jsx-runtime": "preact/compat/jsx-runtime",
}
const specifier = /(\bfrom\s*|\bimport\s*\(\s*|\bimport\s+)(["'])(react(?:-dom)?(?:\/[a-z-]+)?)\2/g

const stage = mkdtempSync(join(tmpdir(), "ui-sdk-preact-"))
execFileSync("tar", ["xzf", tarball, "-C", stage])
const pkgDir = join(stage, "package")
const pkg = JSON.parse(readFileSync(join(pkgDir, "package.json"), "utf8"))
if (pkg.name !== "@ui-sdk/react") {
  console.error(`expected @ui-sdk/react, got ${pkg.name}`)
  process.exit(1)
}

rmSync(out, { recursive: true, force: true })
mkdirSync(out, { recursive: true })
cpSync(join(pkgDir, "dist"), join(out, "dist"), { recursive: true })
cpSync(join(pkgDir, "LICENSE"), join(out, "LICENSE"))

const counts = {}
const leftovers = []
const walk = (dir) => {
  for (const name of readdirSync(dir)) {
    const path = join(dir, name)
    if (statSync(path).isDirectory()) {
      walk(path)
      continue
    }
    if (!/\.(m?js|d\.ts)$/.test(name)) {
      continue
    }
    const text = readFileSync(path, "utf8")
    const targets = name.endsWith(".d.ts") ? typeTargets : codeTargets
    const next = text.replace(specifier, (match, lead, quote, from) => {
      const to = targets[from]
      if (!to) {
        leftovers.push(`${path}: ${from}`)
        return match
      }
      counts[from] = (counts[from] ?? 0) + 1
      return `${lead}${quote}${to}${quote}`
    })
    if (next !== text) {
      writeFileSync(path, next)
    }
  }
}
walk(join(out, "dist"))
const here = new URL(".", import.meta.url).pathname
cpSync(join(here, "src"), join(out, "dist", "_preact"), { recursive: true })

if (leftovers.length) {
  console.error(`unmapped React specifiers:\n${leftovers.join("\n")}`)
  process.exit(1)
}

const { peerDependencies = {}, peerDependenciesMeta, ...rest } = pkg
delete rest.devDependencies
delete rest.scripts
const out_pkg = {
  ...rest,
  name: "ui-sdk-preact",
  description: `@ui-sdk/react ${pkg.version} built for Preact through preact/compat. Unofficial.`,
  private: true,
  keywords: [...(pkg.keywords ?? []).filter((k) => k !== "react"), "preact"],
  // fixes.js installs Preact option hooks on import; a bundler told the
  // package is side-effect free would drop it.
  sideEffects: ["./dist/_preact/*.js"],
  peerDependencies: {
    "@ui-sdk/core": peerDependencies["@ui-sdk/core"],
    preact: ">=11",
  },
}
delete out_pkg.repository
delete out_pkg.bugs
delete out_pkg.homepage
writeFileSync(join(out, "package.json"), JSON.stringify(out_pkg, null, 2) + "\n")
rmSync(stage, { recursive: true, force: true })

console.log(`ui-sdk-preact ${pkg.version} -> ${out}`)
console.log(Object.entries(counts).map(([k, v]) => `  ${k}: ${v}`).join("\n"))
