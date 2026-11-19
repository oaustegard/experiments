#!/usr/bin/env node
// Builds the ui-sdk on Preact showcase: every upstream React example, compiled
// once and run unchanged on Preact, behind a navigation shell made of ui-sdk
// itself.
//
//   node build.mjs --upstream <dir> --esm <web/esm> --vendor <web/vendor> \
//     --modules <node_modules with rolldown, lucide-react> --out <dir>
//
// <upstream> holds components/, shared/, styles/ and docs/ (the React .mdx
// pages) from vercel-labs/ui at the tag the ESM build was made from, so the
// examples and the library are the same release.
//
// Every @ui-sdk import becomes a bare `ui-sdk/<name>.js` import that the page's
// import map resolves to the ESM build, so the shell and all examples share
// one instance of ui-sdk. React resolves to preact/compat.

import { readFileSync, readdirSync, writeFileSync, mkdirSync, rmSync, cpSync, existsSync } from "node:fs"
import { join, dirname, basename } from "node:path"
import { createRequire } from "node:module"
import { pathToFileURL, fileURLToPath } from "node:url"

const args = Object.fromEntries(
  process.argv.slice(2).reduce((pairs, arg, i, all) => {
    if (arg.startsWith("--")) pairs.push([arg.slice(2), all[i + 1]])
    return pairs
  }, [])
)
for (const key of ["upstream", "esm", "vendor", "modules", "out"]) {
  if (!args[key]) {
    console.error(`missing --${key}`)
    process.exit(2)
  }
}
const here = dirname(fileURLToPath(import.meta.url))
const up = args.upstream

// Sidebar and menu groups. Every docs page must land in exactly one.
const CATEGORIES = [
  ["Overlays", ["dialog", "alert-dialog", "drawer", "popover", "hover-card", "tooltip", "toast"]],
  ["Menus and commands", ["menu", "menubar", "trigger-menu", "command", "navigation-menu", "toolbar"]],
  ["Pickers", ["select", "combobox", "autocomplete", "listbox", "token-input", "color-picker"]],
  ["Choices", ["checkbox", "radio", "switch", "toggle", "segmented-control", "slider", "rating"]],
  ["Text and forms", ["button", "input", "textarea", "number-input", "password-input", "otp-input", "field", "label", "file-input"]],
  ["Dates and time", ["calendar", "date-input", "time-input", "date-picker", "timeline"]],
  ["Layout", ["accordion", "collapsible", "tabs", "pagination", "carousel", "resizable", "scroll-area", "separator"]],
  ["Data and media", ["tree", "waterfall", "avatar", "meter", "progress", "image-cropper", "preferences"]],
]

// Inline markdown to the small subset the shell renders: `code` stays, links keep their text.
const plain = (md) =>
  md
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/\*\*([^*]+)\*\*/g, "$1")
    .replace(/\s+/g, " ")
    .trim()

function parseDoc(name) {
  const text = readFileSync(join(up, "docs", `${name}.mdx`), "utf8")
  const fm = text.match(/^---\n([\s\S]*?)\n---\n/)
  const meta = {}
  for (const line of fm[1].split("\n")) {
    const m = line.match(/^(\w+):\s*(.*)$/)
    if (m) meta[m[1]] = m[2].replace(/^"(.*)"$/, "$1")
  }
  const body = text.slice(fm[0].length)
  const sections = [{ heading: null, text: "", examples: [] }]
  let fence = false
  let paragraph = []
  const flush = () => {
    const current = sections[sections.length - 1]
    if (paragraph.length && !current.text) current.text = plain(paragraph.join(" "))
    paragraph = []
  }
  const lines = body.split("\n")
  for (let i = 0; i < lines.length; i++) {
    const line = lines[i]
    if (line.startsWith("```")) {
      fence = !fence
      flush()
      continue
    }
    if (fence) continue
    if (line.startsWith("## ")) {
      flush()
      sections.push({ heading: line.slice(3).trim(), text: "", examples: [] })
      continue
    }
    if (line.startsWith("<ComponentExample")) {
      flush()
      let tag = line
      while (!tag.includes("/>") && i + 1 < lines.length) tag += " " + lines[++i]
      const exampleName = tag.match(/name="([^"]+)"/)?.[1]
      const caption = tag.match(/caption="([^"]*)"/)?.[1] ?? ""
      if (exampleName) sections[sections.length - 1].examples.push({ id: exampleName, caption: plain(caption) })
      continue
    }
    if (!line.trim() || /^[<|#>!-]|^\s*[-*] |^\d+\. /.test(line)) {
      flush()
      continue
    }
    paragraph.push(line)
  }
  flush()
  return { meta, sections }
}

const docNames = readdirSync(join(up, "docs")).map((f) => f.replace(/\.mdx$/, "")).filter((n) => n !== "index")
const categorized = CATEGORIES.flatMap(([, names]) => names)
const missing = docNames.filter((n) => !categorized.includes(n))
const extra = categorized.filter((n) => !docNames.includes(n))
if (missing.length || extra.length) {
  console.error(`categories out of step with docs: missing ${missing}, unknown ${extra}`)
  process.exit(1)
}

// The avatar examples load a portrait from github.com; the page ships its own.
const localize = (source) => source.replaceAll("https://github.com/shadcn.png", "./avatar.svg")

const exampleFile = (id) => join(up, "components", ...id.split("/").slice(0, 1), "react", `${id.split("/")[1]}.tsx`)

const pages = {}
let exampleCount = 0
for (const name of categorized) {
  const { meta, sections } = parseDoc(name)
  const used = new Set()
  for (const section of sections) {
    section.examples = section.examples.filter((e) => existsSync(exampleFile(e.id)))
    for (const e of section.examples) used.add(e.id)
  }
  const dir = join(up, "components", name, "react")
  const rest = existsSync(dir)
    ? readdirSync(dir).filter((f) => f.endsWith(".tsx")).map((f) => `${name}/${f.slice(0, -4)}`).filter((id) => !used.has(id))
    : []
  if (rest.length) sections.push({ heading: "More examples", text: "", examples: rest.map((id) => ({ id, caption: "" })) })
  const kept = sections.filter((s) => s.examples.length || (s.heading === null && s.text))
  exampleCount += kept.reduce((n, s) => n + s.examples.length, 0)
  pages[name] = { name, title: meta.title, description: meta.description ?? "", sections: kept }
}

const VIRTUAL = "\0showcase:"
const pageModule = (name) => {
  const page = pages[name]
  const imports = []
  const sections = page.sections.map((s) => {
    const examples = s.examples.map((e) => {
      const local = `e${imports.length}`
      const file = exampleFile(e.id)
      imports.push(`import { Example as ${local} } from ${JSON.stringify(file)}`)
      const source = localize(readFileSync(file, "utf8"))
      return `{ id: ${JSON.stringify(e.id)}, caption: ${JSON.stringify(e.caption)}, Example: ${local}, source: ${JSON.stringify(source)} }`
    })
    return `{ heading: ${JSON.stringify(s.heading)}, text: ${JSON.stringify(s.text)}, examples: [${examples.join(", ")}] }`
  })
  return `${imports.join("\n")}
export default { name: ${JSON.stringify(name)}, title: ${JSON.stringify(page.title)}, description: ${JSON.stringify(page.description)}, sections: [${sections.join(", ")}] }`
}

const catalogModule = () => {
  const entries = CATEGORIES.map(
    ([category, names]) =>
      `{ category: ${JSON.stringify(category)}, items: [${names
        .map(
          (n) =>
            `{ name: ${JSON.stringify(n)}, title: ${JSON.stringify(pages[n].title)}, description: ${JSON.stringify(pages[n].description)}, count: ${pages[n].sections.reduce((k, s) => k + s.examples.length, 0)}, load: () => import(${JSON.stringify(VIRTUAL + "page:" + n)}) }`
        )
        .join(", ")}] }`
  )
  return `export default [${entries.join(", ")}]`
}

const require = createRequire(join(args.modules, "x.js"))
const { rolldown } = await import(pathToFileURL(require.resolve("rolldown")).href)

const external = (id) => {
  let m
  if ((m = id.match(/^@ui-sdk\/react\/(.+)$/))) return `ui-sdk/${m[1]}.js`
  if (id === "@ui-sdk/core") return "ui-sdk/core/index.js"
  if ((m = id.match(/^@ui-sdk\/core\/(.+)$/))) return `ui-sdk/core/${m[1]}.js`
  if (/^react\/jsx(-dev)?-runtime$/.test(id)) return "preact/compat/jsx-runtime"
  if (id === "react" || id === "react-dom") return "preact/compat"
  if (/^preact(\/|$)/.test(id)) return id
  if (/^(react-dom\/|ui-sdk\/)/.test(id)) throw new Error(`unmapped import ${id}`)
  return null
}

const out = args.out
rmSync(out, { recursive: true, force: true })
mkdirSync(out, { recursive: true })

const input = { app: join(here, "app.jsx"), "examples/log": join(up, "shared", "react", "log.ts") }
for (const name of categorized) input[`examples/${name}`] = VIRTUAL + "page:" + name

const bundle = await rolldown({
  input,
  cwd: args.modules.replace(/\/node_modules\/?$/, ""),
  platform: "browser",
  // The sources live outside any node_modules tree; resolve packages from --modules.
  resolve: { symlinks: false, modules: [args.modules, "node_modules"] },
  transform: {
    jsx: { runtime: "automatic", importSource: "react" },
    define: { "process.env.NODE_ENV": '"production"' },
  },
  plugins: [
    {
      name: "showcase",
      resolveId(id) {
        if (id.startsWith(VIRTUAL)) return id
        if (id === "showcase:catalog") return VIRTUAL + "catalog"
        if (id === "showcase:log") return join(up, "shared", "react", "log.ts")
        const mapped = external(id)
        return mapped ? { id: mapped, external: true } : null
      },
      load(id) {
        if (id === VIRTUAL + "catalog") return catalogModule()
        if (id.startsWith(VIRTUAL + "page:")) return pageModule(id.slice((VIRTUAL + "page:").length))
        return null
      },
      transform(code, id) {
        if (id.startsWith(join(up, "components"))) return localize(code)
        return null
      },
    },
  ],
  onLog: (level, log) => {
    if (log.code === "UNRESOLVED_IMPORT") throw new Error(log.message)
    if (log.code !== "MODULE_LEVEL_DIRECTIVE") console.warn(level, log.code, log.message.split("\n")[0])
  },
})
const result = await bundle.write({
  dir: out,
  format: "es",
  minify: true,
  entryFileNames: "[name].js",
  chunkFileNames: "chunks/c-[hash].js",
})
await bundle.close()

// Styles: upstream's tokens, base and per-component CSS, then this page's theme on top.
const css = [
  readFileSync(join(up, "styles", "tokens.css"), "utf8"),
  readFileSync(join(up, "styles", "base.css"), "utf8"),
  ...readdirSync(join(up, "components"))
    .filter((d) => existsSync(join(up, "components", d, `${d}.css`)))
    .map((d) => readFileSync(join(up, "components", d, `${d}.css`), "utf8")),
].join("\n")
writeFileSync(join(out, "examples.css"), `/* ui-sdk example styles, vercel-labs/ui (Apache-2.0) */\n${css}`)

cpSync(args.esm, join(out, "esm"), { recursive: true })
cpSync(args.vendor, join(out, "vendor"), { recursive: true })
cpSync(join(up, "LICENSE"), join(out, "LICENSE-ui-sdk.txt"))
for (const f of ["index.html", "theme.css", "avatar.svg"]) cpSync(join(here, f), join(out, f))
// austegard.com's deploy skips every file named LICENSE; give the licence files an extension so they ship.
rmSync(join(out, "esm", "LICENSE"), { force: true })
for (const dir of readdirSync(join(out, "vendor"))) {
  const from = join(out, "vendor", dir, "LICENSE")
  if (existsSync(from)) {
    cpSync(from, join(out, "vendor", dir, "LICENSE.txt"))
    rmSync(from)
  }
}

const BUILD = new Date().toISOString().slice(0, 16).replace(/[-:T]/g, "")
writeFileSync(join(out, "index.html"), readFileSync(join(out, "index.html"), "utf8").replaceAll("__BUILD__", BUILD))
writeFileSync(join(out, "build.json"), JSON.stringify({ build: BUILD }) + "\n")

const files = result.output.filter((o) => o.type === "chunk")
console.log(
  `${categorized.length} components, ${exampleCount} examples, ${files.length} JS files, ` +
    `${(files.reduce((n, o) => n + o.code.length, 0) / 1024).toFixed(0)} KB JS, ${(css.length / 1024).toFixed(0)} KB CSS, build ${BUILD}`
)
