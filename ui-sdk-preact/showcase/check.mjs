// Opens every component page of the showcase, in light and dark, and reports
// examples that threw, console errors, empty stages, and axe color-contrast
// violations. Writes /tmp/jail/showcase/report.json and screenshots.
import { chromium } from "/tmp/jail/ui-sdk/node_modules/playwright/index.mjs"
import { readFileSync, writeFileSync } from "node:fs"

const BASE = "http://localhost:5420/ai-tools/ui-sdk-preact/"
const axe = readFileSync("/tmp/jail/showcase/axe.min.js", "utf8")
const browser = await chromium.launch()
const report = { pages: {}, shell: {}, console: [] }

async function newPage(viewport = { width: 1280, height: 900 }) {
  const ctx = await browser.newContext({ viewport, deviceScaleFactor: 1 })
  const page = await ctx.newPage()
  page.on("console", (m) => {
    if (m.type() === "error") report.console.push({ url: page.url(), text: m.text().slice(0, 300) })
  })
  page.on("pageerror", (e) => report.console.push({ url: page.url(), text: "pageerror: " + e.message.slice(0, 300) }))
  return page
}

async function contrast(page) {
  await page.evaluate(axe)
  const result = await page.evaluate(async () => {
    const r = await window.axe.run(document, { runOnly: ["color-contrast"], resultTypes: ["violations", "incomplete"] })
    const pack = (list) =>
      list.flatMap((v) =>
        v.nodes.map((n) => ({
          target: n.target.join(" "),
          html: n.html.slice(0, 140),
          data: n.any?.[0]?.data && {
            fg: n.any[0].data.fgColor,
            bg: n.any[0].data.bgColor,
            ratio: n.any[0].data.contrastRatio,
            expected: n.any[0].data.expectedContrastRatio,
          },
        }))
      )
    return { violations: pack(r.violations), incomplete: r.incomplete.reduce((n, v) => n + v.nodes.length, 0) }
  })
  return result
}

const page = await newPage()
await page.goto(BASE)
await page.waitForSelector("body[data-ready]", { timeout: 20000 })
const names = await page.$$eval(".cards a.card", (as) => as.map((a) => a.getAttribute("href").slice(1)))
report.shell.cards = names.length

for (const scheme of ["light", "dark"]) {
  await page.evaluate((s) => {
    document.documentElement.style.colorScheme = s
    document.documentElement.dataset.theme = s
  }, scheme)
  await page.evaluate(() => (location.hash = ""))
  await page.waitForTimeout(300)
  report.shell[`overview-${scheme}`] = await contrast(page)
  await page.screenshot({ path: `/tmp/jail/showcase/overview-${scheme}.png` })
  for (const name of names) {
    const before = report.console.length
    await page.evaluate((n) => (location.hash = n), name)
    await page.waitForFunction(() => !document.querySelector(".loading") && document.querySelector(".stage"), null, { timeout: 20000 }).catch(() => {})
    await page.waitForTimeout(700)
    const info = await page.evaluate(() => ({
      stages: document.querySelectorAll(".stage").length,
      empty: [...document.querySelectorAll(".stage")].filter((s) => !s.textContent.trim() && !s.querySelector("svg,img,canvas,input,button,[role]")).map((s) => s.dataset.example),
      failures: [...window.__showcaseFailures],
      loadError: document.querySelector(".page > .ex-error")?.textContent ?? null,
    }))
    const entry = (report.pages[name] ??= {})
    if (scheme === "light") Object.assign(entry, info, { consoleErrors: report.console.length - before })
    entry[`contrast-${scheme}`] = await contrast(page)
    if (["dialog", "menu", "select", "date-picker"].includes(name)) await page.screenshot({ path: `/tmp/jail/showcase/${name}-${scheme}.png` })
  }
}

// The shell's own ui-sdk parts: menubar navigation, palette, theme switch.
await page.evaluate(() => (location.hash = ""))
await page.getByRole("menuitem", { name: "Components" }).click()
await page.getByRole("menuitem", { name: "Overlays" }).hover()
await page.getByRole("menuitem", { name: "Tooltip" }).click()
report.shell.menubarNav = await page.evaluate(() => location.hash)
await page.keyboard.press("Control+k")
await page.getByRole("combobox", { name: "Search components" }).fill("pagin")
await page.keyboard.press("Enter")
await page.waitForTimeout(300)
report.shell.paletteNav = await page.evaluate(() => location.hash)
await page.getByRole("menuitem", { name: "View" }).click()
await page.getByRole("menuitemradio", { name: "Dark" }).click()
report.shell.themeAfterMenu = await page.evaluate(() => document.documentElement.dataset.theme)
report.shell.focusAfterNav = await page.evaluate(() => document.activeElement?.id || document.activeElement?.tagName)

const phone = await newPage({ width: 390, height: 844 })
await phone.goto(BASE + "#combobox")
await phone.waitForSelector("body[data-ready]")
await phone.waitForTimeout(1500)
report.shell.phoneOverflow = await phone.evaluate(() => document.documentElement.scrollWidth - document.documentElement.clientWidth)
report.shell.phoneHeader = await phone.evaluate(() => ({ trigger: Math.round(document.querySelector(".palette-trigger").getBoundingClientRect().right), menubar: Math.round(document.querySelector(".top [data-slot=menubar]").getBoundingClientRect().right), width: innerWidth }))
await phone.screenshot({ path: "/tmp/jail/showcase/phone-combobox.png" })

writeFileSync("/tmp/jail/showcase/report.json", JSON.stringify(report, null, 1))
const pages = Object.entries(report.pages)
console.log(JSON.stringify({
  cards: report.shell.cards,
  stages: pages.reduce((n, [, p]) => n + p.stages, 0),
  failures: pages.flatMap(([, p]) => p.failures),
  loadErrors: pages.filter(([, p]) => p.loadError).map(([n]) => n),
  empty: pages.flatMap(([, p]) => p.empty),
  consoleErrors: report.console.length,
  contrastLight: pages.reduce((n, [, p]) => n + p["contrast-light"].violations.length, 0),
  contrastDark: pages.reduce((n, [, p]) => n + p["contrast-dark"].violations.length, 0),
  overview: [report.shell["overview-light"].violations.length, report.shell["overview-dark"].violations.length],
  shell: { menubarNav: report.shell.menubarNav, paletteNav: report.shell.paletteNav, theme: report.shell.themeAfterMenu, focus: report.shell.focusAfterNav, phoneOverflow: report.shell.phoneOverflow, phoneHeader: report.shell.phoneHeader },
}))
await browser.close()
