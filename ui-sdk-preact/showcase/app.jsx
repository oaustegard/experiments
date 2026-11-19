// The showcase shell. Built from ui-sdk itself: the menubar, the command
// palette and the source panels are ui-sdk components on Preact, like the
// examples they frame.
import { render, Component } from "preact"
import { useState, useEffect, useCallback, useRef } from "preact/hooks"
import { Menubar } from "@ui-sdk/react/menubar"
import {
  Menu,
  MenuContent,
  MenuItem,
  MenuItemIndicator,
  MenuRadioGroup,
  MenuRadioItem,
  MenuCheckboxItem,
  MenuSeparator,
  MenuGroup,
  MenuGroupLabel,
  MenuSub,
  MenuSubContent,
  MenuSubTrigger,
  MenuTrigger,
} from "@ui-sdk/react/menu"
import { Dialog, DialogBackdrop, DialogContent, DialogShortcut, DialogTitle, DialogTrigger } from "@ui-sdk/react/dialog"
import {
  Command,
  CommandEmpty,
  CommandGroup,
  CommandGroupLabel,
  CommandInput,
  CommandItem,
  CommandList,
} from "@ui-sdk/react/command"
import { Collapsible, CollapsiblePanel, CollapsibleTrigger } from "@ui-sdk/react/collapsible"
import { ChevronDownIcon, ChevronRightIcon, SearchIcon } from "lucide-react"
import { LogContext } from "showcase:log"
import catalog from "showcase:catalog"

const relay = window.relay || { log() {}, warn() {}, error() {} }
const failures = (window.__showcaseFailures = [])
const all = catalog.flatMap((group) => group.items.map((item) => ({ ...item, category: group.category })))
const byName = Object.fromEntries(all.map((item) => [item.name, item]))
const totalExamples = all.reduce((n, item) => n + item.count, 0)

const stored = (key, fallback) => {
  try {
    return localStorage.getItem(`ui-sdk-preact:${key}`) ?? fallback
  } catch {
    return fallback
  }
}
const store = (key, value) => {
  try {
    localStorage.setItem(`ui-sdk-preact:${key}`, value)
  } catch {}
}

function applyTheme(theme) {
  const root = document.documentElement
  root.dataset.theme = theme
  root.style.colorScheme = theme === "system" ? "light dark" : theme
}

const routeOf = () => {
  const name = decodeURIComponent(location.hash.slice(1))
  return byName[name] ? name : null
}

// Inline `code` from the docs text, everything else as plain text.
function Prose({ text }) {
  return text.split(/(`[^`]+`)/).map((part, i) => (part.startsWith("`") ? <code key={i}>{part.slice(1, -1)}</code> : part))
}

class Boundary extends Component {
  state = { error: null }
  componentDidCatch(error) {
    const message = String(error?.message || error)
    this.setState({ error: message })
    failures.push({ id: this.props.id, message })
    relay.error("example failed", { id: this.props.id, message })
  }
  render() {
    return this.state.error ? (
      <p class="ex-error" role="alert">
        This example failed on Preact: {this.state.error}
      </p>
    ) : (
      this.props.children
    )
  }
}

function ExampleCard({ example, showSource }) {
  const [log, setLog] = useState([])
  const push = useCallback((message) => setLog((entries) => [...entries.slice(-4), String(message)]), [])
  const file = `${example.id.split("/")[1]}.tsx`
  return (
    <figure class="ex" id={example.id.replace("/", "--")}>
      <div class="stage" data-example={example.id}>
        <Boundary id={example.id}>
          <LogContext.Provider value={push}>
            <example.Example />
          </LogContext.Provider>
        </Boundary>
      </div>
      {example.caption && <figcaption>{example.caption}</figcaption>}
      {log.length > 0 && (
        <ol class="log" aria-label="Events from this example" aria-live="polite">
          {log.map((entry, i) => (
            <li key={i}>{entry}</li>
          ))}
        </ol>
      )}
      <Collapsible class="source" defaultOpen={showSource}>
        <CollapsibleTrigger class="source-toggle">
          <ChevronRightIcon aria-hidden="true" class="chev" />
          <span>
            Source <span class="file">{file}</span>
          </span>
          <span class="hint">upstream React, unchanged</span>
        </CollapsibleTrigger>
        <CollapsiblePanel class="source-panel">
          <pre>
            <code>{example.source.trim()}</code>
          </pre>
        </CollapsiblePanel>
      </Collapsible>
    </figure>
  )
}

function ComponentPage({ name, showSource }) {
  const [page, setPage] = useState(null)
  const [error, setError] = useState(null)
  useEffect(() => {
    let live = true
    setPage(null)
    setError(null)
    const started = performance.now()
    byName[name]
      .load()
      .then((module) => {
        if (!live) return
        setPage(module.default)
        relay.log("page", { name, ms: Math.round(performance.now() - started) })
      })
      .catch((e) => {
        if (!live) return
        setError(String(e?.message || e))
        relay.error("page failed to load", { name, message: String(e?.message || e) })
      })
    return () => {
      live = false
    }
  }, [name])
  const item = byName[name]
  return (
    <article class="page" aria-busy={!page && !error}>
      <p class="crumb">{item.category}</p>
      <h1 tabIndex={-1} id="page-title">
        {item.title}
      </h1>
      <p class="lede">{item.description}</p>
      <p class="links">
        <a href={`https://ui-sdk.dev/docs/react/components/${name}`}>ui-sdk docs</a>
        <span aria-hidden="true"> · </span>
        <a href={`https://github.com/vercel-labs/ui/tree/@ui-sdk/react@0.1.0-beta.10/examples/components/${name}/react`}>
          Example sources
        </a>
        <span aria-hidden="true"> · </span>
        <span>
          import from <code>ui-sdk/{name}.js</code>
        </span>
      </p>
      {error && (
        <p class="ex-error" role="alert">
          The examples did not load: {error}
        </p>
      )}
      {!page && !error && <p class="loading">Loading {item.count} examples…</p>}
      {page &&
        page.sections.map((section, i) => (
          <section key={i}>
            {section.heading && <h2>{section.heading}</h2>}
            {section.text && (
              <p class="text">
                <Prose text={section.text} />
              </p>
            )}
            {section.examples.map((example) => (
              <ExampleCard key={example.id} example={example} showSource={showSource} />
            ))}
          </section>
        ))}
      <Pager name={name} />
    </article>
  )
}

function Pager({ name }) {
  const i = all.findIndex((item) => item.name === name)
  const prev = all[i - 1]
  const next = all[i + 1]
  return (
    <nav class="pager" aria-label="Previous and next component">
      {prev ? (
        <a href={`#${prev.name}`} rel="prev">
          <span class="dir">Previous</span> {prev.title}
        </a>
      ) : (
        <span />
      )}
      {next && (
        <a href={`#${next.name}`} rel="next">
          <span class="dir">Next</span> {next.title}
        </a>
      )}
    </nav>
  )
}

function Overview() {
  return (
    <article class="page overview">
      <h1 tabIndex={-1} id="page-title">
        ui-sdk on Preact
      </h1>
      <p class="lede">
        All {all.length} components of <a href="https://ui-sdk.dev">ui-sdk</a>, Vercel's library of accessible,
        unstyled UI behaviors, running on Preact 11. Each page runs every upstream React example for that
        component, {totalExamples} in all, compiled once and rendered by Preact through <code>preact/compat</code>.
        Open the source under any example to see the code that is running.
      </p>
      <p class="text">
        The menubar above, the search palette (<kbd>Ctrl</kbd>/<kbd>⌘</kbd> <kbd>K</kbd>) and the source panels are
        ui-sdk components too. On ui-sdk's own React test suite this port passes 2,868 of 2,991 tests, against React's
        2,884; what differs is listed in the <a href="https://github.com/oaustegard/experiments/tree/main/ui-sdk-preact">write-up</a>.
      </p>
      {catalog.map((group) => (
        <section key={group.category}>
          <h2>{group.category}</h2>
          <ul class="cards">
            {group.items.map((item) => (
              <li key={item.name}>
                <a class="card" href={`#${item.name}`}>
                  <span class="card-title">{item.title}</span>
                  <span class="card-text">{item.description}</span>
                  <span class="card-count">
                    {item.count} example{item.count === 1 ? "" : "s"}
                  </span>
                </a>
              </li>
            ))}
          </ul>
        </section>
      ))}
    </article>
  )
}

function Sidebar({ current }) {
  return (
    <nav class="side" aria-label="Components">
      <a class="side-home" href="#" aria-current={current ? undefined : "page"}>
        Overview
      </a>
      {catalog.map((group) => (
        <div class="side-group" key={group.category}>
          <h2 class="side-heading">{group.category}</h2>
          <ul>
            {group.items.map((item) => (
              <li key={item.name}>
                <a href={`#${item.name}`} aria-current={item.name === current ? "page" : undefined}>
                  {item.title}
                </a>
              </li>
            ))}
          </ul>
        </div>
      ))}
    </nav>
  )
}

function Palette({ go }) {
  const [open, setOpen] = useState(false)
  return (
    <Dialog open={open} onOpenChange={setOpen}>
      <DialogTrigger class="palette-trigger" shortcut="mod+k">
        <SearchIcon aria-hidden="true" />
        <span class="palette-label">Search components</span>
        <DialogShortcut />
      </DialogTrigger>
      <DialogBackdrop />
      <DialogContent class="palette">
        <DialogTitle>Go to a component</DialogTitle>
        <Command
          onAction={(value) => {
            setOpen(false)
            go(value)
          }}
        >
          <div data-slot="row">
            <SearchIcon aria-hidden="true" />
            <CommandInput aria-label="Search components" placeholder="Type a component name…" />
          </div>
          <CommandEmpty>No component by that name.</CommandEmpty>
          <CommandList>
            {catalog.map((group) => (
              <CommandGroup key={group.category}>
                <CommandGroupLabel>{group.category}</CommandGroupLabel>
                {group.items.map((item) => (
                  <CommandItem key={item.name} value={item.name}>
                    {item.title}
                  </CommandItem>
                ))}
              </CommandGroup>
            ))}
          </CommandList>
        </Command>
      </DialogContent>
    </Dialog>
  )
}

function TopBar({ current, go, theme, setTheme, showSource, setShowSource }) {
  const i = all.findIndex((item) => item.name === current)
  return (
    <header class="top" data-example="shell">
      <a class="brand" href="#" aria-label="ui-sdk on Preact, overview">
        <span class="brand-mark" aria-hidden="true">◇</span>{" "}
        <span class="brand-text">
          ui-sdk <span class="brand-on">on Preact</span>
        </span>
      </a>
      <Menubar aria-label="Showcase">
        <Menu>
          <MenuTrigger>
            Components <ChevronDownIcon aria-hidden="true" />
          </MenuTrigger>
          <MenuContent>
            <MenuItem onSelect={() => go(null)}>Overview</MenuItem>
            <MenuSeparator />
            {catalog.map((group) => (
              <MenuSub key={group.category}>
                <MenuSubTrigger>{group.category}</MenuSubTrigger>
                <MenuSubContent>
                  {group.items.map((item) => (
                    <MenuItem key={item.name} onSelect={() => go(item.name)}>
                      {item.title}
                    </MenuItem>
                  ))}
                </MenuSubContent>
              </MenuSub>
            ))}
          </MenuContent>
        </Menu>
        <Menu>
          <MenuTrigger>
            View <ChevronDownIcon aria-hidden="true" />
          </MenuTrigger>
          <MenuContent>
            <MenuGroup>
              <MenuGroupLabel>Theme</MenuGroupLabel>
              <MenuRadioGroup value={theme} onValueChange={setTheme}>
                {[
                  ["system", "Match system"],
                  ["light", "Light"],
                  ["dark", "Dark"],
                ].map(([value, label]) => (
                  <MenuRadioItem key={value} value={value}>
                    <MenuItemIndicator />
                    {label}
                  </MenuRadioItem>
                ))}
              </MenuRadioGroup>
            </MenuGroup>
            <MenuSeparator />
            <MenuCheckboxItem checked={showSource} onCheckedChange={setShowSource}>
              <MenuItemIndicator />
              Show source by default
            </MenuCheckboxItem>
          </MenuContent>
        </Menu>
        <Menu>
          <MenuTrigger>
            Go <ChevronDownIcon aria-hidden="true" />
          </MenuTrigger>
          <MenuContent>
            <MenuItem disabled={i <= 0} onSelect={() => go(all[i - 1]?.name ?? null)}>
              Previous component
            </MenuItem>
            <MenuItem disabled={i === all.length - 1} onSelect={() => go(all[i + 1]?.name)}>
              Next component
            </MenuItem>
            <MenuSeparator />
            <MenuItem render={<a href="https://ui-sdk.dev/docs" />}>ui-sdk documentation</MenuItem>
            <MenuItem render={<a href="https://github.com/oaustegard/experiments/tree/main/ui-sdk-preact" />}>
              How the port was built
            </MenuItem>
            <MenuItem render={<a href="/ai-tools/" />}>All AI tools</MenuItem>
          </MenuContent>
        </Menu>
      </Menubar>
      <Palette go={go} />
    </header>
  )
}

function App() {
  const [current, setCurrent] = useState(routeOf)
  const [theme, setThemeState] = useState(() => stored("theme", "system"))
  const [showSource, setShowSourceState] = useState(() => stored("source", "0") === "1")
  const first = useRef(true)

  useEffect(() => applyTheme(theme), [theme])
  useEffect(() => {
    const onHash = () => setCurrent(routeOf())
    window.addEventListener("hashchange", onHash)
    return () => window.removeEventListener("hashchange", onHash)
  }, [])
  useEffect(() => {
    document.title = current ? `${byName[current].title} · ui-sdk on Preact` : "ui-sdk on Preact"
    if (first.current) {
      first.current = false
      return
    }
    window.scrollTo(0, 0)
    document.getElementById("page-title")?.focus({ preventScroll: true })
  }, [current])

  const go = useCallback((name) => {
    location.hash = name ? name : ""
    if (!name) setCurrent(null)
  }, [])
  const setTheme = useCallback((value) => {
    setThemeState(value)
    store("theme", value)
  }, [])
  const setShowSource = useCallback((value) => {
    setShowSourceState(value)
    store("source", value ? "1" : "0")
  }, [])

  return (
    <>
      <TopBar
        current={current}
        go={go}
        theme={theme}
        setTheme={setTheme}
        showSource={showSource}
        setShowSource={setShowSource}
      />
      <div class="layout">
        <Sidebar current={current} />
        <main id="main">
          {current ? <ComponentPage key={current + showSource} name={current} showSource={showSource} /> : <Overview />}
        </main>
      </div>
    </>
  )
}

applyTheme(stored("theme", "system"))
render(<App />, document.getElementById("app"))
document.body.dataset.ready = ""
