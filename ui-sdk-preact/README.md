# ui-sdk-preact

[ui-sdk](https://ui-sdk.dev) (vercel-labs/ui, Apache-2.0) is a library of
unstyled UI behaviors: menus, dialogs, tabs, comboboxes and about 60 more, with
keyboard handling, focus management and ARIA built in. It ships adapters for
React, Vue, Svelte, Solid and web components, none for Preact.

The question: does the React adapter run on Preact unchanged, through
`preact/compat` (Preact's React compatibility layer), and what breaks? Almost
all of it runs. On ui-sdk's own React test suite, Preact 11 passes 2,868 of
2,991 tests against React's 2,884. The difference is 16 tests: two React
features Preact doesn't have, and one bounded memory leak. Getting there took a
83-line module, `src/fixes.js`, which corrects three places where Preact
renders the adapter's output differently from React. The adapter's code is not
changed. It needs Preact 11 (see Use). The
[showcase](https://austegard.com/ai-tools/ui-sdk-preact/) runs all 55 components
and every upstream example for them on Preact.

Started 2026-11-18, from Oskar Austegard's request: "Could we build a Preact
port for these? https://ui-sdk.dev/docs/core".

## Results

The test was ui-sdk's own Playwright suite: the 2,991 tests of its `react`
project, run against the repo's playground pages for the React adapter. They
cover behavior, accessibility, screen-reader announcements, right-to-left
layout and custom rendering, plus "kitchen sink" pages that mount every variant
of a component, gates on how many times a component re-renders, and leak checks. The first three runs used vercel-labs/ui at main `56a7c93`; the fourth checked out the release tag the npm package was built from, which has one more test. Four runs, all in Chromium:

| Run | Passed | Failed or timed out | Skipped |
|---|---|---|---|
| React 19.2.3 | 2,884 | 1 | 106 |
| Preact 11.2.0, `preact/compat` only | 2,819 | 66 | 106 |
| Preact 11.2.0, compat + `fixes.js`, adapter source | 2,868 | 17 | 106 |
| Preact 11.2.0, the built package (`dist/`) at tag `@ui-sdk/react@0.1.0-beta.10` (2,992 tests) | 2,868 | 18 | 106 |

For the Preact runs, `react` and `react-dom` resolved to `preact/compat`. A
check on the rendered page confirmed Preact did the rendering: Preact's own
field (`__k`) was on the root element, and React's container keys were absent.
The last two runs also had the harness fix described below. The skipped tests
are the same 106 in every run (the suite skips them in this project).

One test failed on React and on every Preact run: `scroll-area.enhancements`,
"hiding native scrollbars injects no style element". So of the compat-only
run's 66 failures, 65 were Preact-only, and of the fixed run's 17, 16 were.
The built package's 18th failure, a menu pointer test ("moving from a submenu
trigger straight to another item highlights that item"), passed 5 of 5 when
rerun on its own; rerun against React at the same tag it also passed 5 of 5,
having failed there at least once in an earlier rerun. It is timing under load.

### The 65 Preact-only failures, by cause

| Cause | Tests | After the fixes |
|---|---|---|
| The test harness: Vite was not told to pre-bundle `@guidepup/virtual-screen-reader`, so every screen-reader spec (`*.sr`) timed out | 38 | pass (harness fixed) |
| Preact's `useId` restarts at `P0-0` in every `render()` root. Kitchen-sink pages mount one root per variant, so ids repeated and a trigger's `aria-controls` found another root's panel | 6 | pass |
| Removing a prop that is a DOM property makes Preact set the property to `""` instead of removing the attribute, so a button rendered as a link kept `type=""` | 2 | pass |
| Core writes `webkitdirectory=""` to mean present. Preact assigns the boolean property, where `""` is false, so the folder picker was off | 3 | pass |
| React's `<Activity>`, which keeps a hidden page's state, has no Preact counterpart (the cached-page specs) | 10 | fail: React-only |
| React's concurrent `useTransition` keeps the old selection on screen until the new one commits; Preact's runs synchronously | 3 | fail: React-only |
| After 200 open/close and 200 mount cycles, one closed popup's DOM is still reachable (dialog, combobox, autocomplete) | 3 | fail: open |

The leak count is one popup's parts, not one per cycle: 4 parts for the dialog
(close, content, description, title) and 5 each for the combobox and the
autocomplete (content, list, items), after 200 cycles.
The specs that unmount a popup while it is open pass. The retaining reference
was not found.

Preact repeating `useId` across roots is a known Preact issue with no fix
upstream ([write-up](https://dev.to/teetotum/whats-wrong-with-useid-react-and-preact-1ifg)).
`fixes.js` gives each container a number the first time it renders on the
client: the first gets `P0-0`, `P0-1`, …, as stock Preact does, the second
`P1-0`, and so on. A page with one root renders the ids it always did.
Hydration was not tested.

### Size

`web/esm/` is a code-split build of every component and every `@ui-sdk/core`
entry point, 1.05 MB in 294 files, with Preact left to the page's import map.
What a page loads for one component, from that build (all the files it
imports, minified, gzipped as one stream):

| Component | Files | Minified | Gzip |
|---|---|---|---|
| button | 18 | 45.6 KB | 16.8 KB |
| tabs | 28 | 62.9 KB | 22.6 KB |
| dialog | 44 | 120.2 KB | 42.7 KB |
| menu | 49 | 148.3 KB | 51.7 KB |
| select | 60 | 169.4 KB | 59.8 KB |
| combobox | 60 | 176.7 KB | 62.2 KB |

A split build costs something over bundling one component alone, because
shared chunks carry code that other components also use. Bundled alone, with
Preact excluded, the same code gzips to 11,628 bytes for button, 46,472 for
dialog and 59,605 for menu. Upstream's `size-budget.json` measures
`@ui-sdk/react` the same way (gzip, React excluded, lazy parts included):
10,110, 43,784 and 55,187 bytes, so the Preact build is 6 to 15% larger;
different bundlers produced the two, and the gap was not traced further. The comparison is between single-component bundles; the
split build in the table above costs more per component. Neither figure
includes Preact itself: a page also loads `preact` (4,997 bytes gzip),
`preact/hooks` (1,589) and `preact/compat` (3,982).

The split build is rolldown's. esbuild's code splitting was tried first and
put 248 KB in front of a button: it assigns whole files to chunks, and core's
barrel modules reach most of the library.

## Use

With a bundler, build the package and import from it; `@ui-sdk/core` and
`preact` (11) are peers:

```bash
npm pack @ui-sdk/react@0.1.0-beta.10
node build.mjs ui-sdk-react-0.1.0-beta.10.tgz ui-sdk-preact/
```

```js
import { Dialog, DialogTrigger, DialogContent, DialogTitle, DialogClose } from "ui-sdk-preact/dialog"
```

Without one, map Preact and the ESM build in an import map, as the showcase
does, and write the markup with htm:

```html
<script type="importmap">
{ "imports": {
  "preact": "./vendor/preact/preact.mjs",
  "preact/hooks": "./vendor/preact/hooks.mjs",
  "preact/compat": "./vendor/preact/compat.mjs",
  "preact/jsx-runtime": "./vendor/preact/jsx-runtime.mjs",
  "preact/compat/jsx-runtime": "./vendor/preact/compat-jsx-runtime.mjs",
  "htm": "./vendor/htm/htm.mjs",
  "ui-sdk/": "./esm/"
} }
</script>
<script type="module">
  import { h, render } from "preact"
  import htm from "htm"
  import { Tabs, TabsList, TabsTab, TabsPanel } from "ui-sdk/tabs.js"
  const html = htm.bind(h)
  render(html`<${Tabs} defaultValue="a">…<//>`, document.getElementById("app"))
</script>
```

Part names, props and data attributes are the React adapter's; the
[ui-sdk docs](https://ui-sdk.dev/docs) apply as written. Style parts by their
state attributes (`[data-highlighted]`, `[data-selected]`, `[data-open]`).

Only Preact 11 was tested. The adapter targets React 19, which passes `ref` to
function components as a prop; Preact 11 does too, Preact 10 does not without
`forwardRef`, so parts that take a ref are expected to lose it on 10.

## Files

| File | What it is |
|---|---|
| `build.mjs` | Turns the published `@ui-sdk/react` tarball into `ui-sdk-preact`: React imports go to `dist/_preact/`, types to `preact/compat`. Fails on any React specifier it doesn't know |
| `src/fixes.js` | The three fixes, as Preact option hooks scoped to ui-sdk parts (`data-slot`) |
| `src/compat.js`, `src/jsx-runtime.js` | What the package imports in place of `react` and `react/jsx-runtime` |
| `esm.mjs` | Builds `web/esm/` with rolldown |
| `catalog.mjs` | Writes the component catalog in the developing-preact skill's `references/ui-sdk.md` from the npm package's docs and the `web/esm/` exports |
| `showcase/` | The page at austegard.com/ai-tools/ui-sdk-preact/: `build.mjs` compiles all 578 upstream React examples (from the release tag) into one module per component, puts ui-sdk, Preact and the icons in one shared chunk (a component page makes three module requests), and adds upstream's example CSS; `app.jsx` is the shell (menubar, palette, sidebar), `theme.css` the restyle, `check.mjs` the check (run inside `jail.sh`). Checked in headless Chromium: all 580 example placements mount without an error, and axe-core finds no color-contrast violation on any page in light or dark |
| `web/index.html` | Redirects to the showcase (it was the first, three-widget demo) |
| `web/esm/`, `web/vendor/` | The build (ui-sdk's LICENSE in `esm/`), Preact 11.2.0 and htm 3.1.1 (MIT) |
| `harness/` | The Vite configs, the runner and the result differ (`diff.py`) used to put ui-sdk's suite on Preact. `react-on-preact.ts` imports `preact-fixes.js`, a copy of `src/fixes.js` placed next to it in the playground |

## Reproduce

The suite ran inside `scripts/jail.sh` (claude-workspace): no network, as
`nobody`. Dependencies came from the registry into a directory holding only
the repo's manifests and lockfile (`pnpm install --ignore-scripts
--frozen-lockfile --filter playground... --filter tests...`), then were copied
into the checkout, so no repo code ran outside the jail. `harness/run.sh`
starts the playground with a given Vite config and runs Playwright against it
in one jail call; `harness/react-on-preact.ts` is what `react` resolves to
(compat, `fixes.js`, and a pass-through `Profiler`).

```bash
sh scripts/jail.sh -C /tmp/jail/runner -e PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers -- \
  /tmp/jail/runner/run.sh vite.preact.config.ts 5322 --project react --workers 4
```
