# ui-sdk-preact

[ui-sdk](https://ui-sdk.dev) (vercel-labs/ui, Apache-2.0) ships adapters for
React, Vue, Svelte, Solid and web components. This is a Preact port: the
published `@ui-sdk/react` 0.1.0-beta.10, run on Preact 11.2 through
`preact/compat`, plus a 90-line module that fixes three places where Preact
renders the adapter's output differently from React. The adapter's code is not
changed. [Demo page](https://austegard.com/experiments/ui-sdk-preact/web/).

Started 2026-11-18. Oskar: "Could we build a Preact port for these?
https://ui-sdk.dev/docs/core".

## Results

ui-sdk's own Playwright suite was the test: the 2,991 tests of its `react`
project (behavior, accessibility, screen reader, right-to-left, rendering,
kitchen sinks, count gates and leak specs, on the playground's React fixtures),
run once on React 19.2 and once with `react` and `react-dom` aliased to
Preact. A probe confirmed which library rendered each page: Preact's `__k` on
the root, and no React container keys.

| Run | Passed | Failed or timed out | Skipped |
|---|---|---|---|
| React 19.2.3 | 2,826 | 1 | 105 |
| Preact 11.2.0, compat only | 2,761 | 66 | 105 |
| Preact 11.2.0, compat + `fixes.js` (adapter source, main `56a7c93`) | 2,810 | 17 | 105 |
| Preact 11.2.0, the built package (`dist/`, tag `@ui-sdk/react@0.1.0-beta.10`) | 2,810 | 18 | 105 |

The one React failure (`scroll-area.enhancements` "hiding native scrollbars
injects no style element") failed on every run. The built package's extra
failure, a menu pointer test, passed 5 of 5 on a rerun; it also failed once on
React at the same tag, so it is a timing flake.

### The 65 Preact-only failures, by cause

| Cause | Tests | Status |
|---|---|---|
| My harness: Vite was not told to pre-bundle `@guidepup/virtual-screen-reader`, so every `*.sr` spec timed out | 40 | fixed in the harness |
| Preact's `useId` restarts at `P0-0` in every `render()` root, so kitchen-sink pages (one root per variant) repeated ids and triggers found another root's panel | 6 | fixed in `fixes.js` |
| Removing a prop that is a DOM property sets it to `""`, so a button rendered as `<a>` kept `type=""` | 2 | fixed in `fixes.js` |
| Core writes `webkitdirectory=""` for present; Preact assigns the boolean property, where `""` is false, so folder picking was off | 3 | fixed in `fixes.js` |
| `<Activity>` (cached pages) has no Preact counterpart | 8 (10 after the fix run) | React-only |
| `useTransition` holding the old selection until the new one commits: Preact's is synchronous | 3 | React-only |
| One closed popup's DOM stays reachable after 200 open/close and mount cycles (dialog, combobox, autocomplete) | 3 | open |

The leak is bounded: the count is the parts of one popup, not 200, and the
unmount-while-open leak specs pass. It looks like a reference Preact keeps to
the last tree it rendered; not diagnosed further. The two cached-page tests
added in the fix run are the sheet-presentation variants, which timed out on
the same missing `<Activity>`; the first run had stopped them earlier.

Preact's repeated `useId` across roots is a known Preact issue with no fix
upstream ([write-up](https://dev.to/teetotum/whats-wrong-with-useid-react-and-preact-1ifg)).
`fixes.js` numbers each container on its first client render and leaves the
first one at `P0-`, so a single-root page, and a hydrated one, renders the ids
Preact always did.

### Size

`web/esm/` is a code-split build of every component and every `@ui-sdk/core`
entry point, 1.05 MB in 294 files, with Preact left to the page's import map.
What a page loads for one component (minified, gzipped as one stream):

| Component | Files | Minified | Gzip |
|---|---|---|---|
| button | 18 | 45.6 KB | 16.8 KB |
| tabs | 28 | 62.9 KB | 22.6 KB |
| dialog | 44 | 120.2 KB | 42.7 KB |
| menu | 49 | 148.3 KB | 51.7 KB |
| select | 60 | 169.4 KB | 59.8 KB |
| combobox | 60 | 176.7 KB | 62.2 KB |

A single-component esbuild bundle of the same code, Preact excluded, gzips to
11,628 bytes for button, 46,472 for dialog and 59,605 for menu. Upstream's
`size-budget.json` measures `@ui-sdk/react` the same way, gzip with React
excluded: 10,110, 43,784 and 55,187 bytes with the lazy parts. The 7 to 15%
gap is bundler output, not added code; compat itself is 3,982 bytes gzip on
top. esbuild's code splitting
was tried first and pulled 248 KB into a button, because it assigns whole files
to chunks and core's barrels reach most of them; rolldown chunks by what each
entry uses.

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

Without one, map Preact and the ESM build in an import map, as `web/index.html`
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
| `web/index.html` | Demo: menu, dialog, and tabs in two roots, with no build step. Reports to `gpu-bridge` with `?log=1` |
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
