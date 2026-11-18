import tailwind from "@tailwindcss/vite"
import { defineConfig } from "vite"

const shim = new URL("./react-on-preact.ts", import.meta.url).pathname
const dist = new URL("../../ui-sdk-pd/dist/", import.meta.url).pathname

// Serves the React fixtures against the built ui-sdk-preact package. Fixtures
// still import "react" for their own hooks, which goes to preact/compat.
export default defineConfig({
  plugins: [tailwind()],
  resolve: {
    alias: [
      { find: /^@ui-sdk\/react\/(.+)$/, replacement: `${dist}$1.js` },
      { find: /^react-dom\/client$/, replacement: "preact/compat/client" },
      { find: /^react-dom\/server$/, replacement: "preact/compat/server" },
      { find: /^react-dom$/, replacement: "preact/compat" },
      { find: /^react\/jsx-runtime$/, replacement: "preact/compat/jsx-runtime" },
      { find: /^react\/jsx-dev-runtime$/, replacement: "preact/compat/jsx-dev-runtime" },
      { find: /^react$/, replacement: shim },
    ],
    dedupe: ["preact"],
  },
  optimizeDeps: { noDiscovery: true, include: ["@guidepup/virtual-screen-reader"] },
})
