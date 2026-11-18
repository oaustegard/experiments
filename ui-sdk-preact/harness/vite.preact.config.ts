import tailwind from "@tailwindcss/vite"
import { defineConfig } from "vite"

const shim = new URL("./react-on-preact.ts", import.meta.url).pathname

// Serves the React fixtures with React swapped for preact/compat.
export default defineConfig({
  plugins: [tailwind()],
  resolve: {
    alias: [
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
