import "./preact-fixes.js"
import * as compat from "preact/compat"
import type { ComponentChildren } from "preact"

// React.Profiler has no Preact counterpart; fixtures only use it to count commits.
const Profiler = ({ children }: { children?: ComponentChildren }) => children

export * from "preact/compat"
export { Profiler }
export default { ...compat, Profiler }
