// Three places where Preact renders the React adapter's output differently
// from React. Each was found by the ui-sdk Playwright suite (see RESULTS.md).
// The hooks use Preact 11's mangled option and vnode names: `__` is the root
// hook, `__k` a container's tree, `__m` the useId mask, `__u` the flags and
// `__e` a vnode's element.
import { options } from "preact"
import "preact/hooks"

// 1. useId restarts at P0-0 in every root, so a page with several roots
// (one per widget, as zero-build pages often do) repeats ids, and a trigger's
// aria-controls finds another root's panel. Number each container on its
// first client render; the first one keeps the stock P0- prefix, so a
// single-root page and a hydrated one render the ids Preact always did.
const masks = new WeakMap()
let roots = 0
const previousRoot = options.__
options.__ = (vnode, container) => {
  if (vnode && typeof vnode === "object" && container) {
    let mask = masks.get(container)
    if (!mask && !(vnode.__u & 32) && !container.__k) {
      mask = [roots++, 0]
      masks.set(container, mask)
    }
    if (mask && !vnode.__m) {
      vnode.__m = mask
    }
  }
  if (previousRoot) {
    previousRoot(vnode, container)
  }
}

// 2. When a prop goes away, Preact sets a reflected property to "" rather
// than removing the attribute, so a part rendered as a link keeps type="".
// 3. Core writes attribute values: "" means present. Preact assigns a boolean
// property like webkitdirectory, where "" means false.
// Both are corrected after the diff, on ui-sdk parts only (they all carry
// data-slot), and only where the element ended up differing from React.
const rendered = new WeakMap()
const skip = new Set(["children", "key", "ref", "style", "dangerouslySetInnerHTML"])
const attrName = (prop) =>
  prop === "className" ? "class" : prop === "htmlFor" ? "for" : prop
const previousDiffed = options.diffed
options.diffed = (vnode) => {
  const el = vnode.__e
  const props = vnode.props
  if (
    typeof vnode.type === "string" &&
    el &&
    el.nodeType === 1 &&
    el.namespaceURI === "http://www.w3.org/1999/xhtml" &&
    props &&
    (props["data-slot"] != null || rendered.has(el))
  ) {
    const before = rendered.get(el)
    if (before) {
      for (const prop in before) {
        if (
          props[prop] == null &&
          before[prop] != null &&
          !skip.has(prop) &&
          !(prop[0] === "o" && prop[1] === "n") &&
          el.getAttribute(attrName(prop)) === ""
        ) {
          el.removeAttribute(attrName(prop))
        }
      }
    }
    for (const prop in props) {
      if (props[prop] === "" && typeof el[prop] === "boolean" && !el[prop]) {
        el.setAttribute(prop.toLowerCase(), "")
      }
    }
    if (props["data-slot"] != null) {
      rendered.set(el, props)
    } else {
      rendered.delete(el)
    }
  }
  if (previousDiffed) {
    previousDiffed(vnode)
  }
}
