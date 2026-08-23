"""`nl2sh` — shell documentation search, with a generator only if you want one.

    nl2sh "go to my home directory"        search, and generate if a backend is set
    nl2sh search "..."                     search only, whatever the config says
    nl2sh doctor                           what is installed, reachable, and usable
    nl2sh config                           the resolved settings and where each came from

The default backend is `none`, and that is the product rather than a
degraded mode: ranked shell documentation with no model, no network and no
API key. A generator is an upgrade you opt into, local or remote, and the
same measured prompt is used either way.

Nothing here ever executes a command. It prints one.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import backends, config as cfgmod

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "vendor"))


def _prompts():
    import prompts                                          # vendored, hash-pinned
    return prompts


def _search_module():
    """Import the retrieval tier, or explain what is missing.

    Search is the floor of this product, so a failure here is fatal in a way a
    missing backend never is — and it must say which of the two it is.
    """
    try:
        from . import search
    except ImportError as e:
        raise SystemExit(
            f"nl2sh: the retrieval tier could not be imported ({e}).\n"
            "This is the part that works without a model, so nothing will run "
            "until it does.\nTry: pip install -e '.[search]'") from e
    return search


# --------------------------------------------------------------------------
# rendering
# --------------------------------------------------------------------------

def _width() -> int:
    return max(60, min(shutil.get_terminal_size((100, 24)).columns, 120))


def render_hits(hits: list[dict], stream=sys.stdout) -> None:
    w = _width()
    if not hits:
        print("no matching utility found", file=stream)
        return
    for i, h in enumerate(hits, 1):
        score = h.get("score")
        head = f"{i}. {h['utility']}"
        if score is not None:
            head += f"   {score:.3f}"
        print(head, file=stream)
        if h.get("summary"):
            print(f"   {h['summary'][:w - 4]}", file=stream)
        if h.get("example"):
            print(f"   $ {h['example'][:w - 6]}", file=stream)
        print(file=stream)


def render_generation(g: backends.Generation, stream=sys.stdout) -> None:
    if g.command:
        print(g.command, file=stream)
    else:
        note = ("the model filled its token budget without producing a command"
                if g.truncated else "the model returned no command")
        print(f"# {note}", file=stream)
    tokens = f", {g.new_tokens} tokens" if g.new_tokens else ""
    print(f"# {g.backend}:{g.model or '-'}  {g.seconds:.2f}s{tokens}",
          file=sys.stderr)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------

def cmd_search(args, cfg) -> int:
    hits = _search_module().search(args.query, k=args.k or cfg.k)
    if args.json:
        json.dump(hits, sys.stdout, indent=1)
        print()
    else:
        render_hits(hits)
    return 0


def cmd_ask(args, cfg) -> int:
    search = _search_module()
    hits = search.search(args.query, k=args.k or cfg.k)

    if cfg.backend == "none":
        # Not an error and not a fallback — the documented default path.
        render_hits(hits)
        if not args.quiet:
            print("# no model configured; showing documentation only. "
                  "`nl2sh doctor` lists the options.", file=sys.stderr)
        return 0

    backend = backends.build(cfg.backend, **cfg.backend_opts())
    av = backend.probe()
    if not av:
        # The search results are still worth printing — a missing backend
        # degrades this product to its floor, not to nothing.
        render_hits(hits)
        print(f"# backend {cfg.backend!r} unavailable: {av.detail}", file=sys.stderr)
        return 2

    if cfgmod.looks_tiny(cfg.model) and cfg.prompt.startswith("instantiate"):
        print(f"# {cfg.model} looks small, and the instantiation prompt collapses "
              f"below ~1B (0.146 routing at 270M, against 0.500 for "
              f"{cfgmod.TINY_PROMPT}). Consider --prompt {cfgmod.TINY_PROMPT}.",
              file=sys.stderr)

    source_lines = [h["source_line"] for h in hits]
    prompt = _prompts().BUILDERS[cfg.prompt](args.query, source_lines)

    if args.show_prompt:
        print(prompt, file=sys.stderr)
        print("---", file=sys.stderr)

    g = backend.generate(prompt, max_tokens=cfg.max_tokens, temperature=cfg.temperature)

    if args.json:
        json.dump({"query": args.query, "command": g.command, "raw": g.raw,
                   "backend": g.backend, "model": g.model, "seconds": g.seconds,
                   "new_tokens": g.new_tokens, "truncated": g.truncated,
                   "prompt": cfg.prompt, "sources": hits}, sys.stdout, indent=1)
        print()
        return 0 if g.command else 1

    render_generation(g)
    if args.explain:
        print("\n# sources handed to the model:", file=sys.stderr)
        for s in source_lines:
            print(f"#   - {s}", file=sys.stderr)
    return 0 if g.command else 1


def cmd_doctor(args, cfg) -> int:
    print(f"config      {cfgmod.CONFIG_PATH}"
          f"{'' if cfgmod.CONFIG_PATH.exists() else '  (absent)'}")
    print(f"cache       {cfgmod.CACHE_DIR}")
    try:
        search = _search_module()
        print(f"search      {search.status()}")
    except SystemExit as e:
        print(f"search      UNAVAILABLE — {e}")
    print()
    print("backends    (* = the one configured)")
    for name, av in backends.survey({cfg.backend: cfg.model}):
        mark = "*" if name == cfg.backend else " "
        state = "ok  " if av else "--  "
        print(f"  {mark} {state}{name:<14} {av.detail}")
    print()
    print("local:  " + ", ".join(backends.LOCAL))
    print("remote: " + ", ".join(backends.REMOTE))
    return 0


def cmd_config(args, cfg) -> int:
    d = cfg.redacted()
    src = d.pop("sources")
    for k, v in d.items():
        print(f"{k:<14}{str(v):<40}{src.get(k, '')}")
    return 0


# --------------------------------------------------------------------------
# entry
# --------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nl2sh",
        description="Search shell documentation; optionally have a model write the command.",
        epilog="With no backend configured this prints ranked documentation and "
               "nothing else, which is the default and needs no model, network or key.")
    p.add_argument("query", nargs="?", help="what you want to do, in plain words")
    p.add_argument("--backend", choices=sorted(backends.REGISTRY),
                   help="generator to use; 'none' (default) means search only")
    p.add_argument("--model", help="model id, HF repo, or GGUF path, per backend")
    p.add_argument("--base-url", dest="base_url", help="for OpenAI-compatible endpoints")
    p.add_argument("--api-key", dest="api_key", help="prefer the environment variable")
    p.add_argument("--prompt", choices=sorted(_p_names()),
                   help=f"prompt shape (default {cfgmod.DEFAULT_PROMPT})")
    p.add_argument("-k", type=int, help="how many utilities to retrieve")
    p.add_argument("--max-tokens", dest="max_tokens", type=int)
    p.add_argument("--temperature", type=float)
    p.add_argument("--json", action="store_true", help="machine-readable output")
    p.add_argument("--explain", action="store_true", help="show the sources used")
    p.add_argument("--show-prompt", dest="show_prompt", action="store_true")
    p.add_argument("-q", "--quiet", action="store_true")

    sub = p.add_subparsers(dest="cmd")
    s = sub.add_parser("search", help="search only, never generate")
    s.add_argument("query")
    s.add_argument("-k", type=int)
    s.add_argument("--json", action="store_true")
    sub.add_parser("doctor", help="what is installed, reachable and usable")
    sub.add_parser("config", help="resolved settings and where each came from")
    return p


def _p_names():
    try:
        return _prompts().BUILDERS
    except Exception:                                       # noqa: BLE001
        return ["generate", "generate_anchored", "instantiate", "instantiate_anchored"]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg, warnings = cfgmod.resolve(
        backend=getattr(args, "backend", None), model=getattr(args, "model", None),
        base_url=getattr(args, "base_url", None), api_key=getattr(args, "api_key", None),
        prompt=getattr(args, "prompt", None), k=getattr(args, "k", None),
        max_tokens=getattr(args, "max_tokens", None),
        temperature=getattr(args, "temperature", None))
    for w in warnings:
        print(f"# {w}", file=sys.stderr)

    for attr in ("json", "explain", "show_prompt", "quiet", "k"):
        if not hasattr(args, attr):
            setattr(args, attr, None if attr == "k" else False)

    if args.cmd == "doctor":
        return cmd_doctor(args, cfg)
    if args.cmd == "config":
        return cmd_config(args, cfg)
    if args.cmd == "search":
        return cmd_search(args, cfg)
    if not args.query:
        build_parser().print_help()
        return 1
    return cmd_ask(args, cfg)


if __name__ == "__main__":
    raise SystemExit(main())
