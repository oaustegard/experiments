#!/usr/bin/env python3
"""Validate a converted trace file against HALO's own index, store, and sandbox.

Runs the three layers a HALO run would exercise, in order, and prints what each
reports — without calling any LLM. That is the whole point: it establishes that
the engine's *tooling* accepts Claude Code trace data, which is separable from
whether the engine's RLM can run (it needs an OpenAI-compatible key).

Usage:
    python validate_dataset.py cc_traces.jsonl [--index-dir DIR]

Requires the engine on sys.path (a checkout of context-labs/halo) or installed
via ``pip install halo-engine``.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

from engine.sandbox.sandbox import Sandbox
from engine.traces.models.trace_dataset_source import TraceDatasetSource
from engine.traces.models.trace_index_config import TraceIndexConfig
from engine.traces.models.trace_query_models import TraceFilters
from engine.traces.trace_index_builder import TraceIndexBuilder
from engine.traces.trace_store import TraceStore

# Runs inside the Deno+Pyodide sandbox. ``trace_store`` is the global the
# runtime's halo_bootstrap binds; TraceFilters has to be imported in-sandbox.
SANDBOX_CODE = r'''
from engine.traces.models.trace_query_models import TraceFilters

tid = trace_store.query_traces(TraceFilters(), limit=1).traces[0].trace_id
view = trace_store.view_trace(tid)
print("view_trace spans:", len(view.spans), "oversized:", view.oversized is not None)
if view.oversized:
    o = view.oversized
    print(f"  {o.span_count} spans, {o.truncated_response_bytes:,}B "
          f"> {o.response_bytes_budget:,}B budget")
    print("  top span names:", o.top_span_names[:5])
hits = trace_store.search_trace(tid, regex_pattern=r'"tool\.name"')
print("search_trace tool-span matches:", len(hits.matches))
'''


async def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("traces", type=Path)
    ap.add_argument("--index-dir", type=Path, default=Path("./halo-index"))
    args = ap.parse_args()

    trace_path = args.traces.resolve()
    args.index_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    index_path = await TraceIndexBuilder.ensure_index_exists(
        trace_path, config=TraceIndexConfig(index_dir=args.index_dir)
    )
    print(f"[index]  {index_path.name} ({index_path.stat().st_size:,}B) "
          f"in {time.time() - t0:.2f}s")

    source = TraceDatasetSource(trace_path=trace_path, index_path=index_path)
    store = TraceStore.load_many([source])
    print(f"[store]  trace_count={store.trace_count}")
    overview = store.get_overview(TraceFilters())
    print("[store]  get_overview():")
    print(json.dumps(json.loads(overview.model_dump_json()), indent=2))

    sandbox = Sandbox.get()
    if sandbox is None:
        print("[sandbox] unavailable (no deno on PATH)", file=sys.stderr)
        return
    t1 = time.time()
    result = await sandbox.run_python(code=SANDBOX_CODE, sources=[source])
    print(f"[sandbox] run_python in {time.time() - t1:.1f}s")
    print(result.stdout, end="")
    if result.stderr:
        print("[sandbox] stderr:", result.stderr[:1000], file=sys.stderr)


if __name__ == "__main__":
    asyncio.run(main())
