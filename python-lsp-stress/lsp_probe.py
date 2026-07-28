#!/usr/bin/env python3
"""Minimal LSP (JSON-RPC over stdio) client to stress-test pyright-langserver.

Drives pyright-langserver against a real codebase and exercises the protocol
surface a Claude session would actually use: initialize, document symbols,
hover, definition, references, workspace symbol search, and live diagnostics.

Usage: python3 lsp_probe.py <workspace_root> <relative_target.py>
"""
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

SERVER = os.environ.get("PYRIGHT_LANGSERVER", "/opt/node22/bin/pyright-langserver")


class LSPClient:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.proc = subprocess.Popen(
            [SERVER, "--stdio"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._id = 0
        self._responses = {}
        self._notifications = []
        self._lock = threading.Lock()
        self._reader = threading.Thread(target=self._read_loop, daemon=True)
        self._reader.start()

    # ---- wire protocol ----
    def _send(self, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        header = f"Content-Length: {len(body)}\r\n\r\n".encode("utf-8")
        self.proc.stdin.write(header + body)
        self.proc.stdin.flush()

    def _read_loop(self):
        stream = self.proc.stdout
        while True:
            # read headers
            line = stream.readline()
            if not line:
                return
            if not line.startswith(b"Content-Length:"):
                continue
            length = int(line.split(b":")[1].strip())
            stream.readline()  # blank line
            body = stream.read(length)
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            with self._lock:
                if "id" in msg and ("result" in msg or "error" in msg):
                    self._responses[msg["id"]] = msg
                else:
                    self._notifications.append(msg)

    def request(self, method: str, params: dict, timeout=120):
        with self._lock:
            self._id += 1
            rid = self._id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        start = time.time()
        while time.time() - start < timeout:
            with self._lock:
                if rid in self._responses:
                    return self._responses.pop(rid), time.time() - start
            time.sleep(0.005)
        raise TimeoutError(f"{method} timed out after {timeout}s")

    def notify(self, method: str, params: dict):
        self._send({"jsonrpc": "2.0", "method": method, "params": params})

    def drain_diagnostics(self, uri: str, settle=8.0):
        """Wait until publishDiagnostics for uri stops changing."""
        start = time.time()
        last = None
        last_change = time.time()
        while time.time() - start < settle:
            with self._lock:
                diags = [n for n in self._notifications
                         if n.get("method") == "textDocument/publishDiagnostics"
                         and n.get("params", {}).get("uri") == uri]
            current = diags[-1]["params"]["diagnostics"] if diags else None
            if current != last:
                last = current
                last_change = time.time()
            elif time.time() - last_change > 1.5:
                break
            time.sleep(0.1)
        return last or []

    def shutdown(self):
        try:
            self.request("shutdown", {}, timeout=10)
            self.notify("exit", {})
        except Exception:
            pass
        self.proc.terminate()


def uri_of(p: Path) -> str:
    return "file://" + str(p.resolve())


def find_symbol_position(symbols, kind_pref=(12, 5, 6)):
    """Pick a useful symbol (function=12, class=5, method=6) from documentSymbol."""
    def walk(nodes):
        for n in nodes:
            yield n
            yield from walk(n.get("children", []))
    flat = list(walk(symbols))
    for kind in kind_pref:
        for n in flat:
            if n.get("kind") == kind:
                rng = n.get("selectionRange") or n.get("range")
                return n["name"], rng["start"]
    if flat:
        rng = flat[0].get("selectionRange") or flat[0].get("range")
        return flat[0]["name"], rng["start"]
    return None, None


def main():
    root = Path(sys.argv[1])
    target = root / sys.argv[2]
    if not target.exists():
        print(f"target not found: {target}", file=sys.stderr)
        sys.exit(1)

    results = {"server": SERVER, "root": str(root), "target": str(target)}
    c = LSPClient(root)

    # initialize
    init_params = {
        "processId": os.getpid(),
        "rootUri": uri_of(root),
        "capabilities": {
            "textDocument": {
                "hover": {"contentFormat": ["markdown", "plaintext"]},
                "definition": {"linkSupport": True},
                "references": {},
                "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
            },
            "workspace": {"symbol": {}},
        },
        "initializationOptions": {},
        "workspaceFolders": [{"uri": uri_of(root), "name": root.name}],
    }
    resp, dt = c.request("initialize", init_params)
    results["initialize_s"] = round(dt, 3)
    results["server_caps"] = sorted(resp["result"]["capabilities"].keys())
    c.notify("initialized", {})

    # open target document
    text = target.read_text(encoding="utf-8", errors="replace")
    turi = uri_of(target)
    c.notify("textDocument/didOpen", {
        "textDocument": {"uri": turi, "languageId": "python",
                         "version": 1, "text": text},
    })

    # diagnostics (cold; includes workspace indexing warmup)
    t0 = time.time()
    diags = c.drain_diagnostics(turi, settle=60)
    results["diagnostics_settle_s"] = round(time.time() - t0, 3)
    results["diagnostics_count"] = len(diags)
    results["diagnostics_sample"] = [
        {"line": d["range"]["start"]["line"] + 1,
         "sev": d.get("severity"), "msg": d.get("message", "")[:120]}
        for d in diags[:5]
    ]

    # document symbols
    resp, dt = c.request("textDocument/documentSymbol", {"textDocument": {"uri": turi}})
    syms = resp.get("result") or []
    results["documentSymbol_s"] = round(dt, 3)
    results["documentSymbol_count"] = len(syms)
    # optional argv[3] = symbol name to probe (else auto-pick)
    want = sys.argv[3] if len(sys.argv) > 3 else None
    name, pos = None, None
    if want:
        def walk(nodes):
            for n in nodes:
                yield n
                yield from walk(n.get("children", []))
        for n in walk(syms):
            if n.get("name") == want:
                rng = n.get("selectionRange") or n.get("range")
                name, pos = n["name"], rng["start"]
                break
    if pos is None:
        name, pos = find_symbol_position(syms)
    results["probe_symbol"] = name

    if pos:
        # hover
        resp, dt = c.request("textDocument/hover", {
            "textDocument": {"uri": turi}, "position": pos})
        results["hover_s"] = round(dt, 3)
        hov = resp.get("result")
        if hov and hov.get("contents"):
            cont = hov["contents"]
            val = cont.get("value") if isinstance(cont, dict) else str(cont)
            results["hover_sample"] = (val or "")[:200]

        # definition
        resp, dt = c.request("textDocument/definition", {
            "textDocument": {"uri": turi}, "position": pos})
        results["definition_s"] = round(dt, 3)
        defs = resp.get("result") or []
        results["definition_count"] = len(defs) if isinstance(defs, list) else 1

        # references (the expensive, whole-workspace query)
        resp, dt = c.request("textDocument/references", {
            "textDocument": {"uri": turi}, "position": pos,
            "context": {"includeDeclaration": True}}, timeout=180)
        results["references_s"] = round(dt, 3)
        refs = resp.get("result") or []
        results["references_count"] = len(refs)

    # workspace symbol search (indexes the whole tree)
    try:
        resp, dt = c.request("workspace/symbol", {"query": "Model"}, timeout=180)
        results["workspaceSymbol_s"] = round(dt, 3)
        wsyms = resp.get("result") or []
        results["workspaceSymbol_count"] = len(wsyms)
    except TimeoutError:
        results["workspaceSymbol_s"] = "timeout"

    c.shutdown()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
