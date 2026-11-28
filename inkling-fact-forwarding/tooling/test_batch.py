#!/usr/bin/env python3
"""Live test for inkling_batch_mcp.py. Spawns the MCP server over stdio and drives the real lane-batched engine.

Needs: erisinky built from upstream with erisinky-batch-json.patch applied (tools/erisinky-batch), the Q4_K_M model,
and a quiet GPU (stop other model processes first; a busy GPU only slows the run, it does not fail it).
Uses a throwaway workspace so the file-input checks cannot touch real files. Exits non-zero on any failure.
Must run outside the Claude Code Bash sandbox (Metal).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SERVER = os.path.join(HERE, "inkling_batch_mcp.py")
REPO = os.path.dirname(os.path.dirname(HERE))
BIN = os.environ.get("INKLING_BATCH_BIN") or os.path.join(REPO, "tools", "erisinky-batch", "target", "release", "inkling")
FAILS: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"  [{detail}]" if detail and not ok else ""))
    if not ok:
        FAILS.append(name)


class Client:
    def __init__(self, env: dict[str, str]):
        self.p = subprocess.Popen([sys.executable, SERVER], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, env={**os.environ, **env})
        self.n = 0

    def rpc(self, method: str, params: dict | None = None) -> dict:
        self.n += 1
        self.p.stdin.write(json.dumps({"jsonrpc": "2.0", "id": self.n, "method": method, "params": params or {}}) + "\n")
        self.p.stdin.flush()
        return json.loads(self.p.stdout.readline())

    def call(self, name: str, **args) -> tuple[object, dict | None, bool, str]:
        r = self.rpc("tools/call", {"name": name, "arguments": args})["result"]
        blocks = [b["text"] for b in r["content"]]
        is_err = bool(r.get("isError"))
        if is_err:
            return None, None, True, blocks[0]
        body = json.loads(blocks[0])
        meta = json.loads(blocks[1]) if len(blocks) > 1 else None
        return body, meta, False, blocks[0]

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait(timeout=30)


def leftover_engines() -> list[str]:
    out = subprocess.run(["ps", "-axo", "pid=,command="], capture_output=True, text=True).stdout
    return [l for l in out.splitlines() if "inkling batch-json" in l and "ps -axo" not in l]


def main() -> int:
    if not os.path.isfile(BIN):
        print(f"engine binary not found: {BIN}\nSet INKLING_BATCH_BIN to the patched erisinky `inkling` binary.")
        return 2
    ws = tempfile.mkdtemp(prefix="inkling-batch-test-")
    with open(os.path.join(ws, "notes.txt"), "w") as f:
        f.write("hello\n")
    with open(os.path.join(ws, ".env"), "w") as f:
        f.write("SECRET=1\n")
    c = Client({"INKLING_WORKSPACE": ws, "INKLING_BATCH_BIN": BIN, "INKLING_BATCH_CTX_CAP": "8192"})
    try:
        print("protocol")
        init = c.rpc("initialize", {"protocolVersion": "2025-06-18"})["result"]
        check("initialize names the server", init["serverInfo"]["name"] == "inkling-batch")
        tools = [t["name"] for t in c.rpc("tools/list")["result"]["tools"]]
        check("four tools listed", tools == ["batch_map", "batch_samples", "batch_split_ask", "batch_status"], str(tools))

        print("status")
        st, _, err, _ = c.call("batch_status")
        check("status ok, engine and model exist", not err and st["engine_exists"] and st["model_exists"], str(st))
        check("engine supports batch-json", st.get("supports_batch_json") is True, str(st.get("supports_batch_json")))
        check("Low Power Mode is off", st["power"].get("low_power") is False, str(st["power"]))
        if st["busy_processes"]:
            print("  note  other GPU-heavy processes running; speeds below will be low:", st["busy_processes"])
            deadline = time.time() + 240
            while st["busy_processes"] and time.time() < deadline:
                time.sleep(10)
                st, _, _, _ = c.call("batch_status")
            print("  note  still busy:" if st["busy_processes"] else "  note  GPU quiet now", st["busy_processes"])

        print("validation")
        for label, kw, frag in [
            ("empty prompts", dict(prompts=[]), "non-empty"),
            ("blank prompt", dict(prompts=["ok", "  "]), "non-empty"),
            ("too many prompts", dict(prompts=["x"] * 300), "at most"),
            ("greedy refused", dict(prompts=["x"], temperature=0), "greedy"),
            ("top_k over 64", dict(prompts=["x"], top_k=65), "top_k"),
            ("bad thinking", dict(prompts=["x"], thinking="max"), "thinking"),
            ("prompt over context cap", dict(prompts=["word " * 12000]), "cap"),
        ]:
            _, _, err, msg = c.call("batch_map", **kw)
            check(f"batch_map rejects: {label}", err and frag in msg, msg[:150])
        _, _, err, msg = c.call("batch_samples", prompt="x", n=1)
        check("batch_samples rejects n=1", err and "n must be" in msg, msg[:150])
        _, _, err, msg = c.call("batch_samples", prompt="x", answer_regex="(")
        check("batch_samples rejects bad regex", err and "regex" in msg, msg[:150])
        _, _, err, msg = c.call("nope")
        check("unknown tool is an error", err and "unknown tool" in msg)

        print("batch_map")
        prompts = [
            "What is 17 times 23? Answer with only the number.",
            "Reply with only the word that is the opposite of 'hot'.",
            "Here are two lines:\nalpha\nbeta\nReply with only the second line, exactly.",
            'Copy the text between the double brackets exactly and output only that text.\n[[a\\b "c" d]]',
            "What is the capital of France? Answer with one word.",
            "What is 9 plus 8? Answer with only the number.",
        ]
        body, meta, err, raw = c.call("batch_map", prompts=prompts, max_tokens=60, lanes=6)
        check("batch_map ok", not err, raw[:200])
        if not err:
            check("six results in order", [r["i"] for r in body] == list(range(6)), str([r["i"] for r in body]))
            check("17 x 23 = 391", "391" in body[0]["content"], body[0]["content"])
            check("opposite of hot", "cold" in body[1]["content"].lower(), body[1]["content"])
            check("newline prompt reached the model", "beta" in body[2]["content"].lower(), body[2]["content"])
            check("backslash and quotes survived the escaping", 'a\\b "c" d' in body[3]["content"], body[3]["content"])
            check("Paris", "paris" in body[4]["content"].lower(), body[4]["content"])
            check("9 + 8 = 17", "17" in body[5]["content"], body[5]["content"])
            check("thinking off: no reasoning tokens", all(not r.get("reasoning_tokens") for r in body), str(body[0]))
            check("meta has throughput and lanes", meta["lanes_used"] == 6 and meta["decode_tok_s_aggregate"] > 0, str(meta))
            print(f"  info  6 lanes: {meta['decode_tok_s_aggregate']} tok/s aggregate decode, "
                  f"prefill {meta['prefill_tok_s']} tok/s, wall {meta['wall_s']} s")

        body, meta, err, raw = c.call("batch_map", prompts=["Say hi.", "word " * 2000 + "\nSay hi."], max_tokens=20,
                                      lanes=2, include_reasoning=True, thinking="low")
        check("thinking low returns reasoning when asked", not err and "reasoning" in body[0], raw[:200])
        body, meta, err, raw = c.call("batch_map", prompts=["Write a long essay about rivers."], max_tokens=16)
        check("length finish is flagged", not err and body[0]["finish"] == "length" and "note" in body[0], raw[:200])

        print("batch_samples")
        body, meta, err, raw = c.call("batch_samples", prompt="What is 12 times 13? End with a final line 'ANSWER: <number>'.",
                                      n=6, tally=True, max_tokens=120, seed=3)
        check("batch_samples ok", not err, raw[:200])
        if not err:
            check("six samples, one shared prefill", len(body["samples"]) == 6 and len({s["prompt_tokens"] for s in body["samples"]}) == 1)
            tal = body["tally"]
            check("majority answer is 156", tal["majority"] is not None and "156" in tal["majority"], json.dumps(tal)[:300])
            print(f"  info  votes {tal['votes']}, unparsed {tal['unparsed']}, {meta['decode_tok_s_aggregate']} tok/s")
        body, meta, err, raw = c.call("batch_samples", prompt="Name a color.", n=4, answer_regex=r"(\w+)", max_tokens=12)
        check("answer_regex tally works", not err and "tally" in body, raw[:200])

        print("batch_split_ask (experimental)")
        paras = []
        filler = "The committee reviewed routine maintenance schedules and noted nothing unusual about the building. "
        for i in range(12):
            if i == 1:
                paras.append("Security note: the vault access code is printed on the card as 4417. " + filler * 3)
            elif i == 6:
                paras.append("Security note: the card must be read backwards before use, so reverse every digit. " + filler * 3)
            else:
                paras.append(filler * 4)
        doc = "\n\n".join(paras)
        body, meta, err, raw = c.call("batch_split_ask", question="What code should be entered into the vault, after any required transformation?",
                                      text=doc, shards=4, answers=3, thinking="low", max_tokens=700, fact_tokens=150)
        check("split_ask ok", not err, raw[:300])
        if not err:
            check("four shards, each with facts text", len(body["shards"]) == 4 and all(s["facts"] for s in body["shards"]))
            pooled = " ".join(s["facts"] for s in body["shards"])
            check("round 1 extracted the code fact", "4417" in pooled, pooled[:300])
            check("round 1 extracted the reverse-it fact", "revers" in pooled.lower() or "backward" in pooled.lower(), pooled[:300])
            check("reports votes and caveat", "votes" in body and "NOT established" in body["experimental"])
            print(f"  info  answer {body['answer']!r} (expected 7144), votes {body['votes']}, wall {meta['wall_s']} s "
                  f"(r1 {meta['round1_extract']['wall_s']} s, r2 {meta['round2_answer']['wall_s']} s)")
        _, _, err, msg = c.call("batch_split_ask", question="q", text="short text")
        check("split_ask refuses text too short to split", err and "too short" in msg, msg[:150])
        _, _, err, msg = c.call("batch_split_ask", question="q")
        check("split_ask needs text or file", err and "exactly one" in msg, msg[:150])
        _, _, err, msg = c.call("batch_split_ask", question="q", file="../../etc/hosts")
        check("file outside workspace refused", err and "inside the workspace" in msg, msg[:150])
        _, _, err, msg = c.call("batch_split_ask", question="q", file=".env")
        check("secret-named file refused", err and "secret" in msg, msg[:150])
        _, _, err, msg = c.call("batch_split_ask", question="q", file="notes.txt")
        check("tiny file reaches the 'too short' check", err and "too short" in msg, msg[:150])

        print("deadline")
        t0 = time.time()
        _, _, err, msg = c.call("batch_map", prompts=["Write a very long story."], max_tokens=4000, deadline_s=1.5)
        check("deadline stops the batch with a clear error", err and "did not finish" in msg, msg[:200])
        check("deadline returned promptly", time.time() - t0 < 20, f"{time.time() - t0:.1f}s")
        time.sleep(1.0)
        check("no engine process left behind", not leftover_engines(), str(leftover_engines()))
    finally:
        c.close()

    print("\nFAILED: " + ", ".join(FAILS) if FAILS else "\nall passed")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
