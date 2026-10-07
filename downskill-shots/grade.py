"""Grade the ablation runs from their Agent-tool transcripts.

    python3 grade.py runs.json > results.json
runs.json maps "<task>-<arm>-r<k>" to a subagent transcript path. The reply graded is
the subagent's last assistant text or SubagentHandback message.
"""
import json
import re
import sqlite3
import sys
from pathlib import Path

import duckdb

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import fixture as F  # noqa: E402

DB = HERE / "shop.db"


def last_text(transcript: str) -> str:
    text = ""
    for line in open(transcript, errors="replace"):
        try:
            m = json.loads(line).get("message")
        except ValueError:
            continue
        if isinstance(m, dict) and m.get("role") == "assistant":
            for b in m.get("content", []):
                if not isinstance(b, dict):
                    continue
                # A subagent's final reply arrives as plain text or as its SubagentHandback message.
                if b.get("type") == "text" and b.get("text", "").strip():
                    text = b["text"]
                elif b.get("type") == "tool_use" and b.get("name") == "SubagentHandback":
                    text = (b.get("input") or {}).get("message", "") or text
    return text


def parse_array(text: str):
    m = re.search(r"\[.*\]", text, re.S)
    if not m:
        return None
    try:
        return {int(r["id"]): r for r in json.loads(m.group(0))}
    except (ValueError, KeyError, TypeError):
        return None


def match(val, gold) -> bool:
    return str(val).upper() in (gold if isinstance(gold, list) else [gold])


def grade_labels(items, keys, reply: str):
    rows = parse_array(reply)
    out = {}
    for i, _, gold, kind, _ in items:
        r = (rows or {}).get(i, {})
        out[i] = {"kind": kind, "ok": bool(r) and all(match(r.get(k), g) for k, g in zip(keys, gold)),
                  "got": [r.get(k) for k in keys]}
    return out, rows is not None


def duck():
    c = duckdb.connect()
    c.execute(f"ATTACH '{DB}' AS s (TYPE sqlite, READ_ONLY)")
    for t in ("customers", "orders", "products", "order_items"):
        c.execute(f"CREATE TABLE {t} AS SELECT * FROM s.{t}")
    c.execute("ALTER TABLE orders ALTER order_date TYPE DATE")
    c.execute("ALTER TABLE customers ALTER signup_date TYPE DATE")
    return c


def norm(v):
    if isinstance(v, float):
        return round(v, 2)
    if hasattr(v, "month") and hasattr(v, "year"):
        return v.month
    if isinstance(v, str) and v.isdigit():
        return int(v)
    return v


def sql_ok(qid: int, gold, chunk: str, con, lite) -> tuple[bool, str]:
    m = re.search(r"```sql\s*(.*?)```", chunk, re.S)
    body = m.group(1) if m else chunk
    if gold is None:
        return "cannot answer" in chunk.lower(), "cannot"
    if "cannot answer" in body.lower():
        return False, "refused"
    try:
        got = con.execute(body.strip().rstrip(";")).fetchall()
    except Exception as e:  # any engine error scores as wrong, reported verbatim
        return False, f"error: {str(e)[:80]}"
    cols = list(zip(*got)) if got else []
    if gold.startswith("LIMIT10:"):
        norway = {r[0] for r in lite.execute("SELECT name FROM customers WHERE country='Norway'")}
        ok = len(got) == 10 and any(set(c) <= norway for c in cols)
        return ok, f"{len(got)} rows"
    want = lite.execute(gold).fetchall()
    if len(got) != len(want):
        return False, f"{len(got)} rows, want {len(want)}"
    gcols = [sorted(map(norm, c), key=str) for c in zip(*want)]
    mcols = [sorted(map(norm, c), key=str) for c in cols]
    return all(g in mcols for g in gcols), "compared"


def grade_sql(reply: str):
    con, lite = duck(), sqlite3.connect(DB)
    parts = re.split(r"(?m)^[ \t*#>_]*Q(\d)\b", reply)
    chunks = {int(parts[k]): parts[k + 1] for k in range(1, len(parts) - 1, 2)}
    out = {}
    for i, _, gold, kind, _ in F.SQL:
        ok, why = sql_ok(i, gold, chunks.get(i, ""), con, lite) if i in chunks else (False, "missing")
        out[i] = {"kind": kind, "ok": ok, "got": why}
    return out, bool(chunks)


def main() -> None:
    runs = json.load(open(sys.argv[1]))
    results = {}
    for label, path in sorted(runs.items()):
        task = label.split("-")[0]
        reply = last_text(path)
        if task == "feedback":
            items, parsed = grade_labels(F.FEEDBACK, ("sentiment", "theme"), reply)
        elif task == "moderation":
            items, parsed = grade_labels(F.MODERATION, ("verdict", "policy"), reply)
        else:
            items, parsed = grade_sql(reply)
        results[label] = {"parsed": parsed, "items": items}
    json.dump(results, sys.stdout, indent=1)


if __name__ == "__main__":
    main()
