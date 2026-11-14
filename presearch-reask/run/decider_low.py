"""Test 1b: the hub decider (scripts/decide.py) on its 104-item eval at --effort low.

Counts samples that fail outright (transport error) or parse to nothing, which is
what a re-ask would recover. Writes results/decider_low.json.

    python3 run/decider_low.py
"""
import json
import sys
from pathlib import Path

HUB = Path(__file__).resolve().parents[3] / "claude-workspace"
sys.path.insert(0, str(HUB / "scripts"))
import decide  # noqa: E402

HERE = Path(__file__).resolve().parents[1]
log = []

_argv = decide.claude_argv
decide.claude_argv = lambda s, u: _argv(s, u)[:-1] + ["--effort", "low", u]

_transport = decide.claude_transport


def transport(system, user):
    try:
        r = _transport(system, user)
    except decide.DeciderError as e:
        log.append({"status": "transport_error", "detail": str(e)[:200]})
        raise
    log.append({"status": "ok", "reply": r if isinstance(r, str) else None})
    return r


decide.claude_transport = transport
_parse = decide.parse_reply


def parse(reply, qs):
    p = _parse(reply, qs)
    if p is None:
        log.append({"status": "unparseable", "reply": str(reply)[:300]})
    elif any(v is None for v in p.values()):
        log.append({"status": "partial", "missing": [k for k, v in p.items() if v is None]})
    return p


decide.parse_reply = parse


def main():
    items = [json.loads(l) for l in (HUB / "tests" / "data" / "decider_eval.jsonl").open()]
    rows = decide.evaluate(items, k=3, backend="claude")
    out = {"report": decide.report(rows),
           "samples": sum(e["status"] in ("ok", "transport_error") for e in log),
           "transport_errors": sum(e["status"] == "transport_error" for e in log),
           "unparseable": sum(e["status"] == "unparseable" for e in log),
           "partial": sum(e["status"] == "partial" for e in log),
           "items_with_no_answer": sum(1 for r in rows if r.get("error")),
           "failures": [e for e in log if e["status"] != "ok"]}
    (HERE / "results" / "decider_low.json").write_text(json.dumps(out, indent=1))
    (HERE / "results" / "decider_low_rows.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    print(json.dumps({k: v for k, v in out.items() if k != "failures"}, indent=1))


if __name__ == "__main__":
    main()
