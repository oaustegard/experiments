#!/usr/bin/env python3
"""Minimal inter-session relay over the shared Turso DB.

Lets two (or more) Claude sessions converse by appending to and polling a
shared message table. No service to deploy: the rendezvous is the Turso DB
every Muninn session already has credentials for (TURSO_URL / TURSO_TOKEN).

Inspired by https://bsky.app/profile/joshuashew.bsky.social/post/3mnx2urp3f22n
(one Claude session built a relay service so two others could coordinate).

Usage:
    relay.py init                                   # create table + index
    relay.py post CHANNEL SENDER BODY               # BODY of '-' reads stdin
    relay.py poll CHANNEL [--after N] [--not-sender NAME]
    relay.py wait CHANNEL [--after N] [--not-sender NAME]
                          [--timeout 120] [--interval 4]
    relay.py history CHANNEL                        # full transcript, pretty

post prints the assigned seq. poll/wait print one JSON object per line:
{"seq": ..., "channel": ..., "sender": ..., "body": ..., "created_at": ...}
Callers keep their own cursor (the max seq they have seen) and pass it back
via --after.
"""

import argparse
import json
import os
import sys
import time

import requests

_raw_url = os.environ["TURSO_URL"].rstrip("/")
if _raw_url.startswith("libsql://"):
    _raw_url = "https://" + _raw_url[len("libsql://"):]
elif "://" not in _raw_url:
    _raw_url = "https://" + _raw_url
URL = _raw_url
HEADERS = {"Authorization": f"Bearer {os.environ['TURSO_TOKEN']}"}


def exec_sql(sql, args=None):
    """Execute one statement via the Turso HTTP pipeline; return list of dicts.

    Turso cold starts surface as 5xx from the egress proxy — retry with
    backoff (ops: proxy-503-retry-pattern) rather than failing the first call.
    """
    stmt = {"sql": sql}
    if args is not None:
        stmt["args"] = [
            {"type": "text", "value": str(v)} if v is not None else {"type": "null"}
            for v in args
        ]
    last_err = None
    for attempt in range(5):
        try:
            resp = requests.post(
                f"{URL}/v2/pipeline",
                headers=HEADERS,
                json={"requests": [{"type": "execute", "stmt": stmt}]},
                timeout=30,
            )
        except requests.RequestException as e:
            last_err = str(e)
            time.sleep(2 ** attempt)
            continue
        if resp.status_code >= 500:
            last_err = f"HTTP {resp.status_code}: {resp.text[:200]}"
            time.sleep(2 ** attempt)
            continue
        data = resp.json()
        if "results" not in data:
            raise RuntimeError(f"Turso error: {data.get('error', data)}")
        result = data["results"][0]
        if result.get("type") == "error":
            raise RuntimeError(f"SQL error: {result['error']['message']}")
        r = result["response"]["result"]
        cols = [c["name"] for c in r["cols"]]
        return [
            {col: cell.get("value") for col, cell in zip(cols, row)}
            for row in r["rows"]
        ]
    raise RuntimeError(f"Turso unreachable after 5 attempts: {last_err}")


def cmd_init(_args):
    exec_sql(
        "CREATE TABLE IF NOT EXISTS relay_messages ("
        " seq INTEGER PRIMARY KEY AUTOINCREMENT,"
        " channel TEXT NOT NULL,"
        " sender TEXT NOT NULL,"
        " body TEXT NOT NULL,"
        " created_at TEXT DEFAULT (datetime('now')))"
    )
    exec_sql(
        "CREATE INDEX IF NOT EXISTS idx_relay_chan"
        " ON relay_messages(channel, seq)"
    )
    print("ok: relay_messages ready")


def cmd_post(args):
    body = sys.stdin.read() if args.body == "-" else args.body
    rows = exec_sql(
        "INSERT INTO relay_messages (channel, sender, body) VALUES (?, ?, ?)"
        " RETURNING seq",
        [args.channel, args.sender, body],
    )
    print(rows[0]["seq"])


def _fetch(channel, after, not_sender):
    sql = "SELECT seq, channel, sender, body, created_at FROM relay_messages WHERE channel = ? AND seq > ?"
    params = [channel, after]
    if not_sender:
        sql += " AND sender != ?"
        params.append(not_sender)
    return exec_sql(sql + " ORDER BY seq", params)


def _print_rows(rows):
    for row in rows:
        row["seq"] = int(row["seq"])
        print(json.dumps(row))


def cmd_poll(args):
    _print_rows(_fetch(args.channel, args.after, args.not_sender))


def cmd_wait(args):
    deadline = time.monotonic() + args.timeout
    while True:
        rows = _fetch(args.channel, args.after, args.not_sender)
        if rows:
            _print_rows(rows)
            return
        if time.monotonic() >= deadline:
            print(f"timeout: no new messages on {args.channel} after seq {args.after}",
                  file=sys.stderr)
            sys.exit(2)
        time.sleep(args.interval)


def cmd_history(args):
    for row in _fetch(args.channel, 0, None):
        print(f"[{row['seq']:>4}] {row['created_at']} {row['sender']}:")
        print(row["body"])
        print()


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("init").set_defaults(func=cmd_init)

    sp = sub.add_parser("post")
    sp.add_argument("channel")
    sp.add_argument("sender")
    sp.add_argument("body", help="message text, or '-' to read stdin")
    sp.set_defaults(func=cmd_post)

    for name in ("poll", "wait"):
        sp = sub.add_parser(name)
        sp.add_argument("channel")
        sp.add_argument("--after", type=int, default=0, help="cursor: last seen seq")
        sp.add_argument("--not-sender", default=None, help="exclude own messages")
        if name == "wait":
            sp.add_argument("--timeout", type=float, default=120)
            sp.add_argument("--interval", type=float, default=4)
        sp.set_defaults(func=cmd_wait if name == "wait" else cmd_poll)

    sp = sub.add_parser("history")
    sp.add_argument("channel")
    sp.set_defaults(func=cmd_history)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
