#!/usr/bin/env python3
"""Print what each request of a session ran at. Usage: analyze.py <session_id | transcript.jsonl> [...]
One line per assistant record: effort, perTurnEffort, model, usage, and what it did; plus any
record that mentions hook context, so delivery of additionalContext is visible."""
import glob, json, os, sys

def find(sid):
    if os.path.exists(sid):
        return sid
    hits = glob.glob(os.path.expanduser(f"~/.claude/projects/*/{sid}.jsonl"))
    return hits[0] if hits else None

def show(path):
    print(f"== {path}")
    seen = set()
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:
            continue
        t = r.get("type")
        if t == "assistant":
            m = r.get("message", {})
            if m.get("id") in seen:  # one API response can span several records
                tag = "  ..same request"
            else:
                seen.add(m.get("id")); tag = ""
            u = m.get("usage", {})
            did = []
            for b in m.get("content", []):
                if b.get("type") == "tool_use":
                    i = b.get("input", {})
                    did.append(f"tool:{b['name']}({str(i.get('skill') or i.get('command') or i.get('subagent_type') or i.get('pattern') or '')[:50]})")
                elif b.get("type") == "text":
                    did.append("text:" + b["text"][:90].replace("\n", " "))
                else:
                    did.append(b.get("type"))
            print(f"A effort={r.get('effort')} perTurn={r.get('perTurnEffort')} model={m.get('model')} "
                  f"in={u.get('input_tokens')} cr={u.get('cache_read_input_tokens')} cw={u.get('cache_creation_input_tokens')} "
                  f"out={u.get('output_tokens')} sidechain={r.get('isSidechain')}{tag} | {'; '.join(did)}")
        elif t == "user":
            c = r.get("message", {}).get("content")
            if isinstance(c, str):
                print("U " + c[:110].replace("\n", " "))
        else:
            s = json.dumps(r)
            if "additionalContext" in s or "hook_additional_context" in s or "advisor nonce" in s or "hook" in str(r.get("subtype", "")) or "hook" in str(r.get("attachment", {}).get("type", "")):
                a = r.get("attachment", {})
                print(f"H type={t} sub={r.get('subtype') or a.get('type')} event={a.get('hookEvent') or a.get('hookName')} | {str(a.get('content') or r.get('content') or '')[:140]}")

for a in sys.argv[1:]:
    p = find(a)
    show(p) if p else print(f"== {a}: no transcript found")
