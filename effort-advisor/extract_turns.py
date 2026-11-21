#!/usr/bin/env python3
"""Extract per-turn records from Claude Code transcripts (~/.claude/projects).
A turn = one human-typed prompt plus everything the assistant did until the next one."""
import json, glob, os, re, sys, collections
ROOTS = [os.path.expanduser('~/.claude/projects') + '/**/*.jsonl', 'data/archive/*.jsonl']
OUT = sys.argv[1] if len(sys.argv) > 1 else 'data/turns.jsonl'

def text_of(content):
    if isinstance(content, str): return content
    out = []
    for b in content or []:
        if isinstance(b, dict) and b.get('type') == 'text': out.append(b.get('text', ''))
    return '\n'.join(out)

def is_tool_result(content):
    return isinstance(content, list) and any(isinstance(b, dict) and b.get('type') == 'tool_result' for b in content)

SYS = re.compile(r'<system-reminder>.*?</system-reminder>', re.S)
def clean(t):
    return SYS.sub('', t).strip()

def classify_user(d):
    """human | scheduled | delegated | peer | command | interrupt | tool_result | other"""
    c = d.get('message', {}).get('content')
    if is_tool_result(c): return 'tool_result', ''
    t = text_of(c)
    if '[Request interrupted by user' in t: return 'interrupt', t
    if '<command-name>' in t or '<local-command-stdout>' in t or '<local-command-caveat>' in t:
        m = re.search(r'<command-name>(.*?)</command-name>', t, re.S)
        args = re.search(r'<command-args>(.*?)</command-args>', t, re.S)
        return 'command', ((m.group(1).strip() if m else '') + ' ' + (args.group(1).strip() if args else '')).strip()
    if d.get('isMeta'): return 'other', t
    t = clean(t)
    if not t: return 'other', ''
    o = d.get('origin') or {}
    kind = o.get('kind')
    if t.startswith(('<task-notification', '<side-sessions-event', '<ci-monitor-event', '<child-session-event', '[Artifact comment', '<bash-input', '<bash-stdout')): return 'other', t
    if t.startswith(('<wake', '<relay', '<project_claude_message', 'Another Claude session')): return 'peer', t
    if o.get('subkind', '').startswith('schedule') or d.get('turnOrigin') == 'scheduled': return 'scheduled', t
    if kind == 'task-notification': return 'delegated', t
    if kind == 'peer': return 'peer', t
    return 'human', t

turns = []
for f in sorted(x for r in ROOTS for x in glob.glob(r, recursive=True)):
    if '/subagents/' in f: continue
    cur = None; cmds = []; sess_turns = []
    for line in open(f, errors='ignore'):
        try: d = json.loads(line)
        except Exception: continue
        ty = d.get('type')
        if d.get('isSidechain'): continue
        if ty == 'user':
            kind, t = classify_user(d)
            if kind in ('human', 'scheduled', 'delegated', 'peer'):
                cur = dict(kind=kind, file=f, uuid=d.get('uuid'), source=('archive' if 'data/archive' in f else 'local'), entry=d.get('entrypoint'), session=os.path.basename(f)[:-6], project=os.path.basename(os.path.dirname(f)), ts=d.get('timestamp'),
                           prompt=t, pre_commands=cmds, models=collections.Counter(), efforts=collections.Counter(),
                           api_calls=0, tools=collections.Counter(), tool_errors=0, out_tokens=0, think_ms=0,
                           interrupted=False, final_text='', last_ts=d.get('timestamp'), perm=d.get('permissionMode'),
                           agent_models=[], trace=[], idx=len(sess_turns))
                cmds = []; sess_turns.append(cur)
            elif kind == 'command': cmds.append(t)
            elif kind == 'interrupt' and cur: cur['interrupted'] = True
            elif kind == 'tool_result' and cur:
                for b in d['message']['content']:
                    if isinstance(b, dict) and b.get('type') == 'tool_result' and b.get('is_error'): cur['tool_errors'] += 1
                cur['last_ts'] = d.get('timestamp') or cur['last_ts']
        elif ty == 'assistant' and cur:
            m = d.get('message', {})
            if m.get('model') == '<synthetic>': continue
            cur['api_calls'] += 1
            cur['models'][m.get('model')] += 1
            e = d.get('perTurnEffort') or d.get('effort')
            if e: cur['efforts'][str(e)] += 1
            cur['out_tokens'] += (m.get('usage') or {}).get('output_tokens', 0) or 0
            cur['think_ms'] += d.get('thinkingDurationMs') or 0
            cur['last_ts'] = d.get('timestamp') or cur['last_ts']
            for b in m.get('content') or []:
                if b.get('type') == 'tool_use':
                    cur['tools'][b.get('name')] += 1
                    if len(cur['trace']) < 40:
                        inp = b.get('input') or {}
                        sig = inp.get('command') or inp.get('file_path') or inp.get('description') or inp.get('pattern') or inp.get('query') or inp.get('skill') or ''
                        cur['trace'].append((b.get('name') or '')[-40:] + ': ' + str(sig)[:90].replace('\n', ' '))
                    if b.get('name') in ('Agent', 'Task'):
                        cur['agent_models'].append((b.get('input') or {}).get('subagent_type') or 'default')
                elif b.get('type') == 'text' and b.get('text', '').strip():
                    cur['final_text'] = b['text']
    for i, t in enumerate(sess_turns):
        nxt = sess_turns[i + 1] if i + 1 < len(sess_turns) else None
        t['next_prompt'] = nxt['prompt'][:600] if nxt else None
        t['next_kind'] = nxt['kind'] if nxt else None
        t['prev_final'] = sess_turns[i - 1]['final_text'][-1200:] if i else None
        t['prev_prompt'] = sess_turns[i - 1]['prompt'][:400] if i else None
        t['next_commands'] = nxt['pre_commands'] if nxt else cmds
        t['n_session_turns'] = len(sess_turns)
        for k in ('models', 'efforts', 'tools'): t[k] = dict(t[k])
        t['n_tools'] = sum(t['tools'].values())
        turns.append(t)

from datetime import datetime
def dur(a, b):
    try: return (datetime.fromisoformat(b.replace('Z', '+00:00')) - datetime.fromisoformat(a.replace('Z', '+00:00'))).total_seconds()
    except Exception: return None
with open(OUT, 'w') as o:
    for i, t in enumerate(turns):
        t['id'] = i; t['dur_s'] = dur(t['ts'], t['last_ts'])
        o.write(json.dumps(t) + '\n')
print(len(turns), 'turns ->', OUT)
