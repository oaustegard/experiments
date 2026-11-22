"""Build a held-out code corpus: source files first added to public repos in the last 3 weeks."""
import json, subprocess, sys, base64, random
REPOS = ['oaustegard/experiments', 'oaustegard/claude-skills', 'oaustegard/muninn-utilities', 'oaustegard/m5']
EXT = ('.py', '.js', '.ts', '.wgsl', '.c', '.cpp', '.h', '.swift', '.rs', '.sh', '.mojo')
SINCE = '2026-11-01T00:00:00Z'
def gh(path):
    r = subprocess.run(['gh', 'api', path], capture_output=True, text=True)
    return json.loads(r.stdout) if r.returncode == 0 else None
seen, picks = set(), []
for repo in REPOS:
    commits = gh(f'repos/{repo}/commits?since={SINCE}&per_page=100') or []
    for c in commits[:100]:
        d = gh(f"repos/{repo}/commits/{c['sha']}") or {}
        for f in d.get('files', []):
            n = f['filename']
            if f['status'] != 'added' or not n.endswith(EXT) or (repo, n) in seen: continue
            if any(s in n.lower() for s in ('vendor', 'node_modules', '.min.', 'dist/', 'third_party', 'secret', '.env', 'prboom')): continue
            if not (60 <= f.get('additions', 0) <= 500): continue
            seen.add((repo, n)); picks.append((repo, n, c['sha']))
    print(repo, len([p for p in picks if p[0] == repo]), file=sys.stderr)
random.seed(0); random.shuffle(picks)
out, total = [], 0
for repo, n, sha in picks:
    d = gh(f'repos/{repo}/contents/{n}?ref={sha}')
    if not d or d.get('encoding') != 'base64': continue
    try: txt = base64.b64decode(d['content']).decode('utf-8')
    except Exception: continue
    out.append(f'# file: {n}\n{txt}\n\n'); total += len(txt)
    if total > 260_000: break
open('code/heldout_code.txt', 'w').write(''.join(out))
print(len(out), 'files', total, 'chars', file=sys.stderr)
