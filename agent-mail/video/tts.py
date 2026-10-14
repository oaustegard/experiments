import json, sys, subprocess
sys.path.insert(0, '/mnt/skills/user/invoking-gemini/scripts')
from gemini_client import generate_speech
n = json.load(open('narration.json'))
items = [("intro", n["intro"])]
for ri, r in enumerate(n["runs"]):
    items.append((f"r{ri}-title", r["title"]))
    items += [(f"r{ri}-b{bi}", b["text"]) for bi, b in enumerate(r["beats"])]
    items.append((f"r{ri}-end", r["end"]))
only = set(sys.argv[1:])
out = {}
for key, text in items:
    if only and key not in only: continue
    r = generate_speech(text, f"vo/{key}.wav", voice="voice_66w0iod1i7ol", style=n["style"])
    out[key] = r["seconds"] if r else None
    print(key, out[key])
json.dump(out, open('vo/durations.json' if not only else 'vo/durations-partial.json', 'w'), indent=1)
