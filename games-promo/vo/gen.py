import json, sys, concurrent.futures as cf
sys.path.insert(0, '/mnt/skills/user/invoking-gemini/scripts')
from gemini_client import generate_speech
S = json.load(open('script.json')); VOICE = sys.argv[1] if len(sys.argv) > 1 else 'Puck'; N = int(sys.argv[2]) if len(sys.argv) > 2 else 4
STYLE = "energetic game-trailer narrator, warm and playful, brisk pace, lively varied intonation, no announcer cheese"
jobs = [(l, s) for l in S for s in range(N)]
def run(j):
    l, s = j
    r = generate_speech(l['text'], output_path=f"takes/{l['id']}_s{s}.wav", voice=VOICE, style=STYLE)
    return l['id'], s, r and round(r['seconds'], 2)
with cf.ThreadPoolExecutor(8) as ex:
    for res in ex.map(run, jobs): print(*res, flush=True)
