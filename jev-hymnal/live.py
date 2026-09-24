"""Play the hymnal live: Jev picks every next bar in real time while you steer from the keyboard.

    pip install numpy scipy sounddevice          # scipy optional (faster synth)
    export TYPESAFE_API_KEY=...                   # or CF_ACCOUNT_ID + CF_API_TOKEN (+ CF_GATEWAY_ID)
    python3 live.py                               # plays until you press e (end) or q (quit)

Keys while it plays:
    +  more energy          -  less energy          d  drop: everything in
    b  break it down        h  play the hook        c  change the chords
    s  go sparse            e  end the piece        q  quit now
    /  type a direction in words, Enter to send ("something eerie", "like a march")

How it keeps time. The audio callback's frame counter is the clock. Bar k+1 is requested from
Jev ASK_AT seconds into bar k, with whatever you asked for so far, and must be answered
DEADLINE seconds before bar k+1 starts; the synth then renders it (~20 ms) into a mixer the
callback reads from. If Jev is late, bar k+1 repeats bar k's patterns over the next chord, so the
music never stops. Every bar is logged with its decision time and slack.

Dry run without speakers (what CI or a container can do):
    python3 live.py --wav out.wav --script "6:+,10:d,16:b,20:/something eerie and slow,24:h,28:e"
replaces the sound card with a simulated one that drains the mixer at the real sample rate, feeds
the scripted keys at those bars, and writes what the device played to a WAV.
"""
import argparse, json, os, queue, random, sys, threading, time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "jev-tag-encoder"))
import jev  # noqa: E402
from hymnal import LAYERS, PROGRESSIONS, PHRASE, energy  # noqa: E402
from synth import events, render_audio, write_wav, BAR, SR  # noqa: E402

BS = int(BAR * SR)
TAIL = int(1.5 * SR)
KEYS_HELP = {"+": "more energy than now", "-": "less energy than now", "d": "the drop: everything in at full energy",
             "b": "break it down: strip back to a pad and one quiet voice", "h": "play the main hook",
             "c": "change to a different chord progression", "s": "go sparse, almost silent",
             "e": "end the piece now: land on the home chord and let it ring"}
QUESTION = {"drums": "Which drum pattern should play in the next bar?",
            "bass": "Which bass pattern should play in the next bar?",
            "keys": "Which keys pattern should play in the next bar?",
            "lead": "Which melody pattern should play in the next bar?"}
QS = {l: {"type": "choice", "instructions": q, "criteria": {k: v[0] for k, v in LAYERS[l].items()}}
      for l, q in QUESTION.items()}
QS_PROG = {"type": "choice", "instructions": "Which chord progression should the next four bars follow?",
           "criteria": {k: v[0] for k, v in PROGRESSIONS.items()}}
END_BARS = 4  # after 'e', stop on the first home-chord bar, or after this many bars


class Mixer:
    """Absolute-sample overlap-add ring buffer. Writers add bars ahead of the cursor; the audio
    callback drains it. Samples a writer delivers after their play time are dropped and counted."""

    def __init__(self, seconds=60):
        self.buf = np.zeros((int(seconds * SR), 2), np.float32)
        self.pos = 0                      # absolute sample index of the next frame to play
        self.lock = threading.Lock()

    def add(self, start, x, now):
        with self.lock:
            late = min(len(x), max(0, now - start, self.pos - start))
            idx = np.arange(start + late, start + len(x)) % len(self.buf)
            self.buf[idx] += x[late:]
            return late

    def read(self, frames):
        with self.lock:
            idx = np.arange(self.pos, self.pos + frames) % len(self.buf)
            out = self.buf[idx].copy()
            self.buf[idx] = 0
            self.pos += frames
        return 0.9 * np.tanh(1.4 * out)


class SimStream:
    """Stands in for the sound card on a dry run: drains the mixer in 1024-frame blocks at the
    real sample rate, so the dry run runs the same clock and deadlines as live playback."""

    def __init__(self, mixer, block=1024):
        self.mixer, self.block, self.blocks, self.run = mixer, block, [], threading.Event()

    def _loop(self):
        t0, n = time.perf_counter(), 0
        while self.run.is_set():
            self.blocks.append(self.mixer.read(self.block)); n += 1
            time.sleep(max(0.0, t0 + n * self.block / SR - time.perf_counter()))

    def start(self):
        self.run.set(); threading.Thread(target=self._loop, daemon=True).start()

    def stop(self):
        self.run.clear(); time.sleep(0.05)

    def close(self):
        pass


def with_timeout(fn, timeout, *args):
    """Run fn on a daemon thread; return its result, or raise TimeoutError. A late call is
    abandoned, never joined, so it cannot delay the music or the exit."""
    box = {}

    def run():
        try:
            box["ok"] = fn(*args)
        except Exception as e:  # noqa: BLE001 - surfaced to the caller below
            box["err"] = e
    t = threading.Thread(target=run, daemon=True)
    t.start(); t.join(timeout)
    if "ok" in box:
        return box["ok"]
    raise box.get("err") or TimeoutError(f"no answer within {timeout:.2f} s")


class Director:
    """What the person steering has asked for, as text for Jev's state."""

    def __init__(self):
        self.q = queue.Queue()
        self.current, self.since_bar, self.ending_from = None, None, None

    def take(self, bar):
        heard = []
        while not self.q.empty():
            heard.append(self.q.get())
        if heard:
            self.current, self.since_bar = "; then ".join(heard), bar
            if any(h == KEYS_HELP["e"] for h in heard) and self.ending_from is None:
                self.ending_from = bar
        return heard

    def text(self, bar):
        if not self.current:
            return "Nobody has asked for anything yet: start quietly and build a groove."
        ago = bar - self.since_bar
        when = "just now" if ago <= 1 else f"{ago} bars ago"
        return f"The person steering the music asked {when} for: {self.current}."


def line(bar, c):
    return f"bar {bar}: " + ", ".join(f"{l} {c[l]}" for l in ("harmony", "drums", "bass", "keys", "lead"))


def state(bar, phrase_pos, hist, director):
    return {"piece": "An open-ended instrumental in A minor at 120 bpm, played live one bar at a time "
                     "and steered by a person in real time.",
            "position": f"Next is bar {bar}, bar {phrase_pos + 1} of the current four-bar phrase.",
            "request": director.text(bar),
            "recent_bars": "\n".join(line(b, c) for b, c in hist[-4:]) or "Nothing has played yet."}


def pick(probs, rng):
    items = [(k, p) for k, p in probs.items() if p >= 0.05] or list(probs.items())
    r, acc = rng.random() * sum(p for _, p in items), 0.0
    for k, p in items:
        acc += p
        if r <= acc:
            return k
    return items[-1][0]


def keyboard(director, stop):
    """cbreak reader on a TTY (main() sets and restores the mode); '/' collects words until Enter."""
    if True:
        typing = None
        while not stop.is_set():
            ch = sys.stdin.read(1)
            if typing is not None:
                if ch in ("\n", "\r"):
                    if typing.strip():
                        director.q.put(typing.strip())
                        print(f"\n  > asked: {typing.strip()}", flush=True)
                    typing = None
                elif ch in ("\x7f", "\b"):
                    typing = typing[:-1]
                else:
                    typing += ch
                continue
            if ch == "/":
                typing = ""; print("\n  type a direction, Enter to send: ", end="", flush=True)
            elif ch == "q":
                stop.set()
            elif ch in KEYS_HELP:
                director.q.put(KEYS_HELP[ch]); print(f"\n  > {ch}: {KEYS_HELP[ch]}", flush=True)


def scripted(director, script, clock_bar, stop):
    """Dry run: feed '<bar>:<key or /words>' items when the clock reaches that bar."""
    items = []
    for part in script.split(","):
        b, _, what = part.partition(":")
        items.append((int(b), what.strip()))
    for b, what in sorted(items):
        while clock_bar() < b and not stop.is_set():
            time.sleep(0.01)
        msg = what[1:] if what.startswith("/") else KEYS_HELP.get(what, what)
        director.q.put(msg); print(f"  > [script bar {b}] {msg}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wav", help="dry run: no audio device, write the stream here")
    ap.add_argument("--script", help="dry run input: '6:+,10:d,20:/something eerie'")
    ap.add_argument("--max-bars", type=int, default=64)
    ap.add_argument("--ask-at", type=float, default=0.7, help="seconds into a bar to ask for the next one")
    ap.add_argument("--deadline", type=float, default=0.35, help="answer needed this long before the bar starts")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default=str(HERE / "runs" / f"live-{time.strftime('%Y%m%d-%H%M%S')}.jsonl"))
    a = ap.parse_args()
    if not (jev.typesafe_key() or os.environ.get("CF_API_TOKEN")):
        sys.exit("no Jev transport: export TYPESAFE_API_KEY (or CF_ACCOUNT_ID + CF_API_TOKEN)")
    jev.RPM = 600.0 if jev.typesafe_key() else 48.0

    rng = random.Random(a.seed)
    dry = a.wav is not None
    mixer, director, stop = Mixer(), Director(), threading.Event()
    render_audio(events([{"bar": 1, "choice": {"harmony": "Am", "drums": "four", "bass": "walk", "keys": "pad",
                                               "lead": "hook"}}]), BAR, normalize=False)  # warm the synth
    Path(a.log).parent.mkdir(exist_ok=True)
    logf = open(a.log, "w")

    def clock_samples():  # the device's frame counter is the clock, real or simulated
        return mixer.pos

    def clock_bar():
        return clock_samples() // BS + 1

    def decide(bar, phrase_pos, hist, prog, force_prog):
        qs = dict(QS)
        if phrase_pos == 0 or force_prog:
            qs["progression"] = QS_PROG
        st = state(bar, phrase_pos, hist, director)
        res, dt, _ = jev.post(st, qs, timeout=10)
        ans = res["answers"]
        return ({l: pick(ans[l]["probabilities"], rng) for l in qs},
                {l: ans[l]["probabilities"] for l in qs}, dt, st["request"])

    hist, prog, phrase_pos, bar = [], None, 0, 1
    # bar 1 is decided before the clock starts
    director.take(1)
    choice, probs, dt, req = decide(1, 0, hist, None, True)
    prog = choice.pop("progression")

    stream, tty_old = None, None
    if dry:
        stream = SimStream(mixer)
    else:
        import sounddevice as sd
        stream = sd.OutputStream(samplerate=SR, channels=2, dtype="float32", latency="high",
                                 callback=lambda out, n, t, s: out.__setitem__(slice(None), mixer.read(n)))
    if a.script:
        threading.Thread(target=scripted, args=(director, a.script, clock_bar, stop), daemon=True).start()
    elif not dry and sys.stdin.isatty():
        import termios, tty
        tty_old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=keyboard, args=(director, stop), daemon=True).start()
    print("keys: + - d b h c s e q, / to type a direction", flush=True)

    def play(bar, choice, prog, phrase_pos):
        c = dict(choice); c["harmony"] = PROGRESSIONS[prog][1][phrase_pos]
        x = render_audio(events([{"bar": 1, "choice": c}]), (BS + TAIL) / SR, normalize=False)
        late = mixer.add((bar - 1) * BS, x[: BS + TAIL], clock_samples())
        return c, late

    c, _ = play(1, choice, prog, 0)
    hist.append((1, c))
    rec = {"bar": 1, "request": req, "choice": c, "progression": prog, "probs": probs, "latency": dt,
           "slack": None, "fallback": False}
    logf.write(json.dumps(rec) + "\n")
    print(f"bar  1 │ {c['harmony']:>2} {prog:<8}│ {c['drums']:<8} {c['bass']:<10} {c['keys']:<8} {c['lead']:<7}│ Jev {dt:.2f}s", flush=True)
    stream.start()

    try:
        while not stop.is_set() and bar < a.max_bars:
            ask_sample = (bar - 1) * BS + int(a.ask_at * SR)
            while clock_samples() < ask_sample and not stop.is_set():
                time.sleep(0.005)
            nxt = bar + 1
            heard = director.take(nxt)
            force = any(h == KEYS_HELP["c"] for h in heard)
            ppos = 0 if force else (phrase_pos + 1) % PHRASE
            due = nxt - 1
            deadline_s = (due * BS - int(a.deadline * SR) - clock_samples()) / SR
            t0 = time.time()
            fallback = False
            try:
                choice, probs, dt, req = with_timeout(decide, max(0.0, deadline_s), nxt, ppos, list(hist), prog, force)
                if "progression" in choice:
                    prog = choice.pop("progression")
            except (TimeoutError, RuntimeError, jev.Blocked, KeyError):
                # late or failed: the same patterns again over the progression's next chord
                fallback, dt, probs, req = True, time.time() - t0, None, director.text(nxt)
                choice = {l: hist[-1][1][l] for l in QS}
            slack = deadline_s - dt
            phrase_pos = ppos
            c, late = play(nxt, choice, prog, phrase_pos)
            hist.append((nxt, c))
            rec = {"bar": nxt, "request": req, "heard": heard, "choice": c, "progression": prog, "probs": probs,
                   "latency": round(dt, 3), "slack": round(slack, 3), "fallback": fallback,
                   "late_samples": late, "energy": energy(c)}
            logf.write(json.dumps(rec) + "\n"); logf.flush()
            flag = " LATE→repeat" if fallback else (f" UNDERRUN {late}" if late else "")
            print(f"bar {nxt:2d} │ {c['harmony']:>2} {prog:<8}│ {c['drums']:<8} {c['bass']:<10} {c['keys']:<8} "
                  f"{c['lead']:<7}│ Jev {dt:.2f}s slack {slack:+.2f}s{flag}", flush=True)
            bar = nxt
            if director.ending_from and (c["harmony"] == "Am" and bar >= director.ending_from
                                         or bar - director.ending_from >= END_BARS):
                break
        # let the last bar and its tail play out
        end = bar * BS + TAIL
        while clock_samples() < end and not stop.is_set():
            time.sleep(0.01)
    finally:
        stop.set()
        stream.stop(); stream.close()
        if tty_old is not None:
            import termios
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, tty_old)
        logf.close()
    if dry:
        out = np.concatenate(stream.blocks)
        write_wav(a.wav, out)
        print(f"wrote {a.wav} ({len(out) / SR:.1f} s)")
    print(f"log: {a.log}")


if __name__ == "__main__":
    main()
