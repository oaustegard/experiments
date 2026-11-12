#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Run doom_host inside the jail, answer its JUDGE requests, post-process.

    python3 play.py --name e1m1-numbers --max-seconds 300
    python3 play.py --name e1m1-judged --judge --max-seconds 300
    python3 play.py --name e1m1-assisted --assist-exit --max-seconds 300

The game binary is PrBoom, code nobody here wrote, so it runs under the hub's
scripts/jail.sh: no network, uid nobody, empty environment, in a copy under
/tmp/jail/. It talks to this script over stdin/stdout only. This script, in
turn, is the only part that reaches the network: the judge calls go through
`claude -p` (Haiku, low effort), and the finished video goes to R2.

The judge is a rotating window, not one long conversation. Each call is a
fresh `claude -p` process that sees the current view, the leading candidates'
views one second later, the planner's numbers, and a short note the previous
judge left. It answers with a pick and a new note, and exits.

Outputs: runs/<name>/ (gitignored: video, frames, logs) and
results/<name>.json (committed: summary, judge stats, video URL).
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
JAIL = Path("/home/user/claude-workspace/scripts/jail.sh")
HUB_SCRIPTS = Path("/home/user/claude-workspace/scripts")
R2_PREFIX = "video/doom-planner"

JUDGE_SYSTEM = """You steer a Doom player on the shareware episode. A planner \
simulated each candidate action for one second in copies of the game and \
shows you where each one leads. Pick the one that best makes progress toward \
the level's exit: new rooms and corridors over places already seen, doors \
and switches over walls, killing monsters that are in the way, not taking \
damage. Reply with exactly two lines and nothing else:
PICK <letter>
NOTE <at most 25 words for the next judge: where you are, what you were \
heading for>"""


def build():
    subprocess.run(["make", "-j4"], cwd=HERE, check=True, stdout=subprocess.DEVNULL)


def jail_dir(name):
    d = Path("/tmp/jail") / f"doom-planner-{name}"
    if d.exists():
        shutil.rmtree(d)
    d.mkdir(parents=True)
    shutil.copy2(HERE / "doom_host", d / "doom_host")
    shutil.copy2(HERE / "wads" / "DOOM1.WAD", d / "DOOM1.WAD")
    subprocess.run(["chown", "-R", "nobody:nogroup", str(d)], check=True)
    return d


def ppm_sheet(options, now_path, out_png):
    """One image for the judge: the current view, then each option labelled."""
    from PIL import Image, ImageDraw

    tiles = [("NOW", now_path)] + [(o["label"], o["frame"]) for o in options]
    ims = [Image.open(p).convert("RGB").resize((320, 240)) for _, p in tiles]
    cols = 3 if len(ims) > 4 else 2
    rows = (len(ims) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 330, rows * 270), (20, 20, 20))
    draw = ImageDraw.Draw(sheet)
    for i, ((label, _), im) in enumerate(zip(tiles, ims)):
        x, y = (i % cols) * 330 + 5, (i // cols) * 270 + 25
        sheet.paste(im, (x, y))
        draw.text((x, y - 18), label if label == "NOW" else f"{label}: {options[i - 1]['action']}",
                  fill=(255, 255, 0))
    sheet.save(out_png)


def ask_haiku(sheet_png, options, note, log):
    lines = [f"{o['label']}: {o['action']} (planner score {o['score']:.0f}; "
             f"dealt {o['dealt']}, hurt {o['hurt']}, new lines {o['explored']}, "
             f"new areas {o['cells']})" for o in options]
    prompt = (f"Read the image {sheet_png}. Top-left is the view now; the others are "
              f"one second after each option.\n\nOptions:\n" + "\n".join(lines) +
              f"\n\nNote from the previous judge: {note or '(none yet)'}")
    env = dict(os.environ, FORCE_PROMPT_CACHING_5M="1")
    t0 = time.time()
    try:
        out = subprocess.run(
            ["claude", "-p", "--model", "haiku", "--effort", "low",
             "--tools", "Read", "--allowedTools", "Read", "--strict-mcp-config",
             "--setting-sources", "", "--disable-slash-commands",
             "--system-prompt", JUDGE_SYSTEM, prompt],
            capture_output=True, text=True, timeout=120, env=env)
        text = out.stdout.strip()
    except subprocess.TimeoutExpired:
        text = ""
    dt = time.time() - t0
    pick, new_note = None, note
    for line in text.splitlines():
        line = line.strip()
        if line.upper().startswith("PICK"):
            letter = line[4:].strip()[:1].upper()
            idx = ord(letter) - ord("A") if letter else -1
            if 0 <= idx < len(options):
                pick = idx
        elif line.upper().startswith("NOTE"):
            new_note = line[4:].strip(" :")[:200]
    log.write(json.dumps({"seconds": round(dt, 2), "pick": pick, "note": new_note,
                          "options": [o["action"] for o in options],
                          "raw": text[:400]}) + "\n")
    log.flush()
    return pick, new_note, dt


def subtitles(decisions_path, ass_path, fps=35):
    """An ASS caption per decision: what was chosen, by whom, against what."""
    def ts(tic):
        s = tic / fps
        return f"{int(s // 3600)}:{int(s // 60) % 60:02d}:{s % 60:05.2f}"

    rows = [json.loads(l) for l in open(decisions_path)]
    out = ["[Script Info]", "ScriptType: v4.00+", "PlayResX: 960", "PlayResY: 720", "",
           "[V4+ Styles]",
           "Format: Name, Fontname, Fontsize, PrimaryColour, OutlineColour, BorderStyle, "
           "Outline, Shadow, Alignment, MarginV",
           "Style: Default,DejaVu Sans Mono,22,&H00FFFFFF,&H00000000,1,2,0,7,10", "",
           "[Events]", "Format: Layer, Start, End, Style, Text"]
    for a, b in zip(rows, rows[1:] + [None]):
        end = b["tic"] if b else a["tic"] + 8
        top = a["top"]
        alts = "  ".join(f"{n} {s:.0f}" for n, s, _ in top[1:3])
        who = "JUDGE" if a["by"] == "judge" else "plan"
        out.append(f"Dialogue: 0,{ts(a['tic'])},{ts(end)},Default,"
                   f"{who}: {a['chosen']} ({top[0][1]:.0f})\\Nnext: {alts}")
    Path(ass_path).write_text("\n".join(out) + "\n")


def upload(path, key):
    sys.path.insert(0, str(HUB_SCRIPTS))
    import embstore  # noqa: E402  (R2 helper; reads CF_* from the environment)

    embstore._put(key, Path(path).read_bytes(), "video/mp4")
    return f"{embstore.PUBLIC}/{key}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--max-seconds", type=int, default=300)
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--assist-exit", action="store_true")
    ap.add_argument("--scenarios", type=int, default=3)
    ap.add_argument("--horizon", type=int, default=52)
    ap.add_argument("--commit", type=int, default=8)
    ap.add_argument("--risk", type=float, default=0.5)
    ap.add_argument("--skill", type=int, default=3)
    ap.add_argument("--warp", nargs=2, type=int, default=[1, 1])
    ap.add_argument("--judge-every", type=int, default=4)
    ap.add_argument("--no-upload", action="store_true")
    args = ap.parse_args()

    build()
    jd = jail_dir(args.name)
    cmd = ["sh", str(JAIL), "-C", str(jd), "--", "./doom_host", "--iwad", "DOOM1.WAD",
           "--out", "run", "--max-tics", str(args.max_seconds * 35),
           "--scenarios", str(args.scenarios), "--horizon", str(args.horizon),
           "--commit", str(args.commit), "--risk", str(args.risk),
           "--skill", str(args.skill), "--warp", str(args.warp[0]), str(args.warp[1]),
           "--judge-every", str(args.judge_every)]
    if args.judge:
        cmd.append("--judge")
    if args.assist_exit:
        cmd.append("--assist-exit")

    out_dir = HERE / "runs" / args.name
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    jlog = open(out_dir / "judge.jsonl", "w")
    events = open(out_dir / "events.log", "w")

    t0 = time.time()
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1,
                            encoding="utf-8", errors="replace")  # the engine prints raw bytes
    note, judge_s, judged = "", 0.0, 0
    for line in proc.stdout:
        line = line.rstrip("\n")
        if line.startswith("JUDGE "):
            req = json.loads(line[6:])
            fix = lambda p: str(jd / p)  # paths are relative to the jail's cwd
            for o in req["options"]:
                o["frame"] = fix(o["frame"])
            sheet = out_dir / "judge" / f"{req['decision']:06d}.png"
            sheet.parent.mkdir(exist_ok=True)
            ppm_sheet(req["options"], fix(req["now"]), sheet)
            pick, note, dt = ask_haiku(sheet, req["options"], note, jlog)
            judge_s += dt
            judged += 1
            proc.stdin.write(f"PICK {pick if pick is not None else -1}\n")
            proc.stdin.flush()
            continue
        events.write(line + "\n")
        events.flush()
        if line.startswith(("START", "STATUS", "LEVEL", "DIED", "END", "GIVEUP", "RESET")):
            print(line, flush=True)
    proc.wait()
    wall = time.time() - t0
    print(f"doom_host exit status {proc.returncode}", flush=True)
    if not (jd / "run" / "summary.json").exists():
        sys.exit(f"no summary.json: doom_host ended early (status {proc.returncode}); "
                 f"see {out_dir / 'events.log'}")

    run = jd / "run"
    for f in ("summary.json", "decisions.jsonl", "video.mp4"):
        if (run / f).exists():
            shutil.copy2(run / f, out_dir / f)
    summary = json.loads((out_dir / "summary.json").read_text())

    video = out_dir / "video.mp4"
    annotated = out_dir / "video_annotated.mp4"
    subtitles(out_dir / "decisions.jsonl", out_dir / "captions.ass")
    captions = out_dir / "captions.ass"
    # 640x480 for R2; 480x360 under the 15 MB limit for an Artifact asset
    for out, size, crf in ((annotated, "640:480:flags=neighbor", "28"),
                           (out_dir / "video_small.mp4", "480:360:flags=bilinear", "31")):
        subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(video),
                        "-vf", f"scale={size},subtitles={captions}", "-c:v", "libx264",
                        "-preset", "medium", "-crf", crf, "-pix_fmt", "yuv420p", str(out)],
                       check=True)

    url = None
    if not args.no_upload:
        url = upload(annotated, f"{R2_PREFIX}/{args.name}.mp4")

    result = {"name": args.name, "args": vars(args), "summary": summary,
              "wall_seconds": round(wall, 1),
              "judge_calls": judged, "judge_seconds": round(judge_s, 1),
              "video_url": url}
    (HERE / "results").mkdir(exist_ok=True)
    (HERE / "results" / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
