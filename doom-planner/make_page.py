#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write page.html (the published Artifact) from results/summary.json.

    python3 make_page.py '{"e1m1-explore": "<asset url>", ...}'

The argument maps run names to the uploaded 480x360 video URLs.
"""

import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ("e1m1-explore", "Scores only",
     "No knowledge of the exit. The planner explores, fights, and once nothing unseen is reachable "
     "it goes to press the switches it has not tried."),
    ("e1m1-assisted", "Exit-assisted",
     "Same planner, plus a reward for walking distance gained toward the level's exit line. "
     "This is the version that knows where to go."),
    ("e1m1-judged", "Haiku on close calls",
     "Scores only, but when the two best options score within 10% of each other, Haiku (low effort) "
     "chooses from the views one and a half seconds ahead, with a short note left by the previous call."),
]


def fmt_time(sec):
    if sec is None:
        return "not reached"
    return f"{int(sec // 60)}:{int(sec % 60):02d}"


def main():
    videos = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {}
    summary = json.loads((HERE / "results" / "summary.json").read_text())
    cards = []
    for name, label, blurb in RUNS:
        s = summary.get(name)
        if not s:
            continue
        judge = s.get("judge")
        rows = [
            ("E1M1 exit", fmt_time(s["first_exit_seconds"])),
            ("E1M1 kills", f'{s["e1m1_kills"]} of 6'),
            ("Deaths in 5 min", str(s["deaths"])),
            ("Decisions", f'{s["decisions"]:,}'),
            ("Rollouts", f'{s["rollouts"]:,}'),
        ]
        if judge:
            rows.append(("Haiku calls", f'{judge["calls"]} ({judge["mean_seconds"]} s each)'))
            rows.append(("Haiku kept the planner's top pick", f'{judge["picked_planner_top_share"]:.0%}'))
        dl = "".join(f"<div><dt>{html.escape(k)}</dt><dd>{html.escape(v)}</dd></div>" for k, v in rows)
        vid = videos.get(name)
        media = (f'<video controls preload="metadata" playsinline src="{html.escape(vid)}"></video>'
                 if vid else '<div class="novideo">Video not uploaded</div>')
        full = s.get("video_url")
        link = (f'<a href="{html.escape(full)}">640×480 version</a>' if full else "")
        cards.append(f"""
      <article class="run">
        <div class="frame">{media}</div>
        <div class="meta">
          <h3>{html.escape(label)}</h3>
          <p>{html.escape(blurb)}</p>
          <dl>{dl}</dl>
          <p class="link">{link}</p>
        </div>
      </article>""")

    tpl = (HERE / "page_template.html").read_text()
    long = summary.get("e1-long")
    long_line = ""
    if long:
        rest = (long["game_seconds"] - long["first_exit_seconds"]) / 60
        long_line = (f"Left to play for {long['game_seconds'] / 60:.0f} minutes without exit knowledge, the same "
                     f"planner finished E1M1 at {fmt_time(long['first_exit_seconds'])} (the identical game: runs "
                     f"are deterministic) and spent the remaining {rest:.0f} minutes on E1M2 without finding "
                     f"its exit, with {long['deaths']} deaths and {long['kills'] - long['e1m1_kills']} kills there.")
        if long.get("video_url"):
            long_line += f' <a href="{html.escape(long["video_url"])}">Watch the 15-minute run</a>.'
    out = tpl.replace("{{RUNS}}", "".join(cards)).replace("{{LONG}}", long_line)
    (HERE / "page.html").write_text(out)
    print("wrote page.html")


if __name__ == "__main__":
    main()
