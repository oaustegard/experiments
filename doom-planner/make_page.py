#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Write web/index.html (published on the experiments Pages site) from
results/summary.json. Videos stream from their R2 URLs.

    python3 make_page.py
"""

import html
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

RUNS = [
    ("e1m1-explore", "Exploring without exit knowledge",
     "The planner explores, fights, and once nothing unseen is reachable "
     "it goes to press the switches it has not tried."),
    ("e1m1-assisted", "Exit-assisted",
     "The same planner, also told where the exit switch is and given points for walking toward it."),
    ("e1m1-judged", "Exploring with Haiku on close calls",
     "As the first run, but when the two best options score within 10% of each other, Haiku makes the call. "
     "It sees a picture of the screen now and of where each close option leads 1.5 s later, with their "
     "scores, plus a one-line note the previous Haiku call left about where the player was heading. "
     "It runs at low reasoning effort, about 5.6 s per call, which stretched this run to 24 minutes "
     "of real time. Haiku's answers can vary, so this run would not repeat exactly."),
]


def fmt_time(sec):
    if sec is None:
        return "not reached"
    sec = int(sec + 0.5)
    return f"{sec // 60}:{sec % 60:02d}"


def main():
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
            ("Deaths in 5 min, any level", str(s["deaths"])),
            ("Decisions", f'{s["decisions"]:,}'),
            ("Rollouts (copies played out)", f'{s["rollouts"]:,}'),
        ]
        if judge:
            rows.append(("Haiku calls", f'{judge["calls"]} ({judge["mean_seconds"]} s each)'))
            rows.append(("Haiku kept the planner's top pick", f'{judge["picked_planner_top_share"]:.0%}'))
        dl = "".join(f"<div><dt>{html.escape(k, quote=False)}</dt><dd>{html.escape(v, quote=False)}</dd></div>"
                     for k, v in rows)
        vid = s.get("video_url")
        media = (f'<video controls preload="metadata" playsinline src="{html.escape(vid)}"></video>'
                 if vid else "")
        cards.append(f"""
      <article class="run">
        <div class="frame">{media}</div>
        <div class="meta">
          <h3>{html.escape(label)}</h3>
          <p>{html.escape(blurb, quote=False)}</p>
          <dl>{dl}</dl>
        </div>
      </article>""")

    tpl = (HERE / "page_template.html").read_text()
    long = summary.get("e1-long")
    long_line = ""
    if long:
        rest = (long["game_seconds"] - long["first_exit_seconds"]) / 60
        long_line = (f"Left to play for {long['game_seconds'] / 60:.0f} minutes without exit knowledge, the same "
                     f"planner finished E1M1 at {fmt_time(long['first_exit_seconds'])} (the same game, since runs repeat exactly) and spent the remaining {rest:.0f} minutes on E1M2 without finding "
                     f"its exit, with {long['deaths']} deaths and {long['kills'] - long['e1m1_kills']} kills there.")
        if long.get("video_url"):
            long_line += f' <a href="{html.escape(long["video_url"])}">Watch the 15-minute run</a>.'
    out = tpl.replace("{{RUNS}}", "".join(cards)).replace("{{LONG}}", long_line)
    (HERE / "web").mkdir(exist_ok=True)
    (HERE / "web" / "index.html").write_text(out)
    print("wrote web/index.html")


if __name__ == "__main__":
    main()
