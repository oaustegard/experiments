"""Re-answer the Jev conditions with a tool-using reader: the same carried text, written to a file,
read by Haiku 4.5 with Read and Bash. Separates what was stored from how well it was read.
Writes results_tally/<cond>+tools__<n>__<seed>.json, reusing the original steps."""
from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

from gen_tally import episode, task
from relay import MODEL

RES = Path(__file__).resolve().parent / "results_tally"
SYS = ("You answer from the material in the current directory. You may write and run code. "
       "End your reply with only the JSON object requested.")


def main(conds=("jev", "jev_fifo"), sizes=(32, 96), seed=0) -> None:
    for cond in conds:
        for n in sizes:
            out = RES / f"{cond}+tools__{n}__{seed}.json"
            if out.exists():
                continue
            r = json.loads((RES / f"{cond}__{n}__{seed}.json").read_text())
            ep = episode(n, seed)
            with tempfile.TemporaryDirectory() as wd:
                Path(wd, "carried.txt").write_text(r["carried"])
                prompt = (f"TASK:\n{task(ep)}\n\nThe stream has ended. Everything carried forward is in "
                          f"./carried.txt ({len(r['carried'].splitlines())} lines, chronological). Report the "
                          f"final register as one JSON object mapping each of the {len(ep['assets'])} assets to "
                          f'{{"holder": <name>, "count": <number of handovers since the start>}}.')
                p = subprocess.run(["claude", "-p", "--model", MODEL, "--tools", "Read", "Bash",
                                    "--permission-mode", "acceptEdits", "--allowedTools", "Bash", "Read",
                                    "--setting-sources", "", "--strict-mcp-config", "--no-session-persistence",
                                    "--system-prompt", SYS, "--output-format", "json"],
                                   input=prompt, cwd=wd, capture_output=True, text=True, timeout=1800)
            d = json.loads(p.stdout)
            txt = d.get("result") or ""
            m = re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", txt, re.S)
            parsed = None
            for cand in sorted(m, key=len, reverse=True):
                try:
                    parsed = json.loads(cand); break
                except json.JSONDecodeError:
                    pass
            r2 = {**r, "cond": f"{cond}+tools", "answer": {"parsed": parsed, "raw": txt[-4000:],
                                                          "cost": d.get("total_cost_usd", 0), "turns": d.get("num_turns")}}
            out.write_text(json.dumps(r2))
            print(cond, n, "turns", d.get("num_turns"), "cost", d.get("total_cost_usd"), "parsed", bool(parsed), flush=True)


if __name__ == "__main__":
    main()
