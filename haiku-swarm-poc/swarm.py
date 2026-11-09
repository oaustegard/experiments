"""Arm B (Haiku swarm) and Arm C (Haiku proposers, Sonnet synthesizer).

One round = K swarms in parallel. A swarm is N proposer calls, each reading
one parent program and one focus x thinking-style prompt and returning a short
idea with no code, then one synthesizer call that reads the swarm's ideas and
parents and writes one complete program. Each swarm's program is scored in
the jail as soon as it exists and joins the archive.

Swarm types, in slot order: refine (one of the top 2), hybridize (3 parents
drawn from the top 10), explore (2 parents drawn from the whole archive, plus
proposers that read only the problem), fresh start (the initial program; at
most one per round). K=4 gives one of each; larger K cycles refine, hybridize,
explore.

Usage: python3 swarm.py RUN_ID --seed S --K 4 --N 8 --synth haiku --cap 3 [--wall 3600]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import problems
import prompts as P
from common import JAIL, jail_eval, run_dir, score_of, setup_jail
from gateway import BudgetExceeded, Gateway, QuotaGuard

SYNTH_EFFORT = {"haiku": "medium", "sonnet": "high"}   # each model's API default
REFINE_CAP = 20                                         # SwarmEvolve's per-program refinement cap


def sha(code: str) -> str:
    return hashlib.sha256(code.encode()).hexdigest()[:16]


def extract_code(text: str, entry: str = "run_packing") -> str | None:
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.S)
    for b in reversed(blocks):
        if f"def {entry}" in b:
            return b.strip() + "\n"
    if f"def {entry}" in text and "```" not in text:
        return text.strip() + "\n"
    return None


def cut_idea(text: str) -> str:
    words = text.strip().split()
    return " ".join(words[:P.IDEA_WORDS])


class Run:
    def __init__(self, a):
        self.a = a
        self.d = run_dir(a.run_id)
        self.rng = random.Random(a.seed)
        self.spec = problems.get(a.problem)
        # Circle packing keeps the prompts the first runs used (problems.get adds nothing new there).
        self.pspec = None if a.problem == "circle" else self.spec
        self.gw = Gateway(self.d, a.cap)
        calib = JAIL / "runs" / "_calib" / "overhead.json"
        if calib.exists() and not self.gw.overhead:
            self.gw.overhead = json.loads(calib.read_text())
        self.archive_path = self.d / "archive.json"
        self.events_path = self.d / "events.jsonl"
        self.clock_path = self.d / "clock.json"
        self.archive: list[dict] = []
        self.round = 0
        used = 0.0
        if self.archive_path.exists():
            st = json.loads(self.archive_path.read_text())
            self.archive, self.round = st["archive"], st["round"]
            self.rng.setstate(_tuplify(st["rng"]))
        if self.clock_path.exists():
            used = json.loads(self.clock_path.read_text())["used"]
        self.t0 = time.time() - used

    # ----------------------------------------------------------- bookkeeping
    def elapsed(self) -> float:
        return time.time() - self.t0

    def save(self):
        tmp = self.archive_path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"archive": self.archive, "round": self.round,
                                   "rng": self.rng.getstate()}))
        tmp.rename(self.archive_path)
        self.clock_path.write_text(json.dumps({"used": self.elapsed()}))

    def event(self, **kw):
        for k, v in (("te", self.elapsed()), ("spent", self.gw.spent), ("spent_api", self.gw.spent_api)):
            kw.setdefault(k, v)
        with self.events_path.open("a") as f:
            f.write(json.dumps(kw) + "\n")

    def ranked(self) -> list[dict]:
        return sorted(self.archive, key=lambda p: -p["score"])

    def add(self, code: str, metrics: dict, kind: str, parents: list[str], when: dict | None = None) -> dict:
        p = {"id": len(self.archive), "sha": sha(code), "code": code, "metrics": metrics,
             "score": score_of(metrics, self.a.problem), "round": self.round, "kind": kind,
             "parents": parents, "refined": 0}
        self.archive.append(p)
        self.event(kind="eval", sha=p["sha"], score=p["score"], round=self.round, swarm=kind,
                   error=str(metrics.get("error", ""))[:200] or None, **(when or {}))
        return p

    # ------------------------------------------------------------- the loop
    def start(self):
        if self.archive:
            return
        code = (JAIL / self.spec["jail_dir"] / "initial_program.py").read_text()
        d = self.d / "r0000"
        d.mkdir(exist_ok=True)
        (d / "s0.py").write_text(code)
        res = jail_eval(d, timeout=self.spec["eval_timeout"], problem_dir=self.spec["jail_dir"])
        self.add(code, res["s0.py"]["metrics"], "initial", [])
        self.save()

    def plan(self) -> list[dict]:
        order = ["refine", "hybrid", "explore", "fresh"]
        types = [order[i] if i < 4 else order[(i - 4) % 3] for i in range(self.a.K)]
        ranked = self.ranked()
        init = self.archive[0]
        swarms, refine_slot = [], 0
        eligible = [p for p in ranked if p["refined"] < REFINE_CAP]
        for t in types:
            if t == "refine":
                pool = eligible[:2] or ranked[:1]
                parent = pool[(self.round + refine_slot) % len(pool)]
                refine_slot += 1
                swarms.append({"type": t, "parents": [parent]})
            elif t == "hybrid":
                top = ranked[:10]
                swarms.append({"type": t, "parents": self.rng.sample(top, min(3, len(top)))})
            elif t == "explore":
                swarms.append({"type": t, "parents": self.rng.sample(self.archive, min(2, len(self.archive))),
                               "problem_only": max(1, self.a.N // 4) if self.a.N else 0})
            else:
                swarms.append({"type": t, "parents": [init]})
        return swarms

    def assignments(self, sw: dict) -> list[dict]:
        N = self.a.N
        if N == 0:
            return []
        aspects = sw["type"] in ("refine", "hybrid")
        foci = P.ASPECTS if aspects else P.PERSPECTIVES
        phrase = "Pay particular attention to" if aspects else "Consider"
        picks = self.rng.sample(foci, N) if N <= len(foci) else [self.rng.choice(foci) for _ in range(N)]
        n_none = sw.get("problem_only", 0)
        parents = sw["parents"]
        out = []
        for i in range(N):
            parent = None if i >= N - n_none else parents[i % len(parents)]
            out.append({"parent": parent, "phrase": phrase, "focus": picks[i],
                        "style": self.rng.choice(P.STYLES)})
        return out

    def run_swarm(self, idx: int, sw: dict, asg: list[dict]) -> dict:
        info = {"idx": idx, "type": sw["type"], "parents": [p["id"] for p in sw["parents"]]}
        ideas = []
        # One proposer per distinct parent goes first so the rest read its cached prefix.
        groups: dict = {}
        for a in asg:
            groups.setdefault(id(a["parent"]), []).append(a)

        def propose(a):
            r = self.gw.call("haiku", P.proposer_system(a["parent"], self.pspec),
                             P.proposer_user(a["phrase"], a["focus"], a["style"]),
                             effort=self.a.prop_effort, max_tokens=4000, role="proposer",
                             meta={"round": self.round, "swarm": idx})
            return cut_idea(r["text"]) if r["ok"] and r["text"].strip() else None

        def run_group(g):
            first = propose(g[0])
            with ThreadPoolExecutor(max(1, len(g) - 1)) as ex:
                rest = list(ex.map(propose, g[1:]))
            return [first] + rest

        with ThreadPoolExecutor(max(1, len(groups))) as ex:
            for res in ex.map(run_group, groups.values()):
                ideas += [t for t in res if t]
        info["ideas"] = ideas
        synth_model = self.a.synth
        r = self.gw.call(synth_model, P.synth_system(sw["parents"], self.pspec),
                         P.synth_user(ideas, len(sw["parents"]), self.pspec),
                         effort=self.a.synth_effort or SYNTH_EFFORT[synth_model], max_tokens=32000, role="synth",
                         meta={"round": self.round, "swarm": idx})
        code = extract_code(r["text"], self.spec["entry"]) if r["ok"] else None
        if code is None:
            info["fail"] = "no code" if r["ok"] else (r.get("error") or "call failed")
            self.event(kind="nocode", round=self.round, swarm=sw["type"], error=info["fail"][:200])
            return info
        h = sha(code)
        if any(p["sha"] == h for p in self.archive):
            info["dup"] = h
            self.event(kind="dup", sha=h, round=self.round, swarm=sw["type"])
            return info
        d = self.d / f"r{self.round:04d}" / f"s{idx}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "prog.py").write_text(code)
        res = jail_eval(d, workers=1, timeout=self.spec["eval_timeout"], problem_dir=self.spec["jail_dir"])
        info["code"], info["metrics"] = code, res["prog.py"]["metrics"]
        # The program counts as evaluated now, at the spend reached now.
        info["when"] = {"te": self.elapsed(), "spent": self.gw.spent, "spent_api": self.gw.spent_api}
        return info

    def step(self) -> bool:
        self.round += 1
        swarms = self.plan()
        for sw in swarms:
            if sw["type"] == "refine":
                sw["parents"][0]["refined"] += 1
        asgs = [self.assignments(sw) for sw in swarms]
        stop = False
        with ThreadPoolExecutor(len(swarms)) as ex:
            futs = [ex.submit(self.run_swarm, i, sw, asg) for i, (sw, asg) in enumerate(zip(swarms, asgs))]
            infos = []
            for f in futs:
                try:
                    infos.append(f.result())
                except BudgetExceeded:
                    stop = True
                except QuotaGuard:
                    raise
        for info in infos:
            if "code" in info:
                if any(p["sha"] == sha(info["code"]) for p in self.archive):
                    continue
                p = self.add(info["code"], info["metrics"], info["type"], info["parents"], info.get("when"))
                info["new_id"] = p["id"]
        with (self.d / "rounds.jsonl").open("a") as f:
            f.write(json.dumps({"round": self.round, "te": self.elapsed(), "spent": self.gw.spent,
                                "swarms": [{k: v for k, v in i.items() if k not in ("code", "when")} for i in infos]}) + "\n")
        self.save()
        return not stop

    def go(self):
        self.start()
        while self.gw.spent < self.a.cap and self.elapsed() < self.a.wall:
            if not self.step():
                break
        best = self.ranked()[0]
        (self.d / "best_program.py").write_text(best["code"])
        summary = {"run": self.a.run_id, "arm": self.a.arm, "problem": self.a.problem,
                   "prop_effort": self.a.prop_effort,
                   "synth_effort": self.a.synth_effort or SYNTH_EFFORT[self.a.synth],
                   "seed": self.a.seed, "K": self.a.K,
                   "N": self.a.N, "synth": self.a.synth, "cap": self.a.cap, "spent": self.gw.spent,
                   "spent_api": self.gw.spent_api, "calls": self.gw.calls, "rounds": self.round,
                   "elapsed": self.elapsed(), "programs": len(self.archive),
                   "best_score": best["score"], "best_sum": best["metrics"].get("sum_radii"),
                   "stopped_by": "cap" if self.gw.spent >= self.a.cap else "wall"}
        (self.d / "DONE").write_text(json.dumps(summary, indent=1))
        print("done ->", json.dumps(summary), flush=True)
        return summary


def _tuplify(x):
    return tuple(_tuplify(i) for i in x) if isinstance(x, list) else x


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("run_id")
    ap.add_argument("--arm", default="B")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--K", type=int, default=4)
    ap.add_argument("--N", type=int, default=8)
    ap.add_argument("--synth", default="haiku", choices=["haiku", "sonnet"])
    ap.add_argument("--cap", type=float, default=3.0)
    ap.add_argument("--wall", type=float, default=3600)
    ap.add_argument("--problem", default="circle", choices=list(problems.PROBLEMS))
    ap.add_argument("--prop-effort", dest="prop_effort", default="low")
    ap.add_argument("--synth-effort", dest="synth_effort", default=None)
    a = ap.parse_args(argv)
    setup_jail()
    if (run_dir(a.run_id) / "DONE").exists():
        print("already done:", a.run_id)
        return json.loads((run_dir(a.run_id) / "DONE").read_text())
    return Run(a).go()


if __name__ == "__main__":
    main()
