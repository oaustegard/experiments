"""Prompt iteration for deciding-with-confidence: one single-call run per angle, stacks simulated offline.

    python3 run/prompts.py run --angles plain,literal,sceptic [--sets bundled,papers,sections] [--split dev]
    python3 run/prompts.py summary [--split dev] [--stacks plain+literal+sceptic,...]

Every angle runs as one low-effort sample per item (options in their listed order), checkpointed per item to
results/prompts/<set>__<angle>.jsonl, so a killed run resumes. A stack is scored by averaging the per-angle
distributions of the same items, which is what decide.py's pool does apart from the option shuffle.
Angles are the built-in names in decide.py or keys of run/angles.json (name -> instruction text).
Items split into dev and test by a hash of their id; choose prompts on dev, report on test.
"""
import argparse
import gzip
import hashlib
import itertools
import json
import random
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[1]
DECIDE = HERE.parent.parent / "claude-skills" / "deciding-with-confidence" / "scripts"
OUT = HERE / "results" / "prompts"
SETS = ("bundled", "papers", "sections")


def split_of(item_id) -> str:
    return "dev" if int(hashlib.sha1(str(item_id).encode()).hexdigest(), 16) % 2 == 0 else "test"


def load_set(name):
    if name == "bundled":
        rows = [json.loads(x) for x in (DECIDE.parent / "assets" / "eval.jsonl").read_text().splitlines() if x.strip()]
    else:
        with gzip.open(HERE / "data" / f"{name}.jsonl.gz", "rt") as f:
            rows = [json.loads(x) for x in f]
    return rows


def angle_texts():
    p = HERE / "run" / "angles.json"
    return json.loads(p.read_text()) if p.exists() else {}


def run(a):
    sys.path.insert(0, str(DECIDE))
    import decide
    decide.EFFORT = a.effort
    decide.TIMEOUT = 180
    transport = decide.pick_transport("cli")
    custom = angle_texts()
    OUT.mkdir(parents=True, exist_ok=True)
    for name in a.sets.split(","):
        items = [r for r in load_set(name) if a.split == "all" or split_of(r["id"]) == a.split][:a.limit or None]
        for angle in a.angles.split(","):
            text = custom.get(angle, angle)
            if angle not in decide.ANGLES and angle not in custom:
                sys.exit(f"unknown angle {angle}")
            # --tag runs decide.py's own k-sample path (option shuffle, its pooling) under the tag's name;
            # sample 0 of it is the single call
            k, texts = (a.k, list(decide.STACK)[:a.k] if a.k > 1 else [text]) if a.tag else (1, [text])
            path = OUT / f"{name}__{a.tag or angle}.jsonl"
            done = set()
            if path.exists():
                done = {json.loads(x)["id"] for x in path.read_text().splitlines() if x.strip()}
            todo = [r for r in items if r["id"] not in done]
            lock = threading.Lock()
            errors = [0]

            def one(item):
                rows = decide.evaluate([item], k, transport, angles=texts)
                if any(r.get("error") for r in rows):
                    errors[0] += 1
                    return  # left out of the file so a rerun retries it
                with lock, path.open("a") as f:
                    for r in rows:
                        f.write(json.dumps({k: r[k] for k in ("id", "name", "type", "label", "raw", "per_sample", "usage") if k in r}
                                           | ({"refusal": True} if r.get("refusal") else {})) + "\n")

            with ThreadPoolExecutor(max_workers=max(1, a.workers // k)) as ex:
                list(ex.map(one, todo))
            print(f"RUN_DONE {name} {angle} items={len(todo)} errors={errors[0]}", flush=True)
    print("done", flush=True)


def load(name, angle, split):
    path = OUT / f"{name}__{angle}.jsonl"
    if not path.exists():
        return None
    rows = {}
    for x in path.read_text().splitlines():
        r = json.loads(x)
        if (split == "all" or split_of(r["id"]) == split) and "raw" in r:
            rows[(r["id"], r["name"])] = r
    return rows


def metrics(pairs):
    """pairs: [(probs, label)] -> accuracy, ECE (10 bins, top prob), AUROC of top prob for right vs wrong, Brier."""
    if not pairs:
        return None
    conf, right, brier = [], [], 0.0
    for p, y in pairs:
        top = max(p, key=p.get)
        conf.append(p[top])
        right.append(top == y)
        brier += sum((v - (k == y)) ** 2 for k, v in p.items())
    n = len(pairs)
    ece = 0.0
    for b in range(10):
        idx = [i for i, c in enumerate(conf) if b / 10 < c <= (b + 1) / 10 or (b == 0 and c == 0)]
        if idx:
            ece += len(idx) / n * abs(sum(right[i] for i in idx) / len(idx) - sum(conf[i] for i in idx) / len(idx))
    pos = [c for c, r in zip(conf, right) if r]
    neg = [c for c, r in zip(conf, right) if not r]
    au = sum((x > y) + 0.5 * (x == y) for x in pos for y in neg) / (len(pos) * len(neg)) if pos and neg else None
    return {"n": n, "acc": sum(right) / n, "ece": ece, "auroc": au, "brier": brier / n, "right": right}


def pool(dists):
    keys = dists[0].keys()
    return {k: sum(d[k] for d in dists) / len(dists) for k in keys}


def boot_diff(r1, r0, reps=2000):
    """Paired bootstrap 90% interval for accuracy(r1) - accuracy(r0) over the same answers."""
    d = [a - b for a, b in zip(r1, r0)]
    rng = random.Random(0)
    n = len(d)
    means = sorted(sum(d[rng.randrange(n)] for _ in range(n)) / n for _ in range(reps))
    return means[int(reps * 0.05)], means[int(reps * 0.95)]


def summary(a):
    names = sorted({p.name.split("__")[1][:-6] for p in OUT.glob("*.jsonl")})
    arms = [(n, [n]) for n in names] + [(s, s.split("+")) for s in (a.stacks.split(",") if a.stacks else [])]
    if a.all_stacks:
        arms += [("+".join(c), list(c)) for c in itertools.combinations(names, 3)]
    table = {}
    for setname in SETS:
        data = {n: load(setname, n, a.split) for n in names}
        data = {n: d for n, d in data.items() if d}
        if not data:
            continue
        # score every arm on the answers all of its angles cover, and plain on the same answers
        print(f"== {setname} ({a.split})")
        for arm, parts in arms:
            if not all(p in data for p in parts) or "plain" not in data:
                continue
            keys = sorted(set.intersection(*(set(data[p]) for p in parts + ["plain"])))
            if not keys:
                continue
            pairs = [(pool([data[p][k]["raw"] for p in parts]), data["plain"][k]["label"]) for k in keys]
            base = [(data["plain"][k]["raw"], data["plain"][k]["label"]) for k in keys]
            m, m0 = metrics(pairs), metrics(base)
            lo, hi = boot_diff(m["right"], m0["right"]) if arm != "plain" else (0, 0)
            first = {k[0]: k for k in keys}  # usage is per request; take it from one answer per item
            cost = sum(data[p][k]["usage"]["cost_usd"] for p in parts for k in first.values()) / len(first)
            table.setdefault(arm, {})[setname] = {k: v for k, v in m.items() if k != "right"} | {"d_acc_vs_plain": m["acc"] - m0["acc"], "ci90": (lo, hi), "cost": cost}
            print(f"  {arm:40s} n {m['n']:5d} acc {m['acc']:.3f} ({m['acc'] - m0['acc']:+.3f} [{lo:+.3f},{hi:+.3f}]) "
                  f"ece {m['ece']:.3f} auroc {m['auroc'] or 0:.3f} brier {m['brier']:.3f} $/req {cost:.4f}")
    print("== mean over sets present")
    for arm, per in table.items():
        if len(per) == len([s for s in SETS if any(s in v for v in table.values())]):
            k = len(per)
            print(f"  {arm:40s} acc {sum(v['acc'] for v in per.values()) / k:.3f} "
                  f"d {sum(v['d_acc_vs_plain'] for v in per.values()) / k:+.3f} "
                  f"ece {sum(v['ece'] for v in per.values()) / k:.3f} auroc {sum((v['auroc'] or 0) for v in per.values()) / k:.3f} "
                  f"brier {sum(v['brier'] for v in per.values()) / k:.3f}")
    (HERE / "results" / f"prompts_summary_{a.split}.json").write_text(json.dumps(table, indent=1) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "summary"])
    ap.add_argument("--angles", default="plain,literal,sceptic")
    ap.add_argument("--sets", default=",".join(SETS))
    ap.add_argument("--split", default="dev", choices=["dev", "test", "all"])
    ap.add_argument("--effort", default="low")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--stacks", default="")
    ap.add_argument("--all-stacks", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="", help="run decide.py's k-sample stack and save under this name")
    ap.add_argument("--k", type=int, default=3)
    a = ap.parse_args()
    run(a) if a.cmd == "run" else summary(a)
