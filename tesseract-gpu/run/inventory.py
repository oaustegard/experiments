#!/usr/bin/env python3
"""Inventory the LSTM network spec of every .traineddata in tessdata_best and tessdata_fast.

Lists each repo with an anonymous blobless clone, downloads each model to
/tmp/tessinv/ (4 at a time, with retries), reads component 17 (the LSTM network)
and component 23 (the version string), deletes the download, and writes
results/inventory.json and notes/inventory.md.
"""
import concurrent.futures as cf
import datetime
import json
import random
import re
import shutil
import struct
import subprocess
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "inventory.json"
NOTES = ROOT / "notes" / "inventory.md"
WORK = Path("/tmp/tessinv")
REPOS = {"best": "tessdata_best", "fast": "tessdata_fast"}
WORKERS = 4
ATTEMPTS = 4
NUM_ENTRIES = 24
COMP_LSTM = 17
COMP_VERSION = 23
COMMON_HEIGHT = 36

# One layer token: O1c-style output, or a type name (capitals then lowercase), then optional args.
TOKEN = r"(?:O\d[cls]|[A-Z][a-z]*)(?:\d+(?:,\d+)*)?"
TOKEN_RE = re.compile(TOKEN)
BODY_RE = re.compile(rf"(?:{TOKEN})*")
HEADER_RE = re.compile(r"\[1,(\d+),(\d+),(\d+)")
HEADER_BYTES_RE = re.compile(rb"\[1,\d+,\d+,\d+")
RUN_BYTES_RE = re.compile(rb"[A-Za-z0-9,\[\]]*")


def layer_type(tok):
    return re.match(r"O\d[cls]|[A-Z][a-z]*", tok).group(0)


def parse_spec(spec):
    """Return (height, layers, output_classes) for a complete spec string, else None."""
    m = HEADER_RE.match(spec)
    if not m or not spec.endswith("]"):
        return None
    body = spec[m.end():-1]
    if not BODY_RE.fullmatch(body):
        return None
    layers = [t.group(0) for t in TOKEN_RE.finditer(body)]
    out = None
    if layers and layers[-1].startswith("O"):
        args = layers[-1][len(layer_type(layers[-1])):]
        out = int(args.split(",")[0]) if args else None
    return int(m.group(1)), layers, out


def find_spec(blob):
    """Longest bracketed spec in the blob that parses in full, or None."""
    best = None
    for hm in HEADER_BYTES_RE.finditer(blob):
        start = hm.start()
        end = RUN_BYTES_RE.match(blob, hm.end()).end()
        cand = blob[start:end]
        # longest prefix ending at a ']' that parses
        for idx in range(len(cand) - 1, -1, -1):
            if cand[idx] != ord("]"):
                continue
            text = cand[: idx + 1].decode("ascii")
            parsed = parse_spec(text)
            if parsed:
                if best is None or len(text) > len(best[0]):
                    best = (text, parsed)
                break
    return best


def read_offsets(data):
    n = struct.unpack_from("<I", data, 0)[0]
    if n != NUM_ENTRIES:
        raise ValueError(f"num_entries={n}, expected {NUM_ENTRIES}")
    return struct.unpack_from(f"<{n}q", data, 4)


def component_bytes(data, offs, i):
    if offs[i] < 0:
        return None
    end = len(data)
    for j in range(i + 1, len(offs)):
        if offs[j] >= 0:
            end = offs[j]
            break
    return data[offs[i]:end]


def fetch(url, dest):
    last = "unknown error"
    for attempt in range(1, ATTEMPTS + 1):
        try:
            subprocess.run(["curl", "-sSLf", "--max-time", "240", "-o", str(dest), url],
                           check=True, capture_output=True, text=True)
            return
        except subprocess.CalledProcessError as e:
            last = (e.stderr or "").strip() or f"curl exit {e.returncode}"
            if attempt < ATTEMPTS:
                time.sleep(min(30, 2 ** attempt) + random.random())
    raise RuntimeError(last)


def list_models(variant):
    repo = REPOS[variant]
    clone = WORK / "clones" / repo
    if clone.exists():
        shutil.rmtree(clone)
    clone.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "clone", "--depth", "1", "--filter=blob:none", "--no-checkout",
                    f"https://github.com/tesseract-ocr/{repo}", str(clone)],
                   check=True, capture_output=True, text=True)
    out = subprocess.run(["git", "-C", str(clone), "ls-tree", "-r", "--name-only", "HEAD"],
                         check=True, capture_output=True, text=True).stdout
    return sorted(p for p in out.splitlines() if p.endswith(".traineddata"))


def process(variant, path):
    repo = REPOS[variant]
    url = f"https://raw.githubusercontent.com/tesseract-ocr/{repo}/main/{path}"
    dest = WORK / variant / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    row = {"variant": variant, "repo": repo, "path": path, "name": Path(path).stem,
           "spec": None, "input_height": None, "layers": None, "output_classes": None,
           "bytes": None, "component17_bytes": None, "version": None,
           "version_has_spec": None, "error": None}
    try:
        fetch(url, dest)
        data = dest.read_bytes()
        row["bytes"] = len(data)
        offs = read_offsets(data)
        lstm = component_bytes(data, offs, COMP_LSTM)
        if lstm is not None:
            row["component17_bytes"] = len(lstm)
        ver = component_bytes(data, offs, COMP_VERSION)
        if ver is not None:
            row["version"] = ver.rstrip(b"\0").decode("utf8", "replace")
        if lstm is None:
            raise ValueError("component 17 absent")
        found = find_spec(lstm)
        if found is None:
            raise ValueError("no parseable spec in component 17")
        text, (height, layers, out) = found
        row.update(spec=text, input_height=height, layers=layers, output_classes=out,
                   version_has_spec=text in (row["version"] or ""))
    except Exception as e:  # a failure degrades to None, never to a guessed value
        row["error"] = f"{type(e).__name__}: {e}"[:300]
    finally:
        dest.unlink(missing_ok=True)
    return row


def render_notes(rows):
    today = datetime.date.today().isoformat()
    ok_rows = [r for r in rows if r["spec"]]
    lines = ["# Tesseract tessdata LSTM inventory", "",
             f"Source: `tesseract-ocr/tessdata_best` and `tesseract-ocr/tessdata_fast`, branch `main`, "
             f"listed {today}. Generated by `run/inventory.py`; one row per model in `results/inventory.json`.", "",
             "## Models per variant", "",
             "| variant | .traineddata files | with spec | without spec | in script/ |",
             "|---|---:|---:|---:|---:|"]
    for v in REPOS:
        vr = [r for r in rows if r["variant"] == v]
        ok = sum(1 for r in vr if r["spec"])
        sc = sum(1 for r in vr if r["path"].startswith("script/"))
        lines.append(f"| {v} | {len(vr)} | {ok} | {len(vr) - ok} | {sc} |")

    specs = Counter(r["spec"] for r in ok_rows)
    lines += ["", f"## Distinct specs ({len(specs)})", "",
              "| spec | models | best | fast | input height | output classes | example |",
              "|---|---:|---:|---:|---:|---:|---|"]
    for spec, n in sorted(specs.items(), key=lambda kv: (-kv[1], kv[0])):
        rs = [r for r in ok_rows if r["spec"] == spec]
        nb = sum(1 for r in rs if r["variant"] == "best")
        lines.append(f"| `{spec}` | {n} | {nb} | {n - nb} | {rs[0]['input_height']} | "
                     f"{rs[0]['output_classes']} | {rs[0]['variant']}/{rs[0]['path']} |")

    type_models, type_occ = Counter(), Counter()
    hidden, outs = [], []
    for r in ok_rows:
        seen = set()
        for tok in r["layers"]:
            t = layer_type(tok)
            type_occ[t] += 1
            seen.add(t)
            args = tok[len(t):]
            if t.startswith("L") and args:
                hidden.append(int(args.split(",")[0]))
        for t in seen:
            type_models[t] += 1
        if r["output_classes"] is not None:
            outs.append(r["output_classes"])
    lines += ["", "## Layer types", "",
              "Types as they appear in the specs; `models` counts models containing the type.", "",
              "| type | models | occurrences |", "|---|---:|---:|"]
    for t in sorted(type_occ, key=lambda t: (-type_models[t], t)):
        lines.append(f"| `{t}` | {type_models[t]} | {type_occ[t]} |")

    lines += ["", "## Sizes", ""]
    if hidden:
        lines.append(f"- LSTM hidden sizes (first arg of `L*` layers): min {min(hidden)}, max {max(hidden)}, "
                     f"distinct {sorted(set(hidden))}")
    if outs:
        lines.append(f"- Output sizes (final `O*` arg, as the spec states it): min {min(outs)}, max {max(outs)}, "
                     f"distinct {sorted(set(outs))}")
    heights = Counter(r["input_height"] for r in ok_rows)
    lines.append(f"- Input heights: {dict(sorted(heights.items()))}")

    skel = lambda r: tuple(layer_type(t) for t in r["layers"])
    common, common_n = Counter(skel(r) for r in ok_rows).most_common(1)[0]
    lines += ["", "## Common pattern and deviations", "",
              f"Common layer-type sequence ({common_n} of {len(ok_rows)} models): "
              f"`{' '.join(common)}`", ""]
    dev = [r for r in ok_rows if r["input_height"] != COMMON_HEIGHT or skel(r) != common]
    if dev:
        lines += ["| model | spec | why |", "|---|---|---|"]
        for r in dev:
            why = []
            if r["input_height"] != COMMON_HEIGHT:
                why.append(f"input height {r['input_height']}")
            if skel(r) != common:
                why.append(f"layers {' '.join(skel(r))}")
            lines.append(f"| {r['variant']}/{r['path']} | `{r['spec']}` | {'; '.join(why)} |")
    else:
        lines.append("No model deviates from the common pattern or has input height other than 36.")
    same_shape = [r for r in ok_rows if skel(r) == common and r["input_height"] == COMMON_HEIGHT]
    other_sizes = sum(1 for r in same_shape if r["spec"] != common_spec(ok_rows, common))
    lines += ["", f"{len(same_shape)} models have the common sequence at height 36; {other_sizes} of them "
                  f"differ from the most common spec only in sizes."]

    failed = [r for r in rows if not r["spec"]]
    lines += ["", f"## Models without a spec ({len(failed)})", ""]
    for r in failed:
        lines.append(f"- {r['variant']}/{r['path']}: {r['error']}")
    if not failed:
        lines.append("None.")
    return "\n".join(lines) + "\n"


def common_spec(ok_rows, common):
    """Most frequent spec among models with the common layer sequence."""
    c = Counter(r["spec"] for r in ok_rows if tuple(layer_type(t) for t in r["layers"]) == common)
    return c.most_common(1)[0][0] if c else None


def main():
    WORK.mkdir(parents=True, exist_ok=True)
    jobs = [(v, p) for v in REPOS for p in list_models(v)]
    print(f"{len(jobs)} .traineddata files listed", flush=True)
    rows = []
    with cf.ThreadPoolExecutor(max_workers=WORKERS) as ex:
        futs = [ex.submit(process, v, p) for v, p in jobs]
        for i, f in enumerate(cf.as_completed(futs), 1):
            rows.append(f.result())
            if i % 20 == 0 or i == len(futs):
                print(f"{i}/{len(futs)} done", flush=True)
    rows.sort(key=lambda r: (r["variant"], r["path"]))
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    RESULTS.write_text(json.dumps(rows, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    NOTES.parent.mkdir(parents=True, exist_ok=True)
    NOTES.write_text(render_notes(rows), encoding="utf-8")
    shutil.rmtree(WORK / "clones", ignore_errors=True)
    print(f"wrote {RESULTS} ({len(rows)} rows, {sum(1 for r in rows if r['spec'])} with spec)")


if __name__ == "__main__":
    main()
