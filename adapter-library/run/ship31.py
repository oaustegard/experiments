"""Ship the ±31 base and the refit spikes (round 3 of the ±31 test, results/refit31_paired.json).

    python3 -I run/ship31.py

The ±31 base goes to web/models/base31/ beside the ±63 base, which stays at web/models/base/ for spikes
trained on it (anything exported from decider-lora-creator before this). Each set's seven spikes are
replaced by their ±31 refits (out/base31/spikes), each spike's .json gains "base": "base31", and system.json
points at ../base31/ with a new version. The routers keep their files: on ±31 they routed 400/400 and
955/955 unchanged (results/base31.json). fixtures.json is rebuilt the way pack_system.py --fixtures builds
it (2 test cases per spike, first question; LoraRef logits and pooled keys), so check.html compares the
page with the NumPy spec of the new files.
"""
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from lora import stock_path  # noqa: E402
from pack_system import WEB, LoraRef, everyday_cases  # noqa: E402
from shared import BD, ROOT, examples, load_cases  # noqa: E402
sys.path.insert(0, str(BD / "run"))
from train_enc import Enc  # noqa: E402

VERSIONS = {"workflows": "0.2.0", "everyday": "0.3.0"}
B31 = ROOT / "out" / "base31"


def main():
    paired = json.loads((ROOT / "results" / "refit31_paired.json").read_text())
    if not paired.get("ship"):
        sys.exit("results/refit31_paired.json does not pass the rule; not shipping")
    dst = WEB / "base31"
    dst.mkdir(exist_ok=True)
    shutil.copyfile(B31 / "weights.bin", dst / "weights.bin")
    shutil.copyfile(B31 / "manifest.json", dst / "manifest.json")
    shutil.copyfile(WEB / "base" / "tokenizer.json", dst / "tokenizer.json")
    man = json.loads((dst / "manifest.json").read_text())
    ref = LoraRef(man, (dst / "weights.bin").read_bytes())
    tok = Enc(Path(stock_path()) / "tokenizer.json")
    for set_, version in VERSIONS.items():
        sysj = json.loads((WEB / set_ / "system.json").read_text())
        fx = {"version": version, "base": "base31", "cases": []}
        for s in sysj["spikes"]:
            w = s["id"]
            sm = json.loads((B31 / "spikes" / set_ / f"{w}.json").read_text())
            sb = (B31 / "spikes" / set_ / f"{w}.bin").read_bytes()
            sm["base"] = "base31"
            (WEB / set_ / f"{s['file']}.json").write_text(json.dumps(sm))
            (WEB / set_ / f"{s['file']}.bin").write_bytes(sb)
            ref.use(sm, sb)
            cases = (load_cases(w, "test") if set_ == "workflows" else everyday_cases(w, "test"))[:2]
            te = [e for e in examples(cases) if e["qname"] == examples(cases)[0]["qname"]]
            for e in te[:2]:
                ids, pos = tok.build(e)
                start = ids.index(tok.sep, pos[-1]) + 1
                z, key = ref.run(ids, pos, (start, len(ids) - 1))
                fx["cases"].append({"spike": w, "ids": ids, "pos": pos, "pool": [start, len(ids) - 1], "logits": z.tolist(), "key8": key[:8].tolist()})
            print(f"{set_} {w}: refit spike installed ({len(sb) / 1e3:.0f} kB), refit {sm.get('refit')}", flush=True)
        sysj["base"] = "../base31/"
        sysj["version"] = version
        sysj["base_note"] = "±31 base since 2026-11-06 (adapter-library RESULTS, the ±31 rounds); ../base/ is the ±63 base older spikes were trained on"
        (WEB / set_ / "system.json").write_text(json.dumps(sysj, indent=1))
        (WEB / set_ / "fixtures.json").write_text(json.dumps(fx))
        print(f"{set_}: system.json -> ../base31/, version {version}, {len(fx['cases'])} fixtures", flush=True)


if __name__ == "__main__":
    main()
