"""Port the packed Ettin-17m decider to Core ML and time it on the Apple Neural Engine.

    python3 run/port.py            # parity, convert at 512 and 256 tokens, benchmark, compute plan -> results/summary.json

Needs macOS on Apple silicon and: torch 2.5.1, coremltools 8.3.0, transformers 4.51.3, tokenizers, numpy<2.2.
(transformers 5.x refuses torch < 2.6; coremltools 8.3 is tested up to torch 2.5.)
"""
import collections, json, os, platform, subprocess, sys, time, warnings
from pathlib import Path

warnings.filterwarnings("ignore")
import numpy as np
import torch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment

sys.path.insert(0, str(experiment("dettin-68m") / "run"))
sys.path.insert(0, str(HERE / "run"))
import coremltools as ct
from coremltools.models.compute_plan import MLComputePlan
from ane_model import DettinANE
from dettin import Dettin

PACK = experiment("browser-decider") / "web" / "models" / "decider"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
LENGTHS = (512, 256)
RUNS = 200


def softmax(z):
    e = np.exp(z - z.max())
    return e / e.sum()


def main():
    m = Dettin.load(PACK, device="cpu")
    cases = json.loads((PACK / "cases.json").read_text())
    items = []
    for c in cases:
        for n, q in c["questions"].items():
            ids, pos, opts = m._build(c["state"], q)
            g = c["gold"][n]["probabilities"]
            items.append(dict(ids=ids, pos=pos, opts=opts, gold=max(g, key=g.get), T=m.temps.get(q["type"], 1.0)))
    lens = [len(i["ids"]) for i in items]
    res = {"machine": subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout.strip(),
           "ram_gb": int(subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True).stdout) / 2**30,
           "macos": platform.mac_ver()[0], "torch": torch.__version__, "coremltools": ct.__version__,
           "pack": "browser-decider/web/models/decider", "variant": json.loads((PACK / "manifest.json").read_text())["variant"],
           "params_m": round(sum(p.numel() for p in m.enc.parameters()) / 1e6, 2), "questions": len(items),
           "tokens": {"min": min(lens), "median": int(np.median(lens)), "p95": int(np.percentile(lens, 95)), "max": max(lens)}}

    # reference: Hugging Face ModernBertModel in PyTorch on the CPU, unpadded, one question per call
    ref, ts = [], []
    with torch.no_grad():
        for it in items:
            t = time.perf_counter()
            hs = m.enc(input_ids=torch.tensor([it["ids"]]), attention_mask=torch.ones(1, len(it["ids"]), dtype=torch.long)).last_hidden_state
            ref.append(torch.softmax(m.head(hs[0, it["pos"]]).squeeze(-1) / it["T"], -1).numpy())
            ts.append(time.perf_counter() - t)
    res["reference"] = {"accuracy": float(np.mean([it["opts"][int(p.argmax())] == it["gold"] for it, p in zip(items, ref)])),
                        "ms_median": float(np.median(ts) * 1000)}
    print("reference", res["reference"], flush=True)

    res["lengths"] = {}
    pad = m.ids["[PAD]"]
    for S in LENGTHS:
        r = {}
        net = DettinANE(m.enc, m.head, S).eval()
        fit = [(it, p) for it, p in zip(items, ref) if len(it["ids"]) <= S]
        r["fit"] = len(fit)
        r["reference_accuracy_on_fit"] = float(np.mean([it["opts"][int(p.argmax())] == it["gold"] for it, p in fit]))

        def feed(it, np_=True):
            ids = np.full((1, S), pad, dtype=np.int32); ids[0, :len(it["ids"])] = it["ids"]
            pm = np.zeros((1, S), dtype=np.float16); pm[0, :len(it["ids"])] = 1
            return {"input_ids": ids, "pad_mask": pm}

        # 1. the rewritten model in PyTorch (fp32) against the reference
        mx = agree = 0
        with torch.no_grad():
            for it, p in fit:
                f = feed(it)
                z = net(torch.tensor(f["input_ids"], dtype=torch.long), torch.tensor(f["pad_mask"], dtype=torch.float32))[0, it["pos"]] / it["T"]
                q = torch.softmax(z, -1).numpy(); mx = max(mx, float(np.abs(q - p).max())); agree += int(q.argmax() == p.argmax())
        r["torch_rewrite"] = {"max_abs_dp": mx, "top_answer_agree": agree}

        # 2. convert
        t = time.time()
        tr = torch.jit.trace(net, (torch.zeros(1, S, dtype=torch.int32), torch.ones(1, S)))
        ml = ct.convert(tr, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16,
                        inputs=[ct.TensorType("input_ids", shape=(1, S), dtype=np.int32), ct.TensorType("pad_mask", shape=(1, S), dtype=np.float16)],
                        outputs=[ct.TensorType("logits")])
        path = OUT / f"dettin17m_S{S}.mlpackage"
        ml.save(str(path))
        r["convert_s"] = round(time.time() - t, 1)
        r["package_mb"] = round(sum(os.path.getsize(os.path.join(dp, f)) for dp, _, fs in os.walk(path) for f in fs) / 1e6, 1)

        # 3. time and score each compute unit
        r["units"] = {}
        for name, cu in (("cpu", ct.ComputeUnit.CPU_ONLY), ("cpu_gpu", ct.ComputeUnit.CPU_AND_GPU), ("cpu_ne", ct.ComputeUnit.CPU_AND_NE), ("all", ct.ComputeUnit.ALL)):
            t = time.time(); mm = ct.models.MLModel(str(path), compute_units=cu); load = time.time() - t
            for _ in range(5): mm.predict(feed(fit[0][0]))
            ts = []
            for i in range(RUNS):
                f = feed(fit[i % len(fit)][0]); t = time.perf_counter(); mm.predict(f); ts.append(time.perf_counter() - t)
            ts = np.array(ts) * 1000
            acc = agree = 0; mx = 0.0
            for it, p in fit:
                q = softmax(np.asarray(mm.predict(feed(it))["logits"], dtype=np.float32)[0, it["pos"]] / it["T"])
                agree += int(q.argmax() == p.argmax()); acc += it["opts"][int(q.argmax())] == it["gold"]; mx = max(mx, float(np.abs(q - p).max()))
            r["units"][name] = {"load_s": round(load, 2), "ms_median": round(float(np.median(ts)), 3), "ms_p90": round(float(np.percentile(ts, 90)), 3),
                                "ms_min": round(float(ts.min()), 3), "accuracy": acc / len(fit), "top_answer_agree": agree, "max_abs_dp": round(mx, 4)}
            print(S, name, r["units"][name], flush=True)

        # 4. where Core ML plans to run each operation
        mm = ct.models.MLModel(str(path), compute_units=ct.ComputeUnit.ALL)
        plan = MLComputePlan.load_from_path(mm.get_compiled_model_path(), compute_units=ct.ComputeUnit.ALL)
        dev, off, cost = collections.Counter(), collections.Counter(), collections.Counter()
        for op in plan.model_structure.program.functions["main"].block.operations:
            u = plan.get_compute_device_usage_for_mlprogram_operation(op)
            if u is None: continue
            d = type(u.preferred_compute_device).__name__.replace("MLComputeDevice", "").replace("ML", "").replace("ComputeDevice", "")
            dev[d] += 1
            if "Neural" not in d: off[op.operator_name] += 1
            c = plan.get_estimated_cost_for_mlprogram_operation(op)
            if c is not None: cost[d] += c.weight
        r["plan"] = {"ops_by_device": dict(dev), "ops_not_on_neural_engine": dict(off), "estimated_cost_share": {k: round(v, 3) for k, v in cost.items()}}
        print(S, "plan", r["plan"], flush=True)
        res["lengths"][str(S)] = r
    (HERE / "results" / "summary.json").write_text(json.dumps(res, indent=1))
    print("wrote results/summary.json")


if __name__ == "__main__":
    main()
