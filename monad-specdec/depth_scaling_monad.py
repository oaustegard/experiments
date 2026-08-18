"""Per-layer marginal decode cost for Monad (width 256) vs Baguettotron (576)."""
import json, statistics, time, torch
from transformers import AutoModelForCausalLM, AutoTokenizer

torch.set_num_threads(4)
rows = {}
for name, repo, depths in [("monad", "PleIAs/Monad", [8, 16, 32, 48, 64]),
                           ("baguettotron", "PleIAs/Baguettotron", [10, 20, 40, 60, 80])]:
    tok = AutoTokenizer.from_pretrained(repo)
    model = AutoModelForCausalLM.from_pretrained(repo, dtype=torch.float32).eval()
    full = list(model.model.layers)
    ids = tok("The Roman aqueducts were built to carry water into cities and",
              add_special_tokens=False, return_tensors="pt").input_ids
    pts = []
    for n in depths:
        model.model.layers = torch.nn.ModuleList(full[:n])
        model.config.num_hidden_layers = n
        with torch.no_grad():
            samples = []
            for rep in range(3):
                out = model(ids, use_cache=True)
                past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                steps = []
                for _ in range(16):
                    s = time.perf_counter()
                    out = model(nxt, past_key_values=past, use_cache=True)
                    past, nxt = out.past_key_values, out.logits[:, -1:].argmax(-1)
                    steps.append((time.perf_counter() - s) * 1000)
                if rep:
                    samples.extend(steps)
        pts.append((n, round(statistics.median(samples), 3)))
        print(name, pts[-1], flush=True)
    # Least-squares line: ms = intercept + slope * layers
    xs = [p[0] for p in pts]; ys = [p[1] for p in pts]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    rows[name] = {"width": model.config.hidden_size, "points": pts,
                  "ms_per_layer": round(slope, 4),
                  "fixed_ms": round(my - slope * mx, 3)}
    del model

rows["per_layer_ratio_bag_over_monad"] = round(
    rows["baguettotron"]["ms_per_layer"] / rows["monad"]["ms_per_layer"], 3)
rows["width_ratio_bag_over_monad"] = round(
    rows["baguettotron"]["width"] / rows["monad"]["width"], 3)
rows["note"] = ("A layer's FLOPs scale with width^2. If decode were compute-bound the "
                "per-layer ratio would approach the square of the width ratio; if it is "
                "overhead-bound the ratio approaches 1.")
print(json.dumps(rows, indent=2))
json.dump(rows, open("depth_scaling_both.json", "w"), indent=2)
