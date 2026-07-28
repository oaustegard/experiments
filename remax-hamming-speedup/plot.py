"""Render the latency sweep from results.json to latency.png (log-log)."""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

r = json.load(open("results.json"))
rows = r["results"]
ns = [x["n"] for x in rows]

series = [
    ("lut_baseline",   "LUT gather (current)", "#c0392b", "o"),
    ("bitcount_u8",    "bitwise_count uint8",  "#e67e22", "s"),
    ("bitcount_u64",   "bitwise_count uint64 (chosen)", "#27ae60", "D"),
    ("pm1_matmul_f32", "±1 BLAS matmul",       "#8e44ad", "^"),
    ("float_cosine",   "float cosine (BLAS target)", "#2980b9", "x"),
]

fig, ax = plt.subplots(figsize=(8, 5.2))
for key, label, color, marker in series:
    xs = [n for n, row in zip(ns, rows) if key in row]
    ys = [row[key] for row in rows if key in row]
    style = dict(color=color, marker=marker, lw=2, ms=6)
    if key == "float_cosine":
        style.update(ls="--", lw=2)
    if key == "bitcount_u64":
        style.update(lw=3, ms=8, zorder=5)
    ax.plot(xs, ys, label=label, **style)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel("corpus size N (documents)")
ax.set_ylabel("latency per query (ms, best-of-40)")
ax.set_title("1-bit Hamming scan kernels vs BLAS float cosine\n"
             "d=512·k=4 → 256 B/row, numpy 2.4.4, single-thread BLAS")
ax.grid(True, which="both", ls=":", alpha=0.4)
ax.legend(fontsize=9, loc="upper left")
fig.tight_layout()
fig.savefig("latency.png", dpi=130)
print("wrote latency.png")
