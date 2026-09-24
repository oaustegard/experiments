import sys, time, render as R
from cairokit import contact_sheet
fr = [float(x) for x in sys.argv[2].split(",")]
keys = sys.argv[1].split(",")
files = []
for i, S in enumerate(R.TL):
    if S["key"] not in keys: continue
    for f in fr:
        T = R.STARTS[i] + S["dur"] * f; t0 = time.time()
        R.frame(T).write_to_png(fn := f"pv_{S['key']}_{f}.png"); files.append(fn)
        print(fn, f"{time.time()-t0:.2f}s")
contact_sheet(files, "sheet.png", cols=len(fr), scale=0.5)
