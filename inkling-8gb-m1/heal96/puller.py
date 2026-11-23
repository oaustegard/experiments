#!/usr/bin/env python3
"""Pull /content/run/adapters_latest.pt and train_log.jsonl off the trainer every 8 min into the persistent volume.
Exits after 105 min (under the 2 h background cap); relaunch at each check-in."""
import os, subprocess, sys, time
S = sys.argv[1]; DST = "/home/user/claude-workspace/.spokes/inkling-heal/state/run"; os.makedirs(DST, exist_ok=True)
t0, last = time.time(), None
while time.time() - t0 < 105 * 60:
    r = subprocess.run(["colab", "ssh", "-s", S, "--", "stat -c %Y /content/run/adapters_latest.pt; tail -1 /content/train.log"],
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=200)
    lines = r.stdout.split("\n")
    mt = lines[0].strip() if lines and lines[0].strip().isdigit() else None
    if mt and mt != last:
        ok = subprocess.run(["colab", "download", "-s", S, "/content/run/adapters_latest.pt", f"{DST}/adapters_latest.pt.tmp"],
                            capture_output=True, text=True, timeout=1500).returncode == 0
        subprocess.run(["colab", "download", "-s", S, "/content/run/train_log.jsonl", f"{DST}/train_log.jsonl"],
                       capture_output=True, text=True, timeout=300)
        if ok:
            os.replace(f"{DST}/adapters_latest.pt.tmp", f"{DST}/adapters_latest.pt"); last = mt
        print(time.strftime("%H:%M:%S"), "pulled" if ok else "pull FAILED", (lines[1] if len(lines) > 1 else "")[:120], flush=True)
    time.sleep(480)
print("PULLER_EXIT", flush=True)
