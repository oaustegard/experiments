#!/usr/bin/env python3
"""Touch each Colab runtime over ssh every 5 min: runtimes left without client contact for ~30-45 min were
reclaimed server-side (2026-11-23), guard cell or not. Exits at 105 min (under the 2 h background cap)."""
import subprocess, sys, time
t0 = time.time()
while time.time() - t0 < 105 * 60:
    for s in sys.argv[1].split(","):
        r = subprocess.run(["colab", "ssh", "-s", s, "--", "date +%T; uptime | cut -c1-40"], stdin=subprocess.DEVNULL,
                           capture_output=True, text=True, timeout=120)
        print(time.strftime("%H:%M:%S"), s, (r.stdout.strip() or r.stderr.strip()[-120:]).replace("\n", " "), flush=True)
    time.sleep(300)
print("PINGER_EXIT", flush=True)
