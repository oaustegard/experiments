# Errors

## 1. A restart that left the first training run alive

**What was wrong.** The first launch used 80,000 steps. To cut the run to 60,000 the restart cell ran
`pkill -f 'python -u train.py'` and deleted the run directory. The trainer runs as `python3 -u train.py`,
so the pattern matched nothing and the 80k-step process kept training. Both runs then shared the A100:
the new one logged 53 s per 100 steps instead of 23, and GPU memory doubled from 17.8 to 35.6 GB. Both
wrote to `/content/wid/run`, so the old run's step-5000 checkpoint would have replaced the new run's
`state.pt` and `ema.pt`.

**How it was caught.** The step time in the mirrored log, then `ps` on the VM, which listed both
processes.

**Fix.** Killed the old process by PID 19 minutes after the restart, before either run reached its
first checkpoint. To restart a Colab job, read the PID from `ps`, kill that PID, and check `ps` again
before relaunching.

**Cost.** About 19 minutes of A100 time (roughly 2 compute units) at half speed for the run that was
kept.
