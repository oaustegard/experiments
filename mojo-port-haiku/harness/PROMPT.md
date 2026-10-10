[no-context]
You are porting a Python library to Mojo. Your working directory is WORKDIR.
Start by reading WORKDIR/TASK.md and WORKDIR/MOJO_NOTES.md, then the two
Modular skill files MOJO_NOTES.md names, then the reference source.

Work loop: edit sources in WORKDIR, run `sh WORKDIR/check.sh` (builds, then
prints one JSON line with the dev-split result), fix, repeat. When check.sh
reports all_pass true, run the benchmark command from TASK.md and improve
speed if the port is slower than Python, re-running check.sh after every
change. Keep going until the dev check passes completely and the benchmark
is as good as you can make it; do not stop at a partial port.

Use only Bash, Read, Edit, Write, Grep and Glob. Write only inside WORKDIR.

Finish with a short report: what is ported to Mojo and what stays in
Python, the last check.sh result, the benchmark numbers, and anything you
could not get working.
