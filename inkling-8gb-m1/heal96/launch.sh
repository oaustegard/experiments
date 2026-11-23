#!/bin/bash
# launch.sh SESSION : push data, teacher state and scripts to a granted runtime and start pipeline_r3 + guard cell
S=$1; cd /tmp/claude-0/ink
q(){ grep -v -i "keyring\|warning" | cut -c1-70; }
./rs $S 'mkdir -p /content/tk /content/run /content/in' >/dev/null
for f in setup_remote.sh heal.py pipeline_r3.sh; do timeout 600 colab upload -s $S $f /content/$f 2>&1 | q; done
./rs $S 'cd /content && chmod +x setup_remote.sh pipeline_r3.sh && (WITH_LLAMA=0 setsid nohup ./setup_remote.sh >/dev/null 2>&1 </dev/null &) && echo setup-started'
for f in relay/part_0* up2/tkp_*; do timeout 600 colab upload -s $S $f /content/in/$(basename $f) 2>&1 | q; done
./rs $S 'cd /content && cat in/part_0* | tar x && cat in/tkp_* | tar x && ls data | wc -l && ls tk | wc -l && (setsid nohup ./pipeline_r3.sh >/dev/null 2>&1 </dev/null &) && echo pipeline-started'
printf '%s\n' 'import os, time, pathlib' 'pathlib.Path("/content/HOLD").touch()' 't0 = time.time()' 'while os.path.exists("/content/HOLD") and time.time() - t0 < 14*3600: time.sleep(60)' | timeout 40 colab exec --timeout 15 -s $S >/dev/null 2>&1
./rs $S 'ls /content/HOLD && echo guard-on'
