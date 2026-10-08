# CONTRACT — applies to every scene builder

You build ONE scene of a 2-minute film called "Murmuration" (starlings; how local rules make a flock). Several builders work at the same time and never see each other's scenes.

## 1. Deliverable
* `scenes/<id>.mp4` — H.264, 1920x1080, 30 fps, yuv420p, no audio, exactly N frames (in the scene spec). `-c:v libx264 -preset medium -crf 17 -pix_fmt yuv420p -r 30 -an -movflags +faststart`.
* Five stills extracted from the final MP4 at 10/30/50/70/90 %.
* `work/<id>/README.txt`: how to rebuild.

## 2. Look
Palette only: NIGHT #0b0f24, INDIGO #232a52, ROSE #b9566b, EMBER #f0a35e, BONE #ede6d6 (plus blends/alpha). Font Inter. Text sparse, uppercase, tracked, BONE 80-90 %, >= 34 px (title >= 120 px), >= 96 px from every edge. Quiet, cinematic, slow; ease every move.

## 3. Time
Frame i is time i/30 s and depends only on i and a fixed seed. The first and last 24 frames are cross-dissolved with neighbours: no essential text or action there. Sync to the narration cues in the brief.

## 4. Machine
2 cores shared by about 8 agents: `nice -n 10`, one render at a time, <= 2 threads, detach long renders and poll, preview at 640x360 first, no installs.

## 5. Verify
Frame count/size/fps via ffprobe; view all five stills; `freezedetect` and `blackdetect` clean; at most 3 full renders.

## 6. Stopping rule
After 3 failed attempts at a blocker, stop and report what works, the exact error and what is ruled out.

## 7. Report
Under 200 words: files, verified frame count, render time, three weakest things in the stills, one change with another hour.
