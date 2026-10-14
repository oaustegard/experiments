# Replay video

`render.py` replays two team mailbox logs (round 2 `mail-plain-1`, round 4
`mail-gate-1`) as one 92-second 1280x720 MP4, one message per 1.5 s, drawn
with PIL and piped to ffmpeg. Fonts are the muninn.austegard.com site fonts
(Fraunces, Source Serif 4, DM Mono) converted from woff2 to TTF into
`fonts/`. Pill states (received, refused, in code, acknowledged, test
fails/passes) are annotated by hand per message in `RUNS`, from the logs in
`runs/r2/mail-plain-1/mail-log.txt` and `runs/r4/mail-gate-1/mail-log.txt`.

    python3 render.py <dir containing mail-plain-1/.board and mail-gate-1/.board> out.mp4

Published copy: https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/muninn/video/2026/10/haiku-agent-teams.mp4
