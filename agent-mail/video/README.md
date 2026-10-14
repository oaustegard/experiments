# Replay video

`render.py` replays two team mailbox logs (round 2 `mail-plain-1`, round 4
`mail-gate-1`) as one narrated 2:55 1280x720 MP4, drawn with PIL and piped to
ffmpeg. Each narration beat in `narration.json` covers a range of messages; the
beat lasts as long as its voice-over (at least 1.8 s per message), and the clip
starts on the frame where the beat's first message appears. Fonts are the muninn.austegard.com site fonts
(Fraunces, Source Serif 4, DM Mono) converted from woff2 to TTF into
`fonts/`. Pill states (received, refused, in code, acknowledged, test
fails/passes) are annotated by hand per message in `RUNS`, from the logs in
`runs/r2/mail-plain-1/mail-log.txt` and `runs/r4/mail-gate-1/mail-log.txt`.

    python3 render.py <dir containing mail-plain-1/.board and mail-gate-1/.board> out.mp4

Narration: `tts.py` synthesizes each line with Gemini 3.8 Flash TTS in the
Muninn narrator voice (`voice_66w0iod1i7ol`) into `vo/`; `check.py` transcribes
every clip with faster-whisper and diffs it against the script (all differences
were number formatting and name spelling). Gemini 3 Pro then reviewed the final
video for sync, pacing, audio and legibility and reported none.

    python3 tts.py && python3 check.py && python3 render.py <boards dir> out.mp4   # writes out-vo.mp4

Narrated copy: https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/muninn/video/2026/10/haiku-agent-teams-narrated.mp4

Silent first cut: https://pub-38421218e82c4d4ba2bc53974c28b55d.r2.dev/muninn/video/2026/10/haiku-agent-teams.mp4

Bluesky: `bsky_video_reply.py <parent at:// uri> <mp4> <text> <alt>` replies to a Muninn post with a natively uploaded video (video.bsky.app upload, job polling, `app.bsky.embed.video`). Bluesky caps video at 3 minutes. The announcement and its video reply: https://bsky.app/profile/muninn.austegard.com/post/3mxtn32xiel26
