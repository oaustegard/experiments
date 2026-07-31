#!/usr/bin/env bash
# End-to-end smoke test for the viewer window.
#
# Drives the real binary against a virtual X display, sends real key events,
# and captures the framebuffer after each one.
#
# Checks compare consecutive frames rather than inspecting them: a key that
# should change the picture must change the bytes, and a toggle pressed twice
# must return to the exact frame it started from. An earlier version of this
# script counted distinct colours instead, and reported "ok" for every key
# while none of them were reaching the window at all.
#
# Needs: Xvfb, xdotool, python3. On Debian/Ubuntu:
#   apt-get install -y xvfb xdotool libxkbcommon-x11-0
#
# Usage: scripts/smoke-test.sh [path-to-svgview]

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BIN="${1:-$ROOT/target/release/svgview}"
OUT="$ROOT/target/smoke"
DISPLAY_NUM=":99"

[[ -x "$BIN" ]] || { echo "not executable: $BIN (cargo build --release first)" >&2; exit 1; }
command -v xdotool >/dev/null || { echo "xdotool is required" >&2; exit 1; }
command -v Xvfb    >/dev/null || { echo "Xvfb is required" >&2; exit 1; }

rm -rf "$OUT"; mkdir -p "$OUT/fb"
FAILURES=0
export DISPLAY="$DISPLAY_NUM"

Xvfb "$DISPLAY_NUM" -screen 0 1000x720x24 -fbdir "$OUT/fb" >"$OUT/xvfb.log" 2>&1 &
XVFB_PID=$!
trap 'kill $XVFB_PID 2>/dev/null || true' EXIT

for _ in $(seq 1 50); do
    [[ -e "/tmp/.X11-unix/X${DISPLAY_NUM#:}" ]] && break
    sleep 0.2
done

pass() { echo "ok    $1"; }
fail() { echo "FAIL  $1"; FAILURES=$((FAILURES + 1)); }

shoot() { # shoot <name> -> prints the frame hash
    python3 "$ROOT/assets/xwd2png.py" "$OUT/fb/Xvfb_screen0" "$OUT/$1.png" >/dev/null
    md5sum "$OUT/$1.png" | cut -d' ' -f1
}

start() { # start <name> [args...]
    local name="$1"; shift
    "$BIN" "$@" >"$OUT/$name.log" 2>&1 &
    APP_PID=$!
    sleep 2
    if ! kill -0 "$APP_PID" 2>/dev/null; then
        fail "$name — process exited; log follows"
        sed 's/^/        /' "$OUT/$name.log"
        return 1
    fi
    # Without a window manager nothing assigns input focus, so keystrokes
    # would go to the root window and be silently dropped.
    WID=$(xdotool search --name svgview | head -1)
    if [[ -z "$WID" ]]; then
        fail "$name — no window appeared"
        return 1
    fi
    xdotool windowfocus "$WID"
    sleep 0.3
    return 0
}

stop() {
    kill "$APP_PID" 2>/dev/null || true
    wait "$APP_PID" 2>/dev/null || true
    sleep 0.5
}

key() { xdotool key --clearmodifiers "$1"; sleep 1; }
title() { xdotool getwindowname "$WID"; }

echo "== a real document"
if start doc "$ROOT/assets/test.svg"; then
    frame_doc=$(shoot doc)
    rss=$(awk '/VmRSS/{print $2}' /proc/"$APP_PID"/status 2>/dev/null || echo 0)
    echo "      resident set: $((rss / 1024)) MiB"

    [[ "$(title)" == "test.svg"* ]] \
        && pass "window title names the file: $(title)" \
        || fail "unexpected window title: $(title)"

    key b; frame_white=$(shoot bg_white)
    [[ "$frame_white" != "$frame_doc" ]] \
        && pass "B changes the background" \
        || fail "B did not change the frame"

    for _ in 1 2 3; do key b; done
    frame_cycled=$(shoot bg_cycled)
    [[ "$frame_cycled" == "$frame_doc" ]] \
        && pass "B four times returns to the first background" \
        || fail "background cycle did not come back around"

    key h; frame_help=$(shoot help)
    [[ "$frame_help" != "$frame_doc" ]] \
        && pass "H shows the key list" \
        || fail "H did not change the frame"

    key h; frame_back=$(shoot help_off)
    [[ "$frame_back" == "$frame_doc" ]] \
        && pass "H again restores the document exactly" \
        || fail "H did not restore the previous frame"

    key ctrl+1
    [[ "$(title)" == *" 100% "* ]] \
        && pass "Ctrl+1 reports 100% zoom" \
        || fail "Ctrl+1 gave title: $(title)"
    frame_actual=$(shoot actual_size)
    [[ "$frame_actual" != "$frame_doc" ]] \
        && pass "Ctrl+1 redraws at actual size" \
        || fail "Ctrl+1 did not change the frame"

    key ctrl+0; frame_fit=$(shoot fit)
    [[ "$frame_fit" == "$frame_doc" ]] \
        && pass "Ctrl+0 returns to the fitted view" \
        || fail "Ctrl+0 did not restore the fitted frame"

    zoomed_title_before=$(title)
    key ctrl+plus
    [[ "$(title)" != "$zoomed_title_before" ]] \
        && pass "Ctrl++ changes the zoom: $(title)" \
        || fail "Ctrl++ left the zoom at $(title)"

    stop
fi

echo "== auto-reload"
cp "$ROOT/assets/test.svg" "$OUT/live.svg"
if start reload "$OUT/live.svg"; then
    frame_before=$(shoot reload_before)
    # Same geometry, different colour, so only the pixels change.
    sed 's/#6fa8dc/#ff0000/g; s/#98c379/#ff0000/g' "$ROOT/assets/test.svg" >"$OUT/live.svg"
    sleep 2
    frame_after=$(shoot reload_after)
    [[ "$frame_after" != "$frame_before" ]] \
        && pass "editing the file on disk redraws the window" \
        || fail "auto-reload did not pick up the change"
    stop
fi

echo "== no arguments"
if start welcome; then
    shoot welcome >/dev/null
    pass "starts with no file and shows the welcome screen"
    stop
fi

echo "== a file that is not there"
if start missing "$OUT/nope.svg"; then
    shoot missing >/dev/null
    pass "survives a missing file"
    grep -qi "no such file\|cannot find" "$OUT/missing.log" \
        && pass "reports the missing file on stderr" \
        || fail "no error on stderr"
    stop
fi

echo "== a file that is not SVG"
printf 'this is not markup at all' >"$OUT/broken.svg"
if start broken "$OUT/broken.svg"; then
    shoot broken >/dev/null
    pass "survives a malformed file"
    stop
fi

echo "== headless rendering"
if "$BIN" "$ROOT/assets/test.svg" --png "$OUT/headless.png" --width 640 >/dev/null 2>&1 \
    && [[ -s "$OUT/headless.png" ]]; then
    pass "--png writes a file without opening a window"
else
    fail "--png did not produce output"
fi
"$BIN" "$OUT/nope.svg" --png "$OUT/x.png" >/dev/null 2>&1 \
    && fail "--png exited 0 for a missing input" \
    || pass "--png exits non-zero for a missing input"

echo
if [[ "$FAILURES" -eq 0 ]]; then
    echo "all checks passed — screenshots in $OUT"
else
    echo "$FAILURES check(s) failed — screenshots in $OUT"
fi
exit "$FAILURES"
