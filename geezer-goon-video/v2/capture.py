"""Headless-Chromium frame capture for web/index.html.

  python3 capture.py still T [T ...]          -> stills/still_<T>.jpg
  python3 capture.py chunk F0 F1 out.mp4      -> frames [F0, F1) encoded
"""
import base64, subprocess, sys, time, os, json
from playwright.sync_api import sync_playwright

CHROME = '/opt/pw-browsers/chromium-1194/chrome-linux/chrome'
URL = 'file://' + os.path.abspath('web/index.html')
FPS = 30


def open_page(p):
    b = p.chromium.launch(executable_path=CHROME, args=['--no-sandbox', '--disable-gpu', '--allow-file-access-from-files',
                                                         '--force-color-profile=srgb', '--font-render-hinting=none'])
    pg = b.new_page(viewport={'width': 1920, 'height': 1080}, device_scale_factor=1)
    logs = []
    pg.on('console', lambda m: logs.append(m.text)); pg.on('pageerror', lambda e: logs.append('PAGEERROR ' + str(e)))
    pg.goto(URL); pg.wait_for_function('window.ready !== undefined'); pg.evaluate('window.ready')
    return b, pg, logs


def grab(pg, t, q=0.93):
    return pg.evaluate("(t)=>{renderFrame(t);return document.getElementById('c').toDataURL('image/jpeg',%s)}" % q, t)


if __name__ == '__main__':
    mode = sys.argv[1]
    with sync_playwright() as p:
        b, pg, logs = open_page(p)
        if mode == 'still':
            os.makedirs('stills', exist_ok=True)
            for a in sys.argv[2:]:
                t0 = time.time(); u = grab(pg, float(a), 0.95)
                open(f'stills/still_{a}.jpg', 'wb').write(base64.b64decode(u.split(',', 1)[1])); print(a, round(time.time() - t0, 2), 's')
        elif mode == 'chunk':
            f0, f1, out = int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
            ff = subprocess.Popen(['ffmpeg', '-hide_banner', '-loglevel', 'error', '-y', '-f', 'image2pipe', '-c:v', 'mjpeg', '-r', str(FPS), '-i', '-',
                                   '-c:v', 'libx264', '-preset', 'medium', '-crf', '18', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
            t0 = time.time()
            for f in range(f0, f1):
                ff.stdin.write(base64.b64decode(grab(pg, f / FPS).split(',', 1)[1]))
                if (f - f0) % 150 == 0: print(out, f, round(time.time() - t0), 's', flush=True)
            ff.stdin.close(); ff.wait()
        for l in logs: print('LOG', l)
        b.close()
