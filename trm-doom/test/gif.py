import json, sys, base64, numpy as np
from PIL import Image, ImageDraw
s = open('/tmp/play.out').read(); r = json.loads(s[s.find('{'):])
pal = np.array([[c & 255, (c >> 8) & 255, (c >> 16) & 255] for c in r['palette']], np.uint8)
ims = [Image.fromarray(pal[np.frombuffer(base64.b64decode(f), np.uint8).reshape(200, 320)]) for f in r['frames']]
ims[0].save(sys.argv[1], save_all=True, append_images=ims[1:], duration=int(sys.argv[2]) if len(sys.argv) > 2 else 200, loop=0)
# contact sheet of every k-th frame
k = max(1, len(ims) // 12); sel = ims[::k][:12]
sheet = Image.new("RGB", (320 * 4, 200 * ((len(sel) + 3) // 4)))
for i, im in enumerate(sel): sheet.paste(im, ((i % 4) * 320, (i // 4) * 200))
sheet.save(sys.argv[1].replace('.gif', '_sheet.png')); print(len(ims), 'frames')
