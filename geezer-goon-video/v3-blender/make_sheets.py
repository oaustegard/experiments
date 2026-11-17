"""Contact sheets of the per-shot review frames:  python3 make_sheets.py frames_dir out_dir"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from PIL import Image, ImageDraw, ImageFont
import shots
fdir, odir = sys.argv[1], sys.argv[2]
os.makedirs(odir, exist_ok=True)
fr = [(int((s['t0'] + 0.6 * (s['t1'] - s['t0'])) * 30), s) for s in shots.SHOTS if s['t1'] - s['t0'] < 30]
font = ImageFont.truetype(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'fonts', 'bebasneue.ttf'), 26)
tw = 480; th = 270
for k in range(0, len(fr), 12):
    sheet = Image.new('RGB', (tw * 4, th * 3), (0, 0, 0))
    for j, (f, s) in enumerate(fr[k:k + 12]):
        p = next((q for q in (os.path.join(fdir, f'f_{f:05d}.jpg'), os.path.join(fdir, f'f_{f:05d}.png')) if os.path.exists(q)), None)
        if p is None:
            continue
        im = Image.open(p).convert('RGB').resize((tw, th), Image.LANCZOS)
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, 230, 30), fill=(0, 0, 0))
        d.text((6, 2), f"#{k + j}  {s['t0']:.1f}-{s['t1']:.1f}  {s['kind']}", fill=(255, 255, 255), font=font)
        sheet.paste(im, ((j % 4) * tw, (j // 4) * th))
    sheet.save(os.path.join(odir, f'sheet_{k // 12}.png'))
print('sheets', len(range(0, len(fr), 12)))
