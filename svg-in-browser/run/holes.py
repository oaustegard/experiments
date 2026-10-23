"""Percent of pixels no path covers: repaint the background rect magenta, render with rsvg-convert.

Usage: python3 run/holes.py a.svg [b.svg ...]
"""
import re, subprocess, sys, cv2, numpy as np
for f in sys.argv[1:]:
    s = open(f).read()
    s = re.sub(r'(<rect width="[\d.]+" height="[\d.]+" fill=")#\w+', r'\1#f0f', s, count=1)
    open(f + '.holes.svg', 'w').write(s)
    vb = re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', s)
    subprocess.run(['rsvg-convert', '-w', str(round(float(vb[1]))), '-o', f + '.holes.png', f + '.holes.svg'], check=True)
    im = cv2.imread(f + '.holes.png').astype(int)
    mag = (im[..., 0] > 200) & (im[..., 2] > 200) & (im[..., 1] < 60)
    print(f'{f}: {mag.mean()*100:.3f}% of pixels uncovered, {len(s)//1024} KB')
