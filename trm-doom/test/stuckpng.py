import json, sys, numpy as np
from PIL import Image
s = open('/tmp/play.out').read(); r = json.loads(s[s.find('{'):])
pal = np.array([[0,0,0],[90,90,90],[230,230,230],[200,120,40],[60,200,60],[40,80,255],[255,40,40]], np.uint8)
tiles = []
for e in r['events']:
    if e.get('grid') is None: continue
    g = np.array(e['grid'], np.uint8).reshape(32, 32); img = pal[g].copy()
    for o in e['path']: img[o // 32, o % 32] = [255, 220, 0] if g[o // 32, o % 32] not in (5, 6) else img[o // 32, o % 32]
    i = int((e['x'] - e['ox']) // 24); j = int((e['y'] - e['oy']) // 24)
    if 0 <= i < 32 and 0 <= j < 32: img[j, i] = [255, 0, 255]
    tiles.append(np.kron(img[::-1], np.ones((8, 8, 1), np.uint8)))
    print(e['tic'], e['kind'], e['x'], e['y'], 'angle', e['angle'], 'pathlen', len(e['path']))
if tiles: Image.fromarray(np.concatenate(tiles[:6], 1)).save(sys.argv[1])
