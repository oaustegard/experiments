"""Read a directory of page images: one .txt (reading-order text) and one .json (lines, ms) per page.

    python3 run/read_pages.py data/pages --layout det
    python3 run/read_pages.py data/pages --layout tess --out data/pipeline/tess

--layout  det (PP-OCRv6 lines, default) or tess (Tesseract's own layout)
--det     tiny (default) or small detector, used with --layout det
--out     output directory (default data/pipeline/<layout>)
--model   traineddata for the recogniser (default models/best/eng.traineddata)
--device  torch device for the recogniser (default cpu)
--mode    exact (bit-faithful, default) or cudnn (faster, approximate)
"""

import argparse
import glob
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from server.pipeline import DEFAULT_MODEL, read_page  # noqa: E402

EXTS = ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('pages', help='directory of page images')
    ap.add_argument('--layout', choices=('det', 'tess'), default='det')
    ap.add_argument('--det', choices=('tiny', 'small'), default='tiny')
    ap.add_argument('--out', default=None)
    ap.add_argument('--model', default=DEFAULT_MODEL)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--mode', choices=('exact', 'cudnn'), default='exact')
    args = ap.parse_args(argv)

    src = os.path.abspath(args.pages)
    files = sorted(f for f in glob.glob(os.path.join(src, '*')) if f.lower().endswith(EXTS))
    if not files:
        raise SystemExit('no page images (%s) in %s' % (', '.join(EXTS), src))
    out = os.path.abspath(args.out or os.path.join(ROOT, 'data', 'pipeline', args.layout))
    os.makedirs(out, exist_ok=True)

    for path in files:
        gray = np.asarray(Image.open(path).convert('L'), dtype=np.uint8)
        t0 = time.perf_counter()
        res = read_page(gray, layout=args.layout, model=args.model, device=args.device,
                        mode=args.mode, det=args.det)
        wall = time.perf_counter() - t0
        stem = os.path.splitext(os.path.basename(path))[0]
        with open(os.path.join(out, stem + '.txt'), 'w', encoding='utf-8') as f:
            f.write(res['text'] + '\n')
        doc = dict(res, source=os.path.relpath(path, ROOT), layout=args.layout,
                   det=args.det if args.layout == 'det' else None,
                   model=os.path.relpath(args.model, ROOT), device=args.device, mode=args.mode,
                   wall_ms=round(wall * 1000, 1))
        with open(os.path.join(out, stem + '.json'), 'w', encoding='utf-8') as f:
            json.dump(doc, f, ensure_ascii=False, indent=1)
        print('%-28s lines %4d  %8.1f ms  (%s)' % (stem, len(res['lines']), res['ms']['total'],
                                                 res['layout_engine']))
    print('wrote %d pages to %s' % (len(files), os.path.relpath(out, ROOT)))
    return 0


if __name__ == '__main__':
    sys.exit(main())
