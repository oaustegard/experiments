"""Page image -> text in reading order: line finding, then the GPU-style line recogniser.

    from server.pipeline import read_page
    res = read_page(gray)                     # Tesseract layout, cleaned lines
    res["text"]                               # reading-order text, blank line between blocks
    res["lines"]                              # [{"box": [x0, y0, x1, y1], "text": str, "conf": float}]
    res["ms"]                                 # per-stage milliseconds

gray is a 2-D uint8 page image (PIL images and RGB arrays are converted to grey).

Line finding
  layout="tess"  Tesseract's own page layout through tesserocr (PSM.AUTO, AnalyseLayout, the
                 RIL.TEXTLINE iterator in Tesseract's order). If tesserocr cannot be imported,
                 the system `tesseract` CLI's TSV level-4 rows are used instead and the result
                 says so in "layout_engine".
  layout="det"   PP-OCRv6 text detector (models/det/<tiny|small>/inference.onnx, onnxruntime,
                 CUDA provider when available) -> tessgpu.layout (boxes, lines, reading order).

Recognition
  Each line box is padded by PAD pixels (Tesseract pads a line box by 4 px). With clean=True (the
  default) the crop goes through tessgpu.lineprep.clean_line (flip if light-on-dark, flat-field, tight
  crop to the ink); with clean=False it is only inverted when tessgpu.ref.needs_invert says the line
  is light-on-dark, which is the input Tesseract itself would use. All lines of the page go through
  TorchTess.recognize in one batch. tessgpu.beam.decode gives the unichar ids, which are joined
  into the line text. conf is the mean over timesteps of the highest softmax value.

Failures raise. The only empty results are real ones: a page with no lines gives no lines.
"""

import csv
import io
import os
import subprocess
import sys
import tempfile
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from server.torch_engine import TorchTess  # noqa: E402
from tessgpu import beam, layout, lineprep, ref  # noqa: E402

PAD = 4                                                    # Tesseract's line-box padding, px
DEFAULT_MODEL = os.path.join(ROOT, 'models', 'best', 'eng.traineddata')
DET_MODELS = {
    'tiny': os.path.join(ROOT, 'models', 'det', 'tiny', 'inference.onnx'),
    'small': os.path.join(ROOT, 'models', 'det', 'small', 'inference.onnx'),
}
DET_SIDE = 1280                                            # detector input: longest side, px
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # applied to B, G, R in that order
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_engines = {}      # (model path, device, mode) -> TorchTess
_det_sessions = {}  # (det size) -> onnxruntime.InferenceSession
_tess_api = None   # tesserocr PyTessBaseAPI, created once per process
_tess_warned = False


# ---------------------------------------------------------------- helpers

def _grey(page):
    if isinstance(page, Image.Image):
        return np.asarray(page.convert('L'), dtype=np.uint8)
    arr = np.asarray(page)
    if arr.ndim == 3:
        return np.asarray(Image.fromarray(arr.astype(np.uint8)).convert('L'), dtype=np.uint8)
    if arr.ndim != 2:
        raise ValueError('page must be a 2-D greyscale image')
    return np.ascontiguousarray(arr, dtype=np.uint8)


def _engine(model, device, mode):
    key = (os.path.abspath(model), device, mode)
    if key not in _engines:
        if str(device).startswith('cpu'):
            # Measured on the 4-core container: the LSTMs take 4 threads ~100x longer than one
            # (1.97 s vs 0.026 s for one 40x140 call), so CPU runs pin torch to one thread.
            # TESSGPU_CPU_THREADS overrides it.
            import torch
            torch.set_num_threads(int(os.environ.get('TESSGPU_CPU_THREADS', '1')))
        _engines[key] = TorchTess(model, device=device, mode=mode)
    return _engines[key]


def _tessdata_dir():
    env = os.environ.get('TESSDATA_PREFIX')
    if env and os.path.exists(os.path.join(env, 'eng.traineddata')):
        return env
    import glob
    for pat in ('/usr/share/tesseract-ocr/*/tessdata', '/usr/share/tessdata',
                '/usr/local/share/tessdata'):
        for d in sorted(glob.glob(pat)):
            if os.path.exists(os.path.join(d, 'eng.traineddata')):
                return d
    raise RuntimeError('no tessdata directory with eng.traineddata found; set TESSDATA_PREFIX')


# ---------------------------------------------------------------- layout: Tesseract

def _tess_api_get():
    global _tess_api
    if _tess_api is None:
        from tesserocr import PSM, PyTessBaseAPI
        _tess_api = PyTessBaseAPI(path=_tessdata_dir(), lang='eng', psm=PSM.AUTO)
    return _tess_api


def _layout_tesserocr(gray):
    from tesserocr import RIL
    api = _tess_api_get()
    api.SetImage(Image.fromarray(gray))
    it = api.AnalyseLayout()
    if it is None:                       # Tesseract's answer for an empty page
        return []
    out = []
    while True:
        x0, y0, x1, y1 = it.BoundingBox(RIL.TEXTLINE)
        out.append({'box': [int(x0), int(y0), int(x1), int(y1)],
                    'block_start': bool(it.IsAtBeginningOf(RIL.BLOCK))})
        if not it.Next(RIL.TEXTLINE):
            break
    return out


def _cli_env():
    """Environment for the tesseract CLI: one OpenMP thread. On a busy multi-core box Tesseract's
    OpenMP team spins (a page took >9 CPU-minutes at default threads, ~1.7 s with one)."""
    env = dict(os.environ)
    env.setdefault('OMP_THREAD_LIMIT', '1')
    return env


def _layout_tesseract_cli(gray):
    """TSV level-4 (line) rows from the tesseract command line."""
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'page.png')
        Image.fromarray(gray).save(p)
        tsv = subprocess.run(['tesseract', p, 'stdout', 'tsv', '--psm', '3', '--oem', '1', '-l', 'eng'],
                             capture_output=True, text=True, check=True, env=_cli_env()).stdout
    out = []
    prev_block = None
    for row in csv.DictReader(io.StringIO(tsv), delimiter='\t', quoting=csv.QUOTE_NONE):
        if row['level'] != '4':
            continue
        x0, y0 = int(row['left']), int(row['top'])
        x1, y1 = x0 + int(row['width']), y0 + int(row['height'])
        block = (row['page_num'], row['block_num'])
        out.append({'box': [x0, y0, x1, y1], 'block_start': block != prev_block})
        prev_block = block
    return out


def _layout_tess(gray):
    """Returns (lines in reading order, engine name)."""
    global _tess_warned
    try:
        import tesserocr  # noqa: F401
    except ImportError:
        if not _tess_warned:
            warnings.warn('tesserocr is not installed; using the tesseract CLI TSV for the layout')
            _tess_warned = True
        return _layout_tesseract_cli(gray), 'tesseract-cli'
    return _layout_tesserocr(gray), 'tesserocr'


# ---------------------------------------------------------------- layout: PP-OCRv6 detector

def _det_session(det):
    import onnxruntime as ort
    if det not in DET_MODELS:
        raise ValueError("det must be 'tiny' or 'small', got %r" % (det,))
    if det not in _det_sessions:
        avail = ort.get_available_providers()
        providers = [p for p in ('CUDAExecutionProvider', 'CPUExecutionProvider') if p in avail]
        _det_sessions[det] = ort.InferenceSession(DET_MODELS[det], providers=providers)
    return _det_sessions[det]


def _detect(sess, gray):
    """PP-OCRv6 text map for a page: float32 [Hp, Wp] and the scale from map to page pixels."""
    img = Image.fromarray(gray).convert('RGB')
    w0, h0 = img.size
    ratio = DET_SIDE / max(w0, h0)
    cw, ch = round(w0 * ratio), round(h0 * ratio)
    im = img.resize((cw, ch), Image.BILINEAR)
    bgr = np.asarray(im, dtype=np.float32)[:, :, ::-1] / np.float32(255.0)
    x = (bgr - MEAN) / STD
    Hp = int(np.ceil(ch / 32) * 32)
    Wp = int(np.ceil(cw / 32) * 32)
    pad = np.zeros((Hp, Wp, 3), dtype=np.float32)
    pad[:ch, :cw] = x
    tensor = np.ascontiguousarray(pad.transpose(2, 0, 1)[None])
    out = sess.run(None, {sess.get_inputs()[0].name: tensor})[0]
    prob = out[0, 0].astype(np.float32)
    return prob, w0 / cw, h0 / ch, w0, h0


def _layout_det(gray, det):
    sess = _det_session(det)
    prob, sx, sy, w, h = _detect(sess, gray)
    res = layout.layout(prob, sx, sy, w, h)
    lines = res['lines']
    out = []
    prev_block = None
    for lid in res['order']:
        ln = lines[lid]
        out.append({'box': list(ln['box']), 'block_start': ln['block'] != prev_block})
        prev_block = ln['block']
    return out


# ---------------------------------------------------------------- recognition

def _recognise(gray, boxes, eng, clean=True):
    """Crop, invert, batch-recognise and beam-decode the lines. Returns per-line text and conf."""
    H, W = gray.shape
    t0 = time.perf_counter()
    crops, idx = [], []
    for i, ln in enumerate(boxes):
        x0, y0, x1, y1 = ln['box']
        x0, y0 = max(0, x0 - PAD), max(0, y0 - PAD)
        x1, y1 = min(W, x1 + PAD), min(H, y1 + PAD)
        if x1 <= x0 or y1 <= y0:
            continue
        c = np.ascontiguousarray(gray[y0:y1, x0:x1])
        if clean:
            c = lineprep.clean_line(c)
        elif ref.needs_invert(c):
            c = np.ascontiguousarray(255 - c)
        crops.append(c)
        idx.append(i)
    t_crop = time.perf_counter() - t0

    t0 = time.perf_counter()
    probs = eng.recognize(crops) if crops else []
    t_rec = time.perf_counter() - t0

    t0 = time.perf_counter()
    texts = [''] * len(boxes)
    confs = [None] * len(boxes)
    chars = eng.model.unicharset
    for i, p in zip(idx, probs):
        ids = beam.decode(p, eng.model)
        raw = ''.join(chars[u] for u in ids)
        texts[i] = ' '.join(raw.split())
        confs[i] = float(p.max(axis=1).mean()) if p.shape[0] else None
    t_dec = time.perf_counter() - t0
    return texts, confs, {'crop': t_crop, 'recognise': t_rec, 'decode': t_dec}


# ---------------------------------------------------------------- public

def read_page(gray, layout="tess", model=DEFAULT_MODEL, device="cpu", mode="exact", det="tiny", clean=True):
    """Read one page. layout: "tess" (Tesseract's layout, default) or "det" (PP-OCRv6 lines, kept as a
    reference point). clean: tessgpu.lineprep clean-up of every line before recognition."""
    if layout not in ("det", "tess"):
        raise ValueError("layout must be 'det' or 'tess', got %r" % (layout,))
    g = _grey(gray)
    total0 = time.perf_counter()

    t0 = time.perf_counter()
    if layout == "det":
        boxes = _layout_det(g, det)
        engine = 'det-' + det
    else:
        boxes, engine = _layout_tess(g)
    t_layout = time.perf_counter() - t0

    eng = _engine(model, device, mode)
    texts, confs, t = _recognise(g, boxes, eng, clean)

    lines = []
    blocks = []
    cur = []
    for ln, text, conf in zip(boxes, texts, confs):
        if ln['block_start'] and cur:
            blocks.append(cur)
            cur = []
        lines.append({'box': ln['box'], 'text': text, 'conf': conf})
        if text:
            cur.append(text)
    if cur:
        blocks.append(cur)
    full = '\n\n'.join('\n'.join(b) for b in blocks)

    total = time.perf_counter() - total0
    ms = {
        'layout': round(t_layout * 1000, 1),
        'crop': round(t['crop'] * 1000, 1),
        'recognise': round(t['recognise'] * 1000, 1),
        'decode': round(t['decode'] * 1000, 1),
        'total': round(total * 1000, 1),
    }
    return {'lines': lines, 'text': full, 'ms': ms, 'layout_engine': engine}
