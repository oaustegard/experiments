# Runs RTMPose-m AP-10K (LiteRT fp16) on the two AP-10K test images with the page's crop, resampling and
# normalisation, and writes both raw SimCC heads to rtm_raw.json for verify-rtmpose.js to decode
# with the page's own decoder. Needs: pip install ai-edge-litert pillow numpy.
# Model: ../web/rtm_animal_fp16.bin (the .tflite from litert-community/RTMPose-Animal-AP10K-LiteRT @ 1daee634).
# Box: the AP-10K ground-truth bbox stands in for the detector box (the page uses EfficientDet-Lite0).
import json, os, sys, numpy as np
from PIL import Image
from ai_edge_litert.interpreter import Interpreter
D = os.path.dirname(os.path.abspath(__file__))
SIZE = 256
RESAMPLE = sys.argv[1] if len(sys.argv) > 1 else "canvas"  # "canvas" (the page) or "pil"
MEAN, STD = np.array([123.675, 116.28, 103.53]), np.array([58.395, 57.12, 57.375])  # web/index.html MEAN/STD
gt = json.load(open(os.path.join(D, "test_ap10k.json")))
files = {4: "ap10k.jpg", 37516: "ap10k_37516.jpg"}
it = Interpreter(os.path.join(D, "..", "web", "rtm_animal_fp16.bin"))
run = it.get_signature_runner("serving_default")
outs = []
for a in gt["annotations"]:
    img = Image.open(os.path.join(D, files[a["image_id"]])).convert("RGB")
    vw, vh = img.size
    bx, by, bw, bh = a["bbox"]
    # same crop as runAnimal(): square of side 1.25*max(w,h) centred on the box, out-of-frame stays black
    s = max(bw, bh) * 1.25; sx, sy = bx + bw / 2 - s / 2, by + bh / 2 - s / 2; k = SIZE / s
    x0, y0, x1, y1 = max(0, sx), max(0, sy), min(vw, sx + s), min(vh, sy + s)
    if RESAMPLE == "canvas":
        # canvas drawImage downscales by plain bilinear sampling, no low-pass filter (METHODS.md,
        # "Canvas drawImage is not an image-model resize"): sample each crop pixel centre bilinearly
        src = np.asarray(img, np.float32)
        u = sx + (np.arange(SIZE) + 0.5) / k; v = sy + (np.arange(SIZE) + 0.5) / k
        inx, iny = (u >= x0) & (u < x1), (v >= y0) & (v < y1)
        fx, fy = np.clip(u - 0.5, 0, vw - 1), np.clip(v - 0.5, 0, vh - 1)
        ix, iy = np.minimum(fx.astype(int), vw - 2), np.minimum(fy.astype(int), vh - 2)
        ax, ay = (fx - ix)[None, :, None], (fy - iy)[:, None, None]
        g = lambda dy, dx: src[(iy + dy)[:, None], (ix + dx)[None, :]]
        c = (g(0, 0) * (1 - ax) + g(0, 1) * ax) * (1 - ay) + (g(1, 0) * (1 - ax) + g(1, 1) * ax) * ay
        crop = np.where((iny[:, None] & inx[None, :])[..., None], np.round(c), 0).astype(np.uint8)
    else:  # PIL's antialiased resize, as model preprocessors do; kept for comparison
        crop = Image.new("RGB", (SIZE, SIZE))
        dx0, dy0, dx1, dy1 = (x0 - sx) * k, (y0 - sy) * k, (x1 - sx) * k, (y1 - sy) * k
        w, h = round(dx1) - round(dx0), round(dy1) - round(dy0)
        crop.paste(img.resize((w, h), Image.BILINEAR, box=(x0, y0, x1, y1)), (round(dx0), round(dy0)))
    x = ((np.asarray(crop, np.float32) - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
    r = run(args_0=x)
    outs.append({"file": files[a["image_id"]], "crop": [sx, sy, s], "size": SIZE,
                 "output_0": r["output_0"].ravel().round(5).tolist(), "output_1": r["output_1"].ravel().round(5).tolist(),
                 "shape": list(r["output_0"].shape)})
json.dump({"resample": RESAMPLE, "outputs_in_graph_order": [o["name"] for o in it.get_output_details()], "runs": outs}, open(os.path.join(D, "rtm_raw.json"), "w"))
print("wrote rtm_raw.json")
