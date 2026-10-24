# Regenerates the test fixtures render-cpu.js reads. Needs: pip install mediapipe numpy; network for the downloads.
# pose_landmarks.json + pose_mask.json: MediaPipe Pose Landmarker (lite, float16 v1) on pose.jpg (Python mediapipe 0.10.32).
# ap10k_gt.json: ground-truth keypoints from mmpose's tests/data/ap10k/test_ap10k.json.
import json, os, urllib.request, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
D = os.path.dirname(os.path.abspath(__file__))
get = lambda url, f: os.path.exists(os.path.join(D, f)) or urllib.request.urlretrieve(url, os.path.join(D, f))
get("https://storage.googleapis.com/mediapipe-assets/pose.jpg", "pose.jpg")
get("https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/ap10k/000000000004.jpg", "ap10k.jpg")
get("https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/ap10k/000000037516.jpg", "ap10k_37516.jpg")
get("https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/ap10k/test_ap10k.json", "test_ap10k.json")
get("https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", "pose_landmarker_lite.task")
pl = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=os.path.join(D, "pose_landmarker_lite.task")), output_segmentation_masks=True, num_poses=1))
img = mp.Image.create_from_file(os.path.join(D, "pose.jpg"))
r = pl.detect(img)
json.dump({"pose": {"w": img.width, "h": img.height, "lm": [[l.x, l.y, l.z, l.visibility] for l in r.pose_landmarks[0]],
                    "world": [[round(l.x, 4), round(l.y, 4), round(l.z, 4)] for l in r.pose_world_landmarks[0]]}}, open(os.path.join(D, "pose_landmarks.json"), "w"))
m = r.segmentation_masks[0].numpy_view()[..., 0]
q = (np.clip(m, 0, 1) * 15 + 0.5).astype(int).ravel()  # 16 levels, run-length encoded as text
runs, i = [], 0
while i < len(q):
    j = i
    while j < len(q) and q[j] == q[i] and j - i < 65535: j += 1
    runs.append("%x%x," % (q[i], j - i)); i = j
json.dump({"w": m.shape[1], "h": m.shape[0], "levels": 16, "rle": "".join(runs)}, open(os.path.join(D, "pose_mask.json"), "w"))
d = json.load(open(os.path.join(D, "test_ap10k.json")))
json.dump({"categories": d["categories"][0]["keypoints"], "annotations": [{"file": "ap10k.jpg" if a["image_id"] == 4 else "ap10k_37516.jpg", "category": a["category_id"], "keypoints": a["keypoints"]} for a in d["annotations"]]}, open(os.path.join(D, "ap10k_gt.json"), "w"))
print("fixtures written")
