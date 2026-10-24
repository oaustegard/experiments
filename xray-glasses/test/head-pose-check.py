# How well does each head-pose source the page uses fit real heads? Crops the head of every person Pose
# Landmarker (full) finds in the mmpose test images, plus two MediaPipe portraits, then:
#  1. face model: rotation from the facial transformation matrix + least-squares scale and offset, scored by
#     the mean distance between 24 projected canonical landmarks and the detected ones;
#  2. pose fallback (as XR.headFromPose): nose and ears, image xy from the 2D landmarks, depth from the world
#     landmarks (hybrid) or all from the world landmarks (world), scored against the face model's yaw/pitch/roll.
# Needs: pip install mediapipe numpy pillow; network. Downloads go to ./facing-cache (gitignored).
import json, os, urllib.request, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
from PIL import Image, ImageOps
D = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(D, "facing-cache"); os.makedirs(C, exist_ok=True)
MM = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/"
IMAGES = [MM + p for p in [
    "aic/054d9ce9201beffc76e5ff2169d2af2f027002ca.jpg", "aic/fa436c914fe4a8ec1ec5474af4d3820b84d17561.jpg", "aic/ff945ae2e729f24eea992814639d59b3bdec8bd8.jpg",
    "coco/000000000785.jpg", "coco/000000040083.jpg", "coco/000000196141.jpg", "coco/000000197388.jpg", "crowdpose/103319.jpg", "crowdpose/106848.jpg",
    "h36m/S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg", "h36m/S5/S5_SittingDown.54138969/S5_SittingDown.54138969_002061.jpg",
    "h36m/S7/S7_Greeting.55011271/S7_Greeting.55011271_000396.jpg", "h36m/S8/S8_WalkDog_1.55011271/S8_WalkDog_1.55011271_000026.jpg",
    "humanart/real_human/acrobatics/000000000590.jpg", "mpi_inf_3dhp/S4_Seq2_Cam0_001033.jpg", "mpi_inf_3dhp/S8_Seq1_Cam8_002165.jpg",
    "mpi_inf_3dhp/TS1_002001.jpg", "mpi_inf_3dhp/TS2_001850.jpg", "mpii/004645041.jpg", "mpii/005808361.jpg", "mpii/051423444.jpg",
    "mpii/052475643.jpg", "mpii/060754485.jpg", "ochuman/000817.jpg", "ochuman/003799.jpg", "ochuman/003896.jpg",
    "panoptic/005880453_01_l.jpg", "panoptic/ex2_2.flv_000040_l.jpg", "posetrack18/images/val/003418_mpii_test/000000.jpg",
    "posetrack18/images/val/009473_mpii_test/000000.jpg", "posetrack18/images/val/012834_mpii_test/000000.jpg"]]
PORTRAITS = ["https://storage.googleapis.com/mediapipe-assets/portrait.jpg", "https://storage.googleapis.com/mediapipe-assets/business-person.png"]
def get(url, name=None):
    p = os.path.join(C, name or url.replace("://", "_").replace("/", "_"))
    if not os.path.exists(p): urllib.request.urlretrieve(url, p)
    return p
model = lambda u: get(u, os.path.basename(u))
pose_full = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model(
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task")), num_poses=3))
pose_lite = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model(
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task")), num_poses=1))
face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=model(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task")), output_facial_transformation_matrixes=True, num_faces=1))
CAN = np.array([list(map(float, l.split()[1:4])) for l in open(get(
    "https://raw.githubusercontent.com/google-ai-edge/mediapipe/master/mediapipe/modules/face_geometry/data/canonical_face_model.obj")) if l.startswith("v ")])
IDX = [1, 10, 152, 234, 454, 33, 263, 133, 362, 61, 291, 168, 172, 397, 116, 345, 127, 356, 9, 13, 14, 199, 4, 6]  # FACE_IDX in the page
NOSE, EARL, EARR = np.array([0, -1.1, 7.5]), np.array([7.4, 1.0, -4.0]), np.array([-7.4, 1.0, -4.0])  # POSE_HEAD
def frame(a, b):
    x = a / np.linalg.norm(a); y = b - x * (b @ x); y /= np.linalg.norm(y); return np.stack([x, y, np.cross(x, y)], 1)
FC = frame(EARL - EARR, NOSE - (EARL + EARR) / 2)
ypr = lambda R: np.array([np.degrees(np.arctan2(R[0, 2], R[2, 2])), np.degrees(np.arcsin(np.clip(-R[1, 2], -1, 1))), np.degrees(np.arctan2(R[1, 0], R[0, 0]))])
mpimg = lambda im: mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im))

crops = [ImageOps.exif_transpose(Image.open(get(u))).convert("RGB") for u in PORTRAITS]
for u in IMAGES:
    im = ImageOps.exif_transpose(Image.open(get(u))).convert("RGB"); W, H = im.size
    for L in pose_full.detect(mpimg(im)).pose_landmarks:
        e = np.array([[L[i].x * W, L[i].y * H] for i in (0, 2, 5, 7, 8)]); c = e.mean(0)
        s = max(np.hypot((L[11].x - L[12].x) * W, (L[11].y - L[12].y) * H) * 0.9, np.ptp(e[:, 0]) * 2.2, 24)
        if s >= 40: crops.append(im.crop((int(c[0] - s), int(c[1] - s), int(c[0] + s), int(c[1] + s))).resize((384, 384), Image.LANCZOS))
fits, errs = [], {"hybrid": [], "world": []}
for im in crops:
    W, H = im.size; fr = face.detect(mpimg(im))
    if not fr.face_landmarks: continue
    L = fr.face_landmarks[0]; M = np.array(fr.facial_transformation_matrixes[0]); R = M[:3, :3] / np.linalg.norm(M[:3, 0])
    obs = np.array([[L[i].x * W, L[i].y * H] for i in IDX]); q = (CAN[IDX] @ R.T)[:, :2] * [1, -1]
    qm, om = q.mean(0), obs.mean(0); k = ((q - qm) * (obs - om)).sum() / ((q - qm) ** 2).sum()
    res = float(np.linalg.norm(k * (q - qm) + om - obs, axis=1).mean())
    fits.append({"resid_px": round(res, 2), "px_per_cm": round(float(k), 2), "resid_cm": round(res / k, 3), "yaw": round(float(ypr(R)[0]), 1), "pitch": round(float(ypr(R)[1]), 1)})
    pr = pose_lite.detect(mpimg(im))
    if not pr.pose_world_landmarks: continue
    Wl, Ll = pr.pose_world_landmarks[0], pr.pose_landmarks[0]
    kz = np.hypot((Ll[7].x - Ll[8].x) * W, (Ll[7].y - Ll[8].y) * H) / max(np.hypot(Wl[7].x - Wl[8].x, Wl[7].y - Wl[8].y), 1e-6)
    for name, P in (("world", lambda i: np.array([Wl[i].x, -Wl[i].y, -Wl[i].z])), ("hybrid", lambda i: np.array([Ll[i].x * W, -Ll[i].y * H, -Wl[i].z * kz]))):
        R2 = frame(P(7) - P(8), P(0) - (P(7) + P(8)) / 2) @ FC.T
        errs[name].append(np.abs(ypr(R2) - ypr(R)))
out = {"heads_cropped": len(crops), "faces_found": len(fits),
       "face_fit": {"resid_px_min": min(f["resid_px"] for f in fits), "resid_px_max": max(f["resid_px"] for f in fits),
                    "resid_cm_max": max(f["resid_cm"] for f in fits), "resid_cm_median": float(np.median([f["resid_cm"] for f in fits])), "px_per_cm_min": min(f["px_per_cm"] for f in fits),
                    "px_per_cm_max": max(f["px_per_cm"] for f in fits), "yaw_min": min(f["yaw"] for f in fits),
                    "yaw_max": max(f["yaw"] for f in fits), "pitch_abs_max": max(abs(f["pitch"]) for f in fits), "per_face": fits},
       "pose_fallback_vs_face": {k: {"n": len(v), "median_abs_err_deg": dict(zip(["yaw", "pitch", "roll"], np.round(np.median(v, 0), 1).tolist())),
                                     "mean_abs_err_deg": dict(zip(["yaw", "pitch", "roll"], np.round(np.mean(v, 0), 1).tolist()))} for k, v in errs.items()}}
os.makedirs(os.path.join(D, "..", "results"), exist_ok=True)
json.dump(out, open(os.path.join(D, "..", "results", "head_pose.json"), "w"), indent=1)
print(json.dumps({k: v for k, v in out.items() if k != "face_fit"}, indent=1)); print({k: v for k, v in out["face_fit"].items() if k != "per_face"})
