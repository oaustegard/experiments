# Which MediaPipe signals tell a person facing the camera from one facing away? Downloads a small hand-labelled
# set (mmpose test images + Wikimedia Commons photos), runs Pose Landmarker lite (as the page does) and Face
# Landmarker, and scores three signals. Needs: pip install mediapipe numpy pillow; network.
# Images go to ./facing-cache (gitignored).
import json, os, time, urllib.error, urllib.parse, urllib.request, numpy as np, mediapipe as mp
from mediapipe.tasks import python as mpt
from mediapipe.tasks.python import vision as V
from PIL import Image, ImageOps
D = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(D, "facing-cache"); os.makedirs(C, exist_ok=True)
MM = "https://raw.githubusercontent.com/open-mmlab/mmpose/main/tests/data/"
CO = "https://commons.wikimedia.org/wiki/Special:FilePath/"  # the API 429s from cloud IPs; FilePath does not
MPA = "https://storage.googleapis.com/mediapipe-assets/"
# (label, url). Front = facing the camera, including three-quarter views; side views were left out.
SET = [("front", MM + p) for p in [
    "aic/054d9ce9201beffc76e5ff2169d2af2f027002ca.jpg", "aic/fa436c914fe4a8ec1ec5474af4d3820b84d17561.jpg",
    "coco/000000000785.jpg", "coco/000000040083.jpg", "coco/000000197388.jpg", "crowdpose/103319.jpg",
    "h36m/S7/S7_Greeting.55011271/S7_Greeting.55011271_000396.jpg", "h36m/S8/S8_WalkDog_1.55011271/S8_WalkDog_1.55011271_000026.jpg",
    "mpi_inf_3dhp/S4_Seq2_Cam0_001033.jpg", "mpi_inf_3dhp/TS1_002001.jpg",
    "mpii/004645041.jpg", "mpii/005808361.jpg", "mpii/052475643.jpg", "mpii/060754485.jpg", "ochuman/000817.jpg",
    "panoptic/005880453_01_l.jpg", "panoptic/ex2_2.flv_000040_l.jpg"]] + \
    [("front", MPA + f) for f in ["portrait.jpg", "business-person.png", "pose.jpg"]] + \
    [("back", MM + "h36m/S1/S1_Directions_1.54138969/S1_Directions_1.54138969_000001.jpg")] + \
    [("back", CO + urllib.parse.quote(t) + "?width=800") for t in [
    "Walking woman Spijkenisse behind.jpg", "Lin Daiying and Vivian Chen standing seen from behind 20220729.jpg",
    "Walking woman with short black hair from behind, Galaxy Mall 3, East Surabaya.jpg",
    "Zoe and her fan standing at Aorus booth seen from behind 20230114.jpg",
    "Elderly woman wearing a black coat walking through Praça de Londres, Lisbon, Portugal julesvernex2.jpg",
    "Nude standing man, seen from behind, looks at lake.jpg", "Young woman walking on the street.jpg",
    "Woman walking down path (Unsplash).jpg"]]
def get(url, name):
    p = os.path.join(C, name)
    for attempt in range(5):
        if os.path.exists(p): break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "xray-glasses-test/1.0 (https://austegard.com)"})
            data = urllib.request.urlopen(req).read(); open(p, "wb").write(data)
        except urllib.error.HTTPError as e:  # Commons answers bursts with 429
            if e.code != 429 or attempt == 4: raise
            time.sleep(10 * (attempt + 1))
    return p
pose = V.PoseLandmarker.create_from_options(V.PoseLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=get(
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task", "pose_lite.task")), num_poses=1))
face = V.FaceLandmarker.create_from_options(V.FaceLandmarkerOptions(base_options=mpt.BaseOptions(model_asset_path=get(
    "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task", "face.task")), num_faces=1))
rows = []
for i, (lab, url) in enumerate(SET):
    im = ImageOps.exif_transpose(Image.open(get(url, f"{i:02d}" + os.path.splitext(urllib.parse.urlparse(url).path)[1]))).convert("RGB")
    m = mp.Image(image_format=mp.ImageFormat.SRGB, data=np.asarray(im))
    r, f = pose.detect(m), face.detect(m)
    if not r.pose_landmarks: print(f"{i:2} {lab:5} no pose"); continue
    L, W = r.pose_landmarks[0], r.pose_world_landmarks[0]
    shx = L[11].x - L[12].x                              # subject's left shoulder right of their right one in the image
    wz = W[0].z - (W[11].z + W[12].z) / 2                # nose depth minus shoulder depth, metres; < 0 = nose nearer
    rows.append((lab, shx, wz, bool(f.face_landmarks)))
    print(f"{i:2} {lab:5} shoulder dx {shx:7.3f}  nose depth {wz:7.3f} m  face {bool(f.face_landmarks)}")
res = {"labelled": {"front": sum(l == "front" for l, _ in SET), "back": sum(l == "back" for l, _ in SET)}, "pose_found": len(rows), "signals": {}}
fz = [r[2] for r in rows if r[0] == "front"]; bz = [r[2] for r in rows if r[0] == "back"]
res["nose_depth_m"] = {"front_nose_lead_min_cm": round(-max(fz) * 100, 1), "front_max": round(max(fz), 3), "front_min": round(min(fz), 3), "back_detected": sorted(round(z, 3) for z in bz)}
for name, front in (("shoulder order (dx > 0)", lambda r: r[1] > 0), ("nose depth (< 0)", lambda r: r[2] < 0), ("face found", lambda r: r[3])):
    nf = sum(r[0] == "front" for r in rows); nb = len(rows) - nf
    nb_all = sum(l == "back" for l, _ in SET)  # back views pose missed count as misses here
    res["signals"][name.split(" (")[0].replace(" ", "_")] = {"front_called_front": f"{sum(front(r) for r in rows if r[0] == 'front')}/{nf}", "back_called_back": f"{sum(not front(r) for r in rows if r[0] == 'back')}/{nb}",
                                                            "back_called_back_of_all_labelled": f"{sum(not front(r) for r in rows if r[0] == 'back')}/{nb_all}"}
    print(f"{name:24} front called front {sum(front(r) for r in rows if r[0] == 'front')}/{nf}, back called back {sum(not front(r) for r in rows if r[0] == 'back')}/{nb}")
os.makedirs(os.path.join(D, "..", "results"), exist_ok=True)
json.dump(res, open(os.path.join(D, "..", "results", "facing.json"), "w"), indent=1)
print("wrote results/facing.json")
