"""Kaggle batch job (Tesla T4, internet on): fine-tune eng tessdata_best on generated lines plus CORD receipts.

    python3 /home/user/claude-workspace/scripts/kaggle_job.py push tess-ft-2 run/ft/kaggle_job.py --accel gpu --internet

  arm C  generated document lines (run/ft/make_train.py) plus CORD v2 receipt lines (run/ft/cord_train.py, CC BY 4.0)
  (tess-ft-1 ran arm A, generated only, and arm B, plus SROIE receipts 30+, whose licence is unconfirmed)
Both: run/ft/train.py, batch 32, lr 1e-4 with warm-up and cosine decay, validation on held-out generated lines
and held-out SROIE receipts (never on the experiment's evaluation sets, which stay on the coordinator's machine).
Outputs in /kaggle/working/<arm>/: eng.traineddata, log.jsonl, train.log. The code comes from
oaustegard/experiments main; the model from tesseract-ocr/tessdata_best; text from WikiText-103 and SROIE via
the Hugging Face Hub; extra fonts (SIL Open Font License) from google/fonts.
"""
import os
import shutil
import subprocess
import sys
import urllib.request

W = "/kaggle/working"
SRC = "/tmp/exp"
R = f"{SRC}/tesseract-gpu"
STEPS = int(os.environ.get("FT_STEPS", "5000"))
FONTS = [
    "ofl/courierprime/CourierPrime-Regular.ttf", "ofl/courierprime/CourierPrime-Bold.ttf",
    "ofl/specialelite/SpecialElite-Regular.ttf", "ofl/vt323/VT323-Regular.ttf",
    "ofl/sharetechmono/ShareTechMono-Regular.ttf", "ofl/dotgothic16/DotGothic16-Regular.ttf",
    "ofl/spacemono/SpaceMono-Regular.ttf", "ofl/spacemono/SpaceMono-Bold.ttf",
    "ofl/ptserif/PT_Serif-Web-Regular.ttf", "ofl/ptsans/PT_Sans-Web-Regular.ttf", "ofl/ptmono/PTM55FT.ttf",
    "ofl/ibmplexmono/IBMPlexMono-Regular.ttf", "ofl/ibmplexsans/IBMPlexSans-Regular.ttf",
    "ofl/ibmplexserif/IBMPlexSerif-Regular.ttf", "ofl/lato/Lato-Regular.ttf", "ofl/lato/Lato-Bold.ttf",
    "ofl/oldstandardtt/OldStandard-Regular.ttf", "ofl/cutivemono/CutiveMono-Regular.ttf",
    "ofl/anonymouspro/AnonymousPro-Regular.ttf", "ofl/overpassmono/OverpassMono[wght].ttf",
    "ofl/librebaskerville/LibreBaskerville-Regular.ttf", "ofl/crimsontext/CrimsonText-Regular.ttf",
    "ofl/robotomono/RobotoMono[wght].ttf", "ofl/roboto/Roboto[wdth,wght].ttf", "ofl/opensans/OpenSans[wdth,wght].ttf",
    "ofl/archivonarrow/ArchivoNarrow[wght].ttf", "ofl/oswald/Oswald[wght].ttf", "ofl/inconsolata/Inconsolata[wdth,wght].ttf",
]


def sh(cmd, log=None):
    print("$", cmd, flush=True)
    if log:
        cmd = f"( {cmd} ) > {log} 2>&1"
    subprocess.run(cmd, shell=True, check=True)


def main():
    sh("nvidia-smi --query-gpu=name,memory.total --format=csv")
    sh("pip install -q rapidfuzz fonttools")
    sh(f"git clone -q --depth 1 --filter=blob:none --sparse https://github.com/oaustegard/experiments {SRC} "
       f"&& cd {SRC} && git sparse-checkout set tesseract-gpu _lib && git log -1 --format='code %H'")
    os.makedirs(f"{R}/models/best", exist_ok=True)
    urllib.request.urlretrieve("https://github.com/tesseract-ocr/tessdata_best/raw/main/eng.traineddata",
                               f"{R}/models/best/eng.traineddata")
    fd = "/tmp/fonts"
    os.makedirs(fd, exist_ok=True)
    got = 0
    for f in FONTS:
        try:
            urllib.request.urlretrieve(f"https://github.com/google/fonts/raw/main/{f}", f"{fd}/{os.path.basename(f)}")
            got += 1
        except Exception as e:
            print("font skipped", f, e, flush=True)
    print(f"{got}/{len(FONTS)} fonts", flush=True)
    env = f"cd {R} && FT_FONT_DIR={fd}"
    sh(f"{env} python3 run/ft/make_train.py --n 30000 --jobs 4", log=f"{W}/make_train.log")
    sh(f"{env} python3 run/ft/cord_train.py", log=f"{W}/cord_train.log")
    sh(f"{env} python3 run/ft/train.py prep --extra data/ft/cord/labels.jsonl", log=f"{W}/prep.log")
    common = f"--steps {STEPS} --batch 32 --eval-every 500 --threads 2 --max-minutes 300"
    pc = subprocess.Popen(f"{env} python3 run/ft/train.py train --run armC --device cuda:0 --real-src cord {common} > {W}/armC_train.log 2>&1", shell=True)
    rc_ = pc.wait()
    print("train exit code", rc_, flush=True)
    ra = rb = rc_
    for arm, rc in (("armC", rc_),):
        os.makedirs(f"{W}/{arm}", exist_ok=True)
        if os.path.exists(f"{R}/data/ft/{arm}/last.pt"):
            sh(f"{env} python3 run/ft/train.py export --run {arm}")
            shutil.copy(f"{R}/data/ft/{arm}/eng.traineddata", f"{W}/{arm}/eng.traineddata")
        if os.path.exists(f"{R}/data/ft/{arm}/log.jsonl"):
            shutil.copy(f"{R}/data/ft/{arm}/log.jsonl", f"{W}/{arm}/log.jsonl")
    if ra or rb:
        sys.exit(1)


if __name__ == "__main__":
    main()
