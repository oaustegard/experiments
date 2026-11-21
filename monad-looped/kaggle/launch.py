"""Fill a Kaggle job template with the current pushed commit and push it.

    python3 kaggle/launch.py kaggle/screen_job.py SLUG [--short]

Requires the working tree's HEAD to be on origin (the job fetches that commit from GitHub).
"""
import argparse
import subprocess
import sys
from pathlib import Path

HUB = Path("/home/user/claude-workspace")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("template")
    ap.add_argument("slug")
    ap.add_argument("--short", action="store_true")
    a = ap.parse_args()
    sha = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    on_remote = subprocess.run(["git", "branch", "-r", "--contains", sha], capture_output=True, text=True).stdout.strip()
    if not on_remote:
        sys.exit(f"{sha} is not on origin; push first")
    src = Path(a.template).read_text().replace("__COMMIT__", sha).replace("__SHORT__", "True" if a.short else "False")
    out = Path("/tmp") / f"{a.slug}.py"
    out.write_text(src)
    subprocess.run([sys.executable, str(HUB / "scripts/kaggle_job.py"), "push", a.slug, str(out), "--accel", "gpu", "--internet"],
                   check=True)
    print("pushed", a.slug, "at", sha)


if __name__ == "__main__":
    main()
