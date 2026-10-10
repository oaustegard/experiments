"""Extract babel's CLDR .dat files from the PyPI wheel into ENVS/_babel-locale-data.

The wheel is a zip read as data; nothing from it is imported or executed.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from workspace import BABEL_DATA  # noqa: E402


def main(version: str = "2.16.0"):
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, "-m", "pip", "download", "--no-deps", "--only-binary=:all:",
                        "-d", d, f"babel=={version}"], check=True)
        whl = next(Path(d).glob("*.whl"))
        BABEL_DATA.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(whl) as z:
            for n in z.namelist():
                if n.startswith("babel/locale-data/") or n == "babel/global.dat":
                    target = BABEL_DATA / n[len("babel/"):]
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(z.read(n))
    subprocess.run(["chmod", "-R", "a+rX", str(BABEL_DATA)], check=True)
    print(len(list((BABEL_DATA / "locale-data").glob("*.dat"))), "locale files")


if __name__ == "__main__":
    main(*sys.argv[1:])
