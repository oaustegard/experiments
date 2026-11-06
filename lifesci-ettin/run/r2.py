"""Put files in the public R2 bucket under lifesci-ettin/ (embstore's PUT; reads need no credentials).

    python3 run/r2.py put LOCAL KEY     # -> https://pub-...r2.dev/lifesci-ettin/KEY
"""
import sys
from pathlib import Path

sys.path.insert(0, "/home/user/claude-workspace/scripts")
import embstore  # noqa: E402

PREFIX = "lifesci-ettin"
URL = f"{embstore.PUBLIC}/{PREFIX}"


def put(local, key, ctype="application/octet-stream"):
    embstore._put(f"{PREFIX}/{key}", Path(local), ctype)
    return f"{URL}/{key}"


if __name__ == "__main__":
    print(put(sys.argv[2], sys.argv[3]))
