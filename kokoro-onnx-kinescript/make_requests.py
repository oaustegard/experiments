"""The ten narration lines of colbert-pooling-explainer as kine voice would send them to a backend.

kine voice drops [[mark]] ids and tags and collapses whitespace (kinescript/voice.py split_marks, untag); this
repeats that with the same regexes so the requests match a real run.
"""
import json
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from _lib.paths import experiment  # noqa: E402

MARK = re.compile(r"\[\[\s*(\w+)\s*\]\]")
TAG = re.compile(r"(?<!\[)\[(?!\[)[^\[\]()]+\](?![\](])")


def spoken(text: str) -> str:
    parts = [p for k, p in enumerate(MARK.split(text)) if k % 2 == 0]
    return re.sub(r"\s+", " ", TAG.sub("", " ".join(parts))).strip()


scene = json.loads((experiment("colbert-pooling-explainer") / "scene.json").read_text())
reqs = [{"id": a["id"], "text": spoken(a["text"]), "speaker": a["speaker"], "rate": a["rate"]}
        for a in scene["audio"] if a.get("text")]
(HERE / "requests.json").write_text(json.dumps(reqs, indent=1, ensure_ascii=False))
print(len(reqs), "requests;", sum(len(r["text"].split()) for r in reqs), "words")
