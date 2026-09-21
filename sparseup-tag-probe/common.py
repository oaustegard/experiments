"""Shared helpers: fixture path, tag cleaning, private-scope filter, text hashing."""
import hashlib, json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
FIXTURE = HERE / "fixture.json"

# ops private-tag-discipline: 'confidential' plus every registered private scope tag.
PRIVATE_TAGS = {"confidential", "career-search", "improve-oskar"}
MIN_LABEL_COUNT = 10

_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_HEX8 = re.compile(r"^[0-9a-f]{8}$")
_NUM = re.compile(r"^#?\d+$|^(issue|issues|pr|prs)-?\d+$")


def raw_tags(row) -> list:
    t = row.get("tags") or []
    if isinstance(t, str):
        try:
            t = json.loads(t)
        except Exception:
            t = [x for x in t.split(",")]
    return [x.strip().lower() for x in t if isinstance(x, str) and x.strip()]


def drop_reason(tag: str):
    if _DATE.match(tag):
        return "date"
    if _HEX8.match(tag):
        return "hex8"
    if _NUM.match(tag):
        return "number"
    return None


def clean_tags(tags: list) -> list:
    seen, out = set(), []
    for t in tags:
        if drop_reason(t) is None and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def text_hash(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8")).hexdigest()


def load_fixture() -> dict:
    return json.loads(FIXTURE.read_text())


def memory_module():
    sys.path.insert(0, "/mnt/skills/user/remembering")
    from scripts import memory  # noqa: E402
    return memory
