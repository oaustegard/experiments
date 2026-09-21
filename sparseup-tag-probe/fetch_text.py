"""Re-fetch memory text by pinned id into data/texts.json (gitignored) and verify hashes."""
import json
from common import DATA, load_fixture, memory_module, text_hash


def main():
    fx = load_fixture()
    memory = memory_module()
    ids = [e["id"] for e in fx["memories"]]
    want = {e["id"]: e["sha256"] for e in fx["memories"]}
    texts, drift, missing = {}, [], []
    for i in range(0, len(ids), 200):
        chunk = ids[i:i + 200]
        q = ",".join("?" * len(chunk))
        for r in memory._exec(f"SELECT id, summary FROM memories WHERE id IN ({q})", chunk):
            texts[r["id"]] = r["summary"] or ""
    for mid in ids:
        if mid not in texts:
            missing.append(mid)
        elif text_hash(texts[mid]) != want[mid]:
            drift.append(mid)
    DATA.mkdir(exist_ok=True)
    (DATA / "texts.json").write_text(json.dumps({"ids": ids, "texts": [texts.get(m, "") for m in ids],
                                                 "missing": missing, "drift": drift}))
    print(f"fetched {len(texts)}/{len(ids)}; missing {len(missing)}; hash drift {len(drift)}")
    if missing or drift:
        print("WARNING: fixture and live store disagree for", (missing + drift)[:10])


if __name__ == "__main__":
    main()
