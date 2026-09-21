"""Pin the fixture ONCE: memory ids + cleaned tags + per-memory content hash + label set.

Refuses to overwrite an existing fixture.json — a seeded sample over the live store is
not a fixture (memory f893648c). Memory text is never written here; fetch_text.py
re-fetches it by id into the gitignored data/ directory and checks the hashes.
"""
import collections, json, sys
from datetime import datetime, UTC
from common import (FIXTURE, MIN_LABEL_COUNT, PRIVATE_TAGS, clean_tags, drop_reason,
                    memory_module, raw_tags, text_hash)


def main():
    if FIXTURE.exists():
        sys.exit(f"{FIXTURE} exists; the fixture is pinned. Delete it deliberately to rebuild.")
    memory = memory_module()
    rows = memory._exec(
        "SELECT id, summary, type, tags FROM memories "
        "WHERE deleted_at IS NULL AND is_superseded=0 ORDER BY id", [])
    n_all = len(rows)
    rows = [r for r in rows if not (set(raw_tags(r)) & PRIVATE_TAGS)]
    n_private = n_all - len(rows)

    dropped = collections.Counter()
    counts = collections.Counter()
    entries = []
    for r in rows:
        rt = raw_tags(r)
        for t in rt:
            why = drop_reason(t)
            if why:
                dropped[why] += 1
        ct = clean_tags(rt)
        counts.update(ct)
        entries.append({"id": r["id"], "type": r["type"], "tags": ct,
                        "sha256": text_hash(r["summary"]), "chars": len(r["summary"] or "")})

    labels = sorted(t for t, c in counts.items() if c >= MIN_LABEL_COUNT)
    label_set = set(labels)
    for e in entries:
        e["labels"] = [t for t in e["tags"] if t in label_set]

    corpus_hash = text_hash("\n".join(f"{e['id']} {e['sha256']}" for e in entries))
    fx = {
        "pinned_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "source": "memories WHERE deleted_at IS NULL AND is_superseded=0",
        "n_rows_live": n_all,
        "n_excluded_private": n_private,
        "private_tags": sorted(PRIVATE_TAGS),
        "n_memories": len(entries),
        "n_memories_with_label": sum(1 for e in entries if e["labels"]),
        "tags_dropped": dict(dropped),
        "n_distinct_clean_tags": len(counts),
        "min_label_count": MIN_LABEL_COUNT,
        "n_labels": len(labels),
        "labels": labels,
        "label_counts": {t: counts[t] for t in labels},
        "corpus_sha256": corpus_hash,
        "memories": entries,
    }
    FIXTURE.write_text(json.dumps(fx, indent=1, sort_keys=False) + "\n")
    print(f"pinned {len(entries)} memories ({n_private} private excluded of {n_all}); "
          f"{len(labels)} labels >= {MIN_LABEL_COUNT}; dropped {dict(dropped)}; "
          f"corpus {corpus_hash[:12]}")


if __name__ == "__main__":
    main()
