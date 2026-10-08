"""Score the fact-listing-step runs. Term lists fixed before any run (2026-10-08).

Invented technical detail = a term the source does not contain. The cache list is
the archive's HALLUCINATED_DETAILS (haiku-assessment/n20/score.py); the search list
was written for this test before the first run. Unsupported claims and reversals are
labelled by reading every output (labels.json), not by this script.
"""
import json, re, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))
from grade import last_text  # noqa: E402

TERMS = {
    "cache": ["lru", "ttl", "write-through", "write through", "redis", "memcached",
              "distributed hash table", "consistent hashing", "cache-aside", "cache aside",
              "p99", "p95", "p50", "rdb", "snapshot", "in-memory", "in memory",
              "sub-millisecond", "milliseconds", "background queue", "background worker",
              "annotation", "decorator", "batch operation", "opentelemetry",
              "structured json", "session-cookie", "jwt", "highcharts",
              "server-side rendering"],
    "search": ["embedding", "vector", "hnsw", "cosine", "nearest neighbor",
               "nearest-neighbor", "ann ", "transformer", "bert", "gpt", "llm",
               "large language model", "faiss", "pinecone", "elasticsearch", "opensearch",
               "pgvector", "milvus", "weaviate", "bm25", "rerank", "re-rank", "hybrid",
               "dimension", "p99", "p95", "milliseconds", " ms ", "recall@", "precision@",
               "gpu", "fine-tun", "index"],
}


def split_visible(text: str):
    """Return (facts, rewrite) for arm V; (None, text) otherwise."""
    f = re.search(r"<facts>(.*?)</facts>", text, re.S)
    r = re.search(r"<rewrite>(.*?)</rewrite>", text, re.S)
    if f and r:
        return f.group(1).strip(), r.group(1).strip()
    return None, text.strip()


def main() -> None:
    runs = json.loads((HERE / "runs.json").read_text())
    rows = {}
    for label, path in sorted(runs.items()):
        name, arm, _ = label.split("-")
        raw = last_text(path)
        facts, rewrite = split_visible(raw) if arm == "V" else (None, raw.strip())
        low = f" {rewrite.lower()} "
        rows[label] = {
            "rewrite": rewrite, "facts": facts,
            "format_ok": (facts is not None) if arm == "V" else ("<" not in rewrite),
            "words": len(re.findall(r"\b\w+\b", rewrite)),
            "term_hits": [t.strip() for t in TERMS[name] if t in low],
            "fact_term_hits": [t.strip() for t in TERMS[name] if facts and t in f" {facts.lower()} "],
        }
    (HERE / "outputs.json").write_text(json.dumps(rows, indent=1))
    print(f"{len(rows)} runs scored")


if __name__ == "__main__":
    main()
