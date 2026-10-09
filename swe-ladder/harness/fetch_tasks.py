"""Download SWE-bench Verified from Hugging Face and keep the pure-Python repos.

    python harness/fetch_tasks.py            -> data/tasks.jsonl (gitignored)

The parquet is public; Hugging Face answered 200 from CCotw on 2026-10-09.
"""
import json
import sys
import urllib.request

import pandas as pd

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent))
from common import DATA, PURE, TASKS  # noqa: E402

URL = ("https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/main/"
       "data/test-00000-of-00001.parquet")


def main():
    DATA.mkdir(exist_ok=True)
    pq = DATA / "verified.parquet"
    if not pq.exists():
        urllib.request.urlretrieve(URL, pq)
    d = pd.read_parquet(pq)
    assert len(d) == 500, len(d)
    from swebench.harness.constants import MAP_REPO_VERSION_TO_SPECS as SPECS  # swebench==4.0.4
    keep = d[d.repo.isin(PURE)]
    with open(TASKS, "w") as f:
        for r in keep.to_dict("records"):
            r["spec_python"] = SPECS[r["repo"]][r["version"]]["python"]
            for k in ("FAIL_TO_PASS", "PASS_TO_PASS"):
                r[k] = json.loads(r[k]) if isinstance(r[k], str) else list(r[k])
            f.write(json.dumps(r) + "\n")
    print(f"{len(keep)} tasks from {sorted(keep.repo.unique())} -> {TASKS}")


if __name__ == "__main__":
    main()
