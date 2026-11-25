"""Check the rebuilt tokenizer.json against slices of the real Linkup file.

The slices were read from hf://models/Linkup-Platform/linkup-sparseup-embed-v2/
tokenizer.json (3,583,844 bytes) through the Hugging Face MCP on 2026-11-25,
at byte offsets 0, 20500, 24300, 600000, 1150000, 2400000 and 3583300.
"""
import json, sys
from pathlib import Path

HERE = Path(__file__).resolve().parent

ADDED = {0: "|||IP_ADDRESS|||", 1: "<|padding|>", 50254: " " * 24, 50276: "  ",
         50277: "|||EMAIL_ADDRESS|||", 50278: "|||PHONE_NUMBER|||", 50279: "<|endoftext|>",
         50280: "[UNK]", 50281: "[CLS]", 50282: "[SEP]", 50283: "[PAD]",
         50360: "[unused75]", 50367: "[unused82]", 50368: "[Q]", 50369: "[D]"}
VOCAB = {"|||IP_ADDRESS|||": 0, "<|padding|>": 1, "!": 2, "A": 34, "`": 65,
         "ĠForum": 24703, "ĠMatter": 24704, "Ġsemantic": 24705, ")âĢĵ": 24713,
         "Ġatheros": 24719, "ĠGPU": 24720, "Dear": 24723,
         "Ġleases": 47194, "surgical": 47198, "Ġoligonucleotides": 47200,
         "}\\{": 47202, "ĠÃ¤n": 47203, "omyc": 47210, "ĠÐ": None}
VOCAB.pop("ĠÐ")
MERGE_RUNS = [
    [("Ġmy", "sql"), ("s", "ocket"), ("ĠT", "IME"), ("ĊĊĊĊ", "ĊĊĊĊ"), ("um", "ann"),
     ("ĠÑģ", "Ð»"), ("Ġnons", "ense"), ("Ġrect", "angle"), ("in", "tern"), ("Ġ19", "7")],
    [("b", "ilt"), ("Ġs", "ial"), ("Ġtodd", "ler"), ("7", "67"), ("it", "ons"), ("ĠP", "IP"),
     ("ĠT", "us"), ("ibr", "ated"), ("Ġfort", "ified"), ("feren", "ced"), ("ĠOut", "comes")],
]
CONFIG = {"normalizer": {"type": "NFC"},
          "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False,
                            "trim_offsets": True, "use_regex": True}}


def check(path=HERE / "tokenizer.json"):
    d = json.loads(Path(path).read_text())
    added = {a["id"]: a["content"] for a in d["added_tokens"]}
    vocab, merges = d["model"]["vocab"], d["model"]["merges"]
    merges = [tuple(m.split(" ", 1)) if isinstance(m, str) else tuple(m) for m in merges]
    fails = [f"added {i}" for i, c in ADDED.items() if added.get(i) != c]
    fails += [f"vocab {t}" for t, i in VOCAB.items() if vocab.get(t) != i]
    for run in MERGE_RUNS:
        at = merges.index(run[0]) if run[0] in merges else -1
        if at < 0 or merges[at:at + len(run)] != run:
            fails.append(f"merge run {run[0]}")
    fails += [k for k, v in CONFIG.items() if d[k] != v]
    if merges[-1] != ("ĠOut", "comes"):
        fails.append("last merge")
    n = len(ADDED) + len(VOCAB) + sum(map(len, MERGE_RUNS)) + len(CONFIG) + 1
    return n, fails


if __name__ == "__main__":
    n, fails = check(*sys.argv[1:])
    print(f"{n - len(fails)}/{n} checks pass", fails or "")
    sys.exit(1 if fails else 0)
