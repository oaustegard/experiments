"""Rebuild linkup-sparseup-embed-v2's tokenizer.json from its public base.

The sandbox this ran in cannot reach huggingface.co, so the tokenizer is
reconstructed: ModernBERT's tokenizer is OLMo's GPT-NeoX-20B PII tokenizer
(allenai/OLMo, olmo_data/tokenizers/allenai_eleuther-ai-gpt-neox-20b-pii-special.json)
plus [UNK] [CLS] [SEP] [PAD] [MASK] and [unused0..82] at 50280..50367, and
Linkup adds [Q] [D] at 50368/50369. check_tokenizer.py verifies the result
against slices of the real file read through the Hugging Face MCP.
"""
import json, sys, urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = ("https://raw.githubusercontent.com/allenai/OLMo/main/olmo_data/tokenizers/"
       "allenai_eleuther-ai-gpt-neox-20b-pii-special.json")


def tok(i, c, special, normalized):
    return {"id": i, "content": c, "single_word": False, "lstrip": False,
            "rstrip": False, "normalized": normalized, "special": special}


def build(out=HERE / "tokenizer.json"):
    d = json.loads(urllib.request.urlopen(SRC).read())
    added = {a["id"]: a for a in d["added_tokens"]}
    # The OLMo file has <|endoftext|> at 0 and |||IP_ADDRESS||| at 50279;
    # ModernBERT/Linkup swap them (id 0 = |||IP_ADDRESS|||, 50279 = <|endoftext|>).
    vocab = d["model"]["vocab"]
    vocab.pop("<|endoftext|>", None)
    vocab["|||IP_ADDRESS|||"] = 0
    added[0] = tok(0, "|||IP_ADDRESS|||", False, True)
    added[50279] = tok(50279, "<|endoftext|>", True, False)
    for i, c in enumerate(["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"], 50280):
        added[i] = tok(i, c, True, False)
    for k in range(83):
        added[50285 + k] = tok(50285 + k, f"[unused{k}]", False, True)
    added[50368] = tok(50368, "[Q]", True, False)
    added[50369] = tok(50369, "[D]", True, False)
    d["added_tokens"] = [added[i] for i in sorted(added)]
    d["post_processor"] = {
        "type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                   {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
        "pair": [{"SpecialToken": {"id": "[CLS]", "type_id": 0}},
                 {"Sequence": {"id": "A", "type_id": 0}},
                 {"SpecialToken": {"id": "[SEP]", "type_id": 0}},
                 {"Sequence": {"id": "B", "type_id": 0}},
                 {"SpecialToken": {"id": "[SEP]", "type_id": 0}}],
        "special_tokens": {t: {"id": t, "ids": [i], "tokens": [t]} for t, i in
                           [("[CLS]", 50281), ("[MASK]", 50284), ("[PAD]", 50283),
                            ("[SEP]", 50282), ("[UNK]", 50280)]},
    }
    d["padding"] = {"strategy": "BatchLongest", "direction": "Right", "pad_to_multiple_of": None,
                    "pad_id": 50283, "pad_type_id": 0, "pad_token": "[PAD]"}
    d["truncation"] = None
    out.write_text(json.dumps(d, ensure_ascii=False))
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "tokenizer.json"))
