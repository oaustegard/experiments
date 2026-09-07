"""Dump a python-tokenizer reference for verify_tokenizer.mjs, then run node."""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import data as D
from transformers import AutoTokenizer

HERE = os.path.dirname(os.path.abspath(__file__))
tok = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-135M")

texts = []
for op, t in D.TEMPLATES:
    texts.append(t.format(a=1234, b=56))
texts += [
    "12 + 3 =", "What is 4567 plus 89?", "Compare 100 and 100:",
    "Subtract 987654 from 1000000.", "Is 2064 greater or less than 1653?",
    "0 * 0 =", "-45 - 7 =", "  spaced   out  ",
    "What is 999999 multiplied by 999999?", "1234567890 + 9876543210 =",
    "Hello, world! 42?", "a+b=c 7,8;9 (10) [11] {12}",
    "tab\there\nnewline", "Émile has 3 apples & 4 pears.",
    "3.14 and 2.71", "no numbers here at all", "1", "1 2 3 4 5",
]
rows = []
for t in texts[:30]:
    ids = tok(t)["input_ids"]
    rows.append({"text": t, "ids": ids,
                 "decoded": tok.decode(ids, skip_special_tokens=True)})
with open(os.path.join(HERE, "tok_ref.json"), "w") as f:
    json.dump(rows, f)
print(f"{len(rows)} reference prompts")
sys.exit(subprocess.call(["node", "verify_tokenizer.mjs"], cwd=HERE))
