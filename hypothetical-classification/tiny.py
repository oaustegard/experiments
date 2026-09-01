"""Can a 57M / 321M Pleias model be the hallucinating half?

Monad has no chat template — it is a completion model, so it gets few-shot
`query -> label` pairs. Baguettotron has one, and it forces a <think> block, so its
output is parsed after </think>. Few-shot completion is tested on both, because the
task is "continue this pattern in this register", which is what few-shot IS.
"""
from __future__ import annotations
import json, re, sys, time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

SHOTS = [
    ("wood coffee table", "Coffee Tables"),
    ("navy throw pillow", "Throw Pillows"),
    ("5 drawer dresser", "Dressers & Chests"),
    ("counter height stool", "Bar Stools"),
    ("kids bunk bed", "Kids Beds"),
    ("round wall mirror", "Wall & Accent Mirrors"),
    ("outdoor patio umbrella", "Patio Umbrellas"),
    ("stainless trash can", "Trash & Recycling Bins"),
]
PREAMBLE = ("A product taxonomy maps each search query to its category label.\n"
            "Labels are short plural noun phrases in plain retail wording.\n\n")


def fewshot_prompt(query: str) -> str:
    body = "".join(f"Query: {q}\nCategory: {l}\n\n" for q, l in SHOTS)
    return PREAMBLE + body + f"Query: {query}\nCategory:"


class Tiny:
    def __init__(self, mid: str):
        self.mid = mid
        self.tok = AutoTokenizer.from_pretrained(mid)
        self.model = AutoModelForCausalLM.from_pretrained(mid, dtype=torch.float32).eval()
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token or "[PAD]"

    @torch.no_grad()
    def _gen(self, prompt: str, max_new: int, stop_newline: bool) -> str:
        enc = self.tok(prompt, return_tensors="pt")
        out = self.model.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                  pad_token_id=self.tok.pad_token_id)
        text = self.tok.decode(out[0][enc["input_ids"].shape[1]:], skip_special_tokens=True)
        if stop_newline:
            text = text.split("\n")[0]
        return text.strip()

    def fewshot(self, query: str) -> str:
        return self._gen(fewshot_prompt(query), 16, True).strip(' "')

    def chat(self, query: str, max_new: int = 220) -> str:
        if not self.tok.chat_template:
            return ""
        msgs = [{"role": "user", "content":
                 "A product taxonomy maps a search query to a category label: a short "
                 "plural noun phrase in plain retail wording, like Coffee Tables or "
                 "Dressers & Chests. Give the category label for this query, and nothing "
                 f"else.\n\nQuery: {query}"}]
        p = self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        raw = self._gen(p, max_new, False)
        tail = raw.split("</think>")[-1] if "</think>" in raw else raw
        tail = re.sub(r"<\|im_end\|>.*", "", tail, flags=re.S)
        return tail.strip().split("\n")[0].strip(' "*')


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    D = json.load(open("h40.json"))
    queries = D["q"][:n]
    out = {}
    for mid, modes in (("PleIAs/Monad", ["fewshot"]),
                       ("PleIAs/Baguettotron", ["fewshot", "chat"])):
        m = Tiny(mid)
        for mode in modes:
            t = time.time()
            labels = [getattr(m, mode)(q) for q in queries]
            dt = time.time() - t
            key = f"{mid.split('/')[1]}-{mode}"
            out[key] = labels
            print(f"{key:26} {dt/len(queries)*1000:7.0f} ms/query   {labels[:5]}", flush=True)
        del m
    json.dump(out, open("tiny_arms.json", "w"), indent=1)
