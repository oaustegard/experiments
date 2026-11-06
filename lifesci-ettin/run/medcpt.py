"""MedCPT teacher: query encoder for queries, article encoder for [title, text] passages; CLS, 768-d."""
import time

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

torch.set_num_threads(4)


class MedCPT:
    def __init__(self):
        self.qt = AutoTokenizer.from_pretrained("ncbi/MedCPT-Query-Encoder")
        self.q = AutoModel.from_pretrained("ncbi/MedCPT-Query-Encoder").eval()
        self.at = AutoTokenizer.from_pretrained("ncbi/MedCPT-Article-Encoder")
        self.a = AutoModel.from_pretrained("ncbi/MedCPT-Article-Encoder").eval()

    @torch.inference_mode()
    def queries(self, texts, bs=64, max_len=64):
        out = []
        for i in range(0, len(texts), bs):
            f = self.qt(texts[i:i + bs], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            out.append(self.q(**f).last_hidden_state[:, 0].float().numpy())
        return np.concatenate(out)

    @torch.inference_mode()
    def articles(self, titles, texts, bs=32, max_len=512, log=None):
        order = np.argsort([len(t) for t in texts])
        out = np.zeros((len(texts), 768), np.float32)
        t0 = time.time()
        for n, i in enumerate(range(0, len(order), bs)):
            ix = order[i:i + bs]
            f = self.at([[titles[j], texts[j]] for j in ix], truncation=True, padding=True, max_length=max_len, return_tensors="pt")
            out[ix] = self.a(**f).last_hidden_state[:, 0].float().numpy()
            if log and n % 50 == 0:
                log(f"{i}/{len(texts)} {time.time() - t0:.0f}s")
        return out
