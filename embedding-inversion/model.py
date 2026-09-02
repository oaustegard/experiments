"""vec2text-shaped inverter on t5-small.

Zero-step:  encoder input = k soft tokens projected from the target vector e.
Corrector:  encoder input = [proj_t(e); proj_h(phi(x_hat)); proj_d(e - phi(x_hat));
            embed(tokens of x_hat)], i.e. the target, the hypothesis's own
            embedding, their difference, and the hypothesis text.

Both share the T5 body; the projections are separate 2-layer MLPs, one per
slot, each emitting k * d_model values reshaped to k pseudo-tokens.
"""
from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration

T5_ID = "google-t5/t5-small"


class Proj(nn.Module):
    def __init__(self, d_in: int, d_model: int, k: int) -> None:
        super().__init__()
        self.k, self.d = k, d_model
        self.net = nn.Sequential(nn.Linear(d_in, 1024), nn.GELU(), nn.Linear(1024, k * d_model))

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        return self.net(e).view(e.shape[0], self.k, self.d)


class Inverter(nn.Module):
    def __init__(self, mode: str, d_in: int = 384, k: int = 8) -> None:
        super().__init__()
        assert mode in ("zero", "correct")
        self.mode, self.k = mode, k
        self.t5 = T5ForConditionalGeneration.from_pretrained(T5_ID)
        d = self.t5.config.d_model
        self.proj_t = Proj(d_in, d, k)
        if mode == "correct":
            self.proj_h = Proj(d_in, d, k)
            self.proj_d = Proj(d_in, d, k)

    def encoder_inputs(self, e, e_hyp=None, hyp_ids=None, hyp_mask=None):
        parts = [self.proj_t(e)]
        if self.mode == "correct":
            parts += [self.proj_h(e_hyp), self.proj_d(e - e_hyp), self.t5.shared(hyp_ids)]
        x = torch.cat(parts, 1)
        B = e.shape[0]
        soft = torch.ones(B, self.k * (3 if self.mode == "correct" else 1), dtype=torch.long, device=e.device)
        mask = soft if self.mode == "zero" else torch.cat([soft, hyp_mask], 1)
        return x, mask

    def forward(self, e, labels, e_hyp=None, hyp_ids=None, hyp_mask=None):
        x, mask = self.encoder_inputs(e, e_hyp, hyp_ids, hyp_mask)
        return self.t5(inputs_embeds=x, attention_mask=mask, labels=labels)

    @torch.no_grad()
    def generate(self, e, e_hyp=None, hyp_ids=None, hyp_mask=None, *, num_beams=4,
                 max_new_tokens=48, num_return_sequences=1):
        x, mask = self.encoder_inputs(e, e_hyp, hyp_ids, hyp_mask)
        return self.t5.generate(inputs_embeds=x, attention_mask=mask, num_beams=num_beams,
                                max_new_tokens=max_new_tokens,
                                num_return_sequences=num_return_sequences, do_sample=False)

    def param_groups(self, lr_t5: float, lr_proj: float):
        proj = [p for n, p in self.named_parameters() if n.startswith("proj_")]
        body = [p for n, p in self.named_parameters() if not n.startswith("proj_")]
        return [{"params": body, "lr": lr_t5}, {"params": proj, "lr": lr_proj}]


def tokenizer():
    return AutoTokenizer.from_pretrained(T5_ID)


def encode_labels(tok, texts, max_len=48):
    b = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_len)
    labels = b.input_ids.clone()
    labels[b.attention_mask == 0] = -100
    return labels, b.input_ids, b.attention_mask
