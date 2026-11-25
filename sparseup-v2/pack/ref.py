"""PyTorch reference for linkup-sparseup-embed-v2, without the model repo's remote code.

The Hub model's `modeling_splade.py` subclasses transformers' built-in `ModernBertForMaskedLM` and adds
a SPLADE head. Executing that file was refused in this session (see REPORT.md), so this module loads the
same safetensors into the built-in class and re-implements the head from reading the file:

    w = log1p(relu(logits - shift) / sigma)          per position, fp32
    keep w >= (12th largest w of that position)      ties keep more than 12
    w *= pooling_mask                                 prefix token "[Q]" / "[D]" excluded, CLS/SEP kept
    v = max over positions, then scatter-amax by vocab_fold_index

    ref = Reference("hf")                                # fp32
    v = ref.encode(["a query"], kind="query")            # torch [N, V] float32
    ref.load_weights(pack_state_dict)                     # swap in dequantized pack weights

Texts are run one at a time (no padding), truncated to max_length tokens (default 512).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import AutoTokenizer, ModernBertConfig, ModernBertForMaskedLM

HEAD_KEYS = ("logit_shift", "logit_sigma", "position_top_k", "vocab_fold", "query_prefix",
             "document_prefix", "add_prefix_space", "query_max_length", "doc_max_length",
             "auto_map", "architectures")


class Reference:
    def __init__(self, hf_dir: str | Path, dtype=torch.float32, threads: int | None = None, device: str = "cpu"):
        hf_dir = Path(hf_dir)
        if threads:
            torch.set_num_threads(threads)
        raw = json.loads((hf_dir / "config.json").read_text())
        self.cfg = {k: raw[k] for k in HEAD_KEYS if k in raw}
        conf = ModernBertConfig(**{k: v for k, v in raw.items() if k not in HEAD_KEYS})
        conf._attn_implementation = "sdpa"
        self.model = ModernBertForMaskedLM(conf).eval()
        sd = load_file(str(hf_dir / "model.safetensors"))
        self.fold = sd.pop("vocab_fold_index").long()
        missing, unexpected = self.model.load_state_dict(sd, strict=False)
        # decoder.weight is tied to the token table and not stored.
        if unexpected or set(missing) - {"decoder.weight"}:
            raise RuntimeError(f"state dict mismatch: missing {missing}, unexpected {unexpected}")
        self.model.tie_weights()
        assert self.model.decoder.weight.data_ptr() == self.model.model.embeddings.tok_embeddings.weight.data_ptr()
        self.device = torch.device(device)
        if self.device.type == "cuda":   # keep fp32 matmuls fp32 (no TF32)
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        self.model.to(self.device, dtype)
        self.fold = self.fold.to(self.device)
        self.dtype = dtype
        self.tok = AutoTokenizer.from_pretrained(str(hf_dir))
        self.V = conf.vocab_size

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {k: v for k, v in self.model.state_dict().items() if k != "decoder.weight"}

    def load_weights(self, sd: dict[str, torch.Tensor]):
        """Replace weights in place (e.g. dequantized pack weights); keys as in state_dict()."""
        own = self.model.state_dict()
        with torch.no_grad():
            for k, v in sd.items():
                own[k].copy_(v.to(own[k].dtype))

    def tokenize(self, text: str, kind: str = "document", max_length: int = 512):
        """-> (ids list, pooling mask list), the remote _tokenize rule for one text."""
        prefix = self.cfg["query_prefix"] if kind == "query" else self.cfg["document_prefix"] if kind == "document" else ""
        if self.cfg.get("add_prefix_space") and not text.startswith(" "):
            text = " " + text
        enc = self.tok(prefix + text, truncation=True, max_length=max_length,
                       return_offsets_mapping=True, return_special_tokens_mask=True)
        pool = [int(not (prefix and o[0] < len(prefix) and not s))
                for o, s in zip(enc["offset_mapping"], enc["special_tokens_mask"])]
        return enc["input_ids"], pool

    @torch.inference_mode()
    def token_weights(self, ids: list[int]) -> torch.Tensor:
        """[L, V] gated per-position weights (before pooling mask)."""
        x = torch.tensor([ids], device=self.device)
        logits = self.model(input_ids=x, attention_mask=torch.ones_like(x)).logits[0].float()
        w = torch.log1p(torch.relu(logits - self.cfg["logit_shift"]) / self.cfg["logit_sigma"])
        k = self.cfg.get("position_top_k")
        if k is not None and k < w.shape[-1]:
            cutoff = w.topk(k, dim=-1).values[..., -1:]
            w = w * (w >= cutoff)
        return w

    @torch.inference_mode()
    def encode_ids(self, ids: list[int], pool: list[int]) -> torch.Tensor:
        w = self.token_weights(ids) * torch.tensor(pool, dtype=torch.float32, device=self.device)[:, None]
        v = w.max(dim=0).values
        if self.cfg.get("vocab_fold"):
            v = torch.zeros_like(v).scatter_reduce(0, self.fold, v, reduce="amax", include_self=False)
        return v.cpu()

    def encode(self, texts: list[str], kind: str = "document", max_length: int = 512) -> torch.Tensor:
        return torch.stack([self.encode_ids(*self.tokenize(t, kind, max_length)) for t in texts])
