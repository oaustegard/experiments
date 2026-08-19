"""Device-aware EAGLE head training for Baguettotron. Single file, no repo deps.

Written for Apple Silicon but runs on CPU or CUDA unchanged. Two differences
from the CPU pipeline it replaces:

1. Fused harvest and training. The CPU version cached hidden states to disk
   because a forward pass cost more than re-reading 290 MB. Where forward passes
   are cheap, caching is the wrong trade: 50M tokens of fp16 hidden states is
   58 GB, and that ceiling is what limited the CPU run to 2.7M tokens. Streaming
   recomputes them each epoch and needs no disk at all.
2. --cache lets you keep the old behaviour when you want many epochs over a
   fixed corpus.

The MPS path is UNTESTED — written on a Linux CPU container with no Apple
hardware. Run `--verify` first; it checks the three things that would silently
corrupt training if the device behaved differently.

  python3 eagle_mac.py --verify
  python3 eagle_mac.py --tokens 20000000 --epochs 2
"""
import argparse, json, os, time
import numpy as np
import torch
import torch.nn as nn

MODEL = "PleIAs/Baguettotron"


def pick_device(requested=None):
    if requested and requested != "auto":
        return torch.device(requested)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


class EagleHead(nn.Module):
    """EAGLE autoregression head: FC(2h->h) then one decoder layer.

    Embedding and LM head come from the target and stay frozen, so only this
    trains — 4.2M parameters against the target's 321M.
    """

    def __init__(self, cfg):
        super().__init__()
        from transformers.models.llama.modeling_llama import (
            LlamaDecoderLayer, LlamaRotaryEmbedding)
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
        self.rotary = LlamaRotaryEmbedding(cfg)

    def forward(self, feat, emb):
        x = self.fc(torch.cat([feat, emb], dim=-1))
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        rot = self.rotary(x, pos)
        mask = torch.full((x.shape[1], x.shape[1]), float("-inf"), device=x.device)
        mask = torch.triu(mask, diagonal=1)[None, None]
        out = self.layer(x, attention_mask=mask, position_embeddings=rot,
                         position_ids=pos)
        return out[0] if isinstance(out, tuple) else out


def load_target(device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=dtype).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return tok, model


def corpus_stream(tok, synth_path, wiki_path):
    """SYNTH rendered through the model's own chat template, 4:1 with wikitext.

    SYNTH is what Baguettotron was trained on and it is a reasoning model, so
    raw prose is out of distribution for it. The parquet lives under
    `default/partial-train/`, not `train/` — read the URL off the datasets-server
    /parquet endpoint rather than guessing the path.
    """
    import pyarrow.parquet as pq

    def synth():
        f = pq.ParquetFile(synth_path)
        for batch in f.iter_batches(batch_size=512):
            d = batch.to_pydict()
            for i in range(len(d["language"])):
                if d["language"][i] != "en":
                    continue
                q, r, a = d["query"][i], d["synthetic_reasoning"][i], d["synthetic_answer"][i]
                if not (q and a):
                    continue
                think = f"<think>\n{r}</think>\n" if r else ""
                yield (f"<|im_start|>user\n{q}<|im_end|>\n"
                       f"<|im_start|>assistant\n{think}{a}<|im_end|>\n")

    def wiki():
        f = pq.ParquetFile(wiki_path)
        buf = []
        for batch in f.iter_batches(batch_size=2048):
            for t in batch.column("text").to_pylist():
                if not t or not t.strip():
                    continue
                buf.append(t)
                if sum(len(x) for x in buf) > 6000:
                    yield "".join(buf); buf = []

    s, w = synth(), wiki()
    while True:
        for _ in range(4):
            try:
                yield tok(next(s), add_special_tokens=False).input_ids
            except StopIteration:
                return
        try:
            yield tok(next(w), add_special_tokens=False).input_ids
        except StopIteration:
            pass


def chunks(stream, seq, limit):
    buf, n = [], 0
    for ids in stream:
        buf.extend(ids)
        while len(buf) >= seq and n < limit:
            yield buf[:seq]; buf = buf[seq:]; n += seq
        if n >= limit:
            return


def verify(device, dtype):
    """Three checks that would silently corrupt training if the device differs."""
    tok, model = load_target(device, dtype)
    ok = True

    ids = tok("The aqueduct carries water into the city.", add_special_tokens=False,
              return_tensors="pt").input_ids.to(device)
    with torch.no_grad():
        out = model(ids, output_hidden_states=True)
    delta = (model.lm_head(out.hidden_states[-1]) - out.logits).abs().max().item()
    print(f"[1] lm_head(hidden_states[-1]) vs logits: max|delta| = {delta:.2e}")
    if delta > 1e-2:
        print("    FAIL: the last hidden state is not what feeds the LM head here.")
        ok = False

    tied = model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()
    print(f"[2] embeddings tied: {tied}")
    ok &= tied

    from transformers import AutoConfig
    head = EagleHead(AutoConfig.from_pretrained(MODEL)).to(device)
    f = torch.randn(1, 8, model.config.hidden_size, device=device, dtype=dtype)
    e = model.model.embed_tokens(torch.zeros(1, 8, dtype=torch.long, device=device))
    with torch.no_grad():
        y = head(f.float(), e.float())
    finite = bool(torch.isfinite(y).all())
    print(f"[3] head forward finite on {device}: {finite}  shape={tuple(y.shape)}")
    ok &= finite

    print("VERIFY", "PASS" if ok else "FAIL")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--tokens", type=int, default=5_000_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--loss-pos", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--synth", default="synth0.parquet")
    ap.add_argument("--wiki", default="wiki0.parquet")
    ap.add_argument("--out", default="eagle_head_mac.pt")
    ap.add_argument("--val-seqs", type=int, default=48)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--dtype", default="float32", choices=["float32", "float16", "bfloat16"])
    a = ap.parse_args()

    device = pick_device(a.device)
    dtype = getattr(torch, a.dtype)
    print(f"device={device} dtype={dtype}")
    if a.verify:
        raise SystemExit(0 if verify(device, dtype) else 1)

    from transformers import AutoConfig
    tok, target = load_target(device, dtype)
    cfg = AutoConfig.from_pretrained(MODEL)
    embed, lm_head = target.model.embed_tokens, target.lm_head
    head = EagleHead(cfg).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=a.lr, weight_decay=0.01)
    print(f"trainable: {sum(p.numel() for p in head.parameters())/1e6:.2f}M")

    def target_pass(chunk_ids):
        """One frozen forward. Returns (features, tokens, target's own argmax)."""
        x = torch.tensor([chunk_ids], device=device)
        with torch.no_grad():
            o = target(x, output_hidden_states=True)
        h = o.hidden_states[-1][0, :-1].float()
        return h, x[0, 1:], o.logits[0, 1:].argmax(-1)

    # Held-out set, drawn once and kept resident.
    stream = corpus_stream(tok, a.synth, a.wiki)
    val = []
    for c in chunks(stream, a.seq, a.val_seqs * a.seq):
        val.append(target_pass(c))
    print(f"val: {len(val)} sequences")

    def evaluate():
        head.eval(); hit = tot = 0
        with torch.no_grad():
            for h, t, lab in val:
                pred = head(h[None], embed(t[None]).float())
                for s in range(0, pred.shape[1], 128):
                    am = lm_head(pred[:, s:s + 128]).argmax(-1)[0]
                    hit += int((am == lab[s:s + 128]).sum()); tot += len(am)
        head.train(); return hit / tot

    print(f"untrained acceptance: {evaluate():.4f}")
    log, t0, steps = [], time.perf_counter(), 0
    for ep in range(a.epochs):
        stream = corpus_stream(tok, a.synth, a.wiki)
        batch, seen = [], 0
        for c in chunks(stream, a.seq, a.tokens):
            batch.append(target_pass(c)); seen += a.seq
            if len(batch) < a.batch:
                continue
            h = torch.stack([b[0] for b in batch])
            t = torch.stack([b[1] for b in batch])
            lab = torch.stack([b[2] for b in batch])
            batch = []
            pred = head(h, embed(t).float())
            sel = torch.randint(0, pred.shape[1], (a.loss_pos,), device=device)
            lg = lm_head(pred[:, sel])
            loss = nn.functional.cross_entropy(
                lg.reshape(-1, lg.shape[-1]), lab[:, sel].reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
            opt.step(); steps += 1
            if steps % 100 == 0:
                el = time.perf_counter() - t0
                print(f"  ep{ep+1} step{steps} seen={seen} loss={loss.item():.3f} "
                      f"{seen/el:.0f} tok/s", flush=True)
        acc = evaluate()
        el = time.perf_counter() - t0
        log.append({"epoch": ep + 1, "acceptance": round(acc, 4),
                    "tokens": seen, "minutes": round(el / 60, 1)})
        print(f"[ep{ep+1}] alpha={acc:.4f}  {seen} tokens  {el/60:.1f} min", flush=True)
        torch.save(head.state_dict(), a.out)
        json.dump({"device": str(device), "log": log}, open("eagle_mac.json", "w"), indent=2)


if __name__ == "__main__":
    main()
