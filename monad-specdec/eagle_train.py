"""Train an EAGLE head for Baguettotron on 4 CPU cores, and measure acceptance.

The head is EAGLE's autoregression module: an FC layer mapping
[target_feature ; token_embedding] down to hidden size, then one decoder layer.
The embedding and LM head are the target's own frozen (and tied) weights, so
only 4.2M parameters train.

Acceptance here is alpha at gamma=1 under teacher forcing: the fraction of
held-out positions where the head's argmax equals the target's argmax. Free-
running acceptance can differ, because errors compound once the draft starts
conditioning on its own output.
"""
import json, math, statistics, time
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaDecoderLayer, LlamaRotaryEmbedding

torch.set_num_threads(4)
torch.manual_seed(0)

DATA = "/tmp/specdec/eagle_data.npz"
EPOCHS = 12
BATCH_SEQ = 4          # sequences per step
LOSS_POS = 128         # positions per sequence scored through the LM head
LR = 3e-4


class EagleHead(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Linear(2 * cfg.hidden_size, cfg.hidden_size, bias=False)
        self.layer = LlamaDecoderLayer(cfg, layer_idx=0)
        self.rotary = LlamaRotaryEmbedding(cfg)

    def forward(self, feat, emb):
        x = self.fc(torch.cat([feat, emb], dim=-1))
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        rot = self.rotary(x, pos)
        mask = torch.full((x.shape[1], x.shape[1]), float("-inf"))
        mask = torch.triu(mask, diagonal=1)[None, None]
        out = self.layer(x, attention_mask=mask, position_embeddings=rot,
                         position_ids=pos)
        return out[0] if isinstance(out, tuple) else out


def main():
    d = np.load(DATA)
    h, tk, lb, seq = d["h"], d["tok"], d["lab"], int(d["seq"])
    L = seq - 1
    n_seq = h.shape[0] // L
    h = h[:n_seq * L].reshape(n_seq, L, -1)
    tk = tk[:n_seq * L].reshape(n_seq, L)
    lb = lb[:n_seq * L].reshape(n_seq, L)
    n_val = max(4, n_seq // 10)
    print(f"{n_seq} sequences of {L}  ->  train {n_seq-n_val}, val {n_val}", flush=True)

    cfg = AutoConfig.from_pretrained("PleIAs/Baguettotron")
    target = AutoModelForCausalLM.from_pretrained("PleIAs/Baguettotron",
                                                  dtype=torch.float32).eval()
    embed, lm_head = target.model.embed_tokens, target.lm_head
    for p in target.parameters():
        p.requires_grad_(False)

    head = EagleHead(cfg)
    n_train_params = sum(p.numel() for p in head.parameters() if p.requires_grad)
    print(f"trainable: {n_train_params/1e6:.2f}M", flush=True)
    opt = torch.optim.AdamW(head.parameters(), lr=LR, weight_decay=0.01)

    def evaluate(model, max_seq=None):
        model.eval()
        hit = tot = 0
        with torch.no_grad():
            rng = range(n_seq - n_val, n_seq if max_seq is None
                        else min(n_seq, n_seq - n_val + max_seq))
            for i in rng:
                feat = torch.from_numpy(h[i]).float().unsqueeze(0)
                toks = torch.from_numpy(tk[i].astype(np.int64)).unsqueeze(0)
                pred = model(feat, embed(toks))
                # Score in slices; a full 511 x 65536 logit tensor is 134 MB.
                for s in range(0, L, 128):
                    lg = lm_head(pred[:, s:s + 128])
                    am = lg.argmax(-1)[0]
                    gold = torch.from_numpy(lb[i][s:s + 128].astype(np.int64))
                    hit += int((am == gold).sum()); tot += len(gold)
        model.train()
        return hit / tot

    base = evaluate(head, max_seq=4)
    print(f"untrained acceptance: {base:.4f}", flush=True)

    steps = 0
    t0 = time.perf_counter()
    log = []
    order = np.arange(n_seq - n_val)
    for ep in range(EPOCHS):
        np.random.shuffle(order)
        for b in range(0, len(order), BATCH_SEQ):
            idx = order[b:b + BATCH_SEQ]
            if len(idx) < BATCH_SEQ:
                continue
            feat = torch.from_numpy(h[idx]).float()
            toks = torch.from_numpy(tk[idx].astype(np.int64))
            with torch.no_grad():
                emb = embed(toks)
            pred = head(feat, emb)
            sel = torch.randint(0, L, (LOSS_POS,))
            lg = lm_head(pred[:, sel])
            gold = torch.from_numpy(lb[idx][:, sel.numpy()].astype(np.int64))
            loss = nn.functional.cross_entropy(lg.reshape(-1, lg.shape[-1]),
                                               gold.reshape(-1))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 0.5)
            opt.step()
            steps += 1
            if steps % 10 == 0:
                el = time.perf_counter() - t0
                print(f"  ep{ep} step {steps} loss {loss.item():.3f} "
                      f"{el/steps:.2f}s/step", flush=True)
                log.append({"step": steps, "loss": round(loss.item(), 4),
                            "sec": round(el, 1)})
            if steps % 100 == 0:
                a = evaluate(head, max_seq=4)
                print(f"  >> step {steps} acceptance {a:.4f}", flush=True)
                log.append({"step": steps, "acceptance": round(a, 4)})
                torch.save(head.state_dict(), "/tmp/specdec/eagle_head.pt")

    final = evaluate(head)
    el = time.perf_counter() - t0
    torch.save(head.state_dict(), "/tmp/specdec/eagle_head.pt")
    out = {"trainable_params_m": round(n_train_params / 1e6, 2),
           "train_sequences": int(n_seq - n_val), "val_sequences": int(n_val),
           "seq_len": L, "epochs": EPOCHS, "steps": steps,
           "train_tokens": int((n_seq - n_val) * L),
           "wall_clock_s": round(el, 1), "sec_per_step": round(el / steps, 2),
           "untrained_acceptance": round(base, 4),
           "trained_acceptance_alpha_gamma1": round(final, 4),
           "log": log}
    print(json.dumps({k: v for k, v in out.items() if k != "log"}, indent=2))
    json.dump(out, open("eagle_train.json", "w"), indent=2)


if __name__ == "__main__":
    main()
