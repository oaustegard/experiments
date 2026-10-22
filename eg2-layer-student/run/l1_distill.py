"""L1 (and L2 with --vocab): EG2's first k layers trained so their query vectors land on EG2's.

Usage: python3 -I run/l1_distill.py --k 4 [--epochs 1] [--lr 3e-3] [--vocab out/vocab/k32000] [--init out/l1/k4/final.pt]

Student = run/model.py Student(k), initialised from the q4 teacher's own weights, so step 0
is L0. Loss = 1 - cos(student, EG2) on the #102 targets (165K: msmarco_train, synth_train,
gpu_more, gpu_posts snippets rebuilt from post text). Everything but embed_tokens trains:
the 262,144-row table stays the teacher's, so a vocabulary cut after training slices the
same rows the student learned against. With --vocab DIR, the table is DIR/rows.json and
texts are tokenized with DIR/tokenizer.json (the A4 cut), and the cut table trains too.
Batches are length-bucketed. Checkpoints out/l1/<tag>/last (trainable weights, optimizer, position) every --save-every
steps and resumes from it; scores the test sets every --eval-every steps.
Writes results/l1_<tag>.json.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from tokenizers import Tokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bench import TESTS, Bench  # noqa: E402
from model import PREV, ROOT, Student, batches, embed, load, tokenize  # noqa: E402
from common import Teacher  # noqa: E402
from build_web import snippet  # noqa: E402

SETS = ["msmarco_train", "synth_train", "gpu_more", "gpu_posts"]


def train_texts(names):
    posts = None
    texts, Y = [], []
    for n in names:
        rows, E = load(n)
        if n == "gpu_posts":
            if posts is None:
                posts = [json.loads(l)["text"] for l in (PREV / "out" / "posts.jsonl").open()]
            keep = [i for i, r in enumerate(rows) if posts[r["post"]]]
            texts += [snippet(posts[rows[i]["post"]]) for i in keep]
            Y.append(E[keep])
        else:
            texts += [r["q"] for r in rows]
            Y.append(E)
    return texts, np.concatenate(Y)


def epoch_batches(lens, batch, ep):
    """Shuffled batches of similar length: random chunks of 50 batches, sorted by length
    inside each chunk, then the batch order shuffled. Padding drops from ~69 tokens per
    row (random batches) to about the mean of 26. Deterministic per epoch, so a resume
    lands on the same batches."""
    rng = np.random.default_rng(ep)
    perm = rng.permutation(len(lens))
    out = []
    for c in range(0, len(perm), batch * 50):
        chunk = perm[c:c + batch * 50]
        chunk = chunk[np.argsort(lens[chunk], kind="stable")]
        out += [chunk[i:i + batch] for i in range(0, len(chunk), batch)]
    return [out[i] for i in rng.permutation(len(out))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--train", nargs="+", default=SETS)
    ap.add_argument("--vocab", help="dir with rows.json + tokenizer.json (pruned vocabulary)")
    ap.add_argument("--train-embed", action="store_true", help="train embed_tokens too (implied by --vocab)")
    ap.add_argument("--init", help="start from this state dict (e.g. out/l1/k4/final.pt) instead of the teacher's weights")
    ap.add_argument("--seed", type=int, default=0, help="offset for the batch order (a continuation should not replay it)")
    ap.add_argument("--device", default="cpu", help="cpu, mps (Apple GPU) or cuda; auto picks the best present")
    ap.add_argument("--tag")
    ap.add_argument("--save-every", type=int, default=300)
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--max-steps", type=int, help="stop after this many steps (lr probes)")
    ap.add_argument("--img-mix", type=int, default=0,
                    help="add the image-search caption targets (run/img_teacher.py): train-split alt texts repeated "
                         "this many times plus train-split post texts once")
    ap.add_argument("--bluesky-n", type=int, help="keep a fixed random subset of this many Bluesky targets")
    a = ap.parse_args()
    tag = a.tag or f"k{a.k}"
    torch.set_num_threads(4)
    torch.manual_seed(0)
    if a.device == "auto":
        a.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    dev = torch.device(a.device)
    T = Teacher()
    rows = tok = None
    if a.vocab:
        vd = ROOT / a.vocab
        rows = json.loads((vd / "rows.json").read_text())
        tok = Tokenizer.from_file(str(vd / "tokenizer.json"))
    m = Student(a.k, rows)
    if a.init:
        sd = torch.load(ROOT / a.init)
        w = sd.get("tm.embed_tokens.weight")
        if rows is not None and w is not None and len(w) != len(rows):
            sd["tm.embed_tokens.weight"] = w[torch.as_tensor(rows)]
        assert not m.load_state_dict(sd, strict=False).unexpected_keys
    train_embed = a.train_embed or bool(a.vocab)
    m.tm.embed_tokens.weight.requires_grad_(train_embed)
    m.to(dev)
    texts, Y = train_texts(a.train)
    if a.bluesky_n:
        keep = np.sort(np.random.default_rng(7).choice(len(texts), a.bluesky_n, replace=False))
        texts, Y = [texts[i] for i in keep], Y[keep]
    if a.img_mix:
        img = ROOT / "out" / "img"
        for name, rep in (("train_alt", a.img_mix), ("train_post", 1)):
            t = json.loads((img / f"teacher_{name}.json").read_text())
            texts += t * rep
            Y = np.concatenate([Y] + [np.load(img / f"teacher_{name}.npy")] * rep)
    enc = tokenize(T, texts, remap=tok)
    B = Bench()
    test_enc = {t: tokenize(T, B.texts(t), remap=tok) for t in TESTS}
    work = ROOT / "out" / "l1" / tag
    work.mkdir(parents=True, exist_ok=True)
    resp = ROOT / "results" / f"l1_{tag}.json"
    res = json.loads(resp.read_text()) if resp.exists() else {
        "k": a.k, "train": a.train, "n_train": len(texts), "lr": a.lr, "batch": a.batch,
        "epochs": a.epochs, "img_mix": a.img_mix, "bluesky_n": a.bluesky_n, "vocab": a.vocab, "init": a.init, "device": a.device, "seed": a.seed, "trains": "all but embed_tokens" if not train_embed else "all",
        "evals": {}}
    params = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    steps_per_epoch = (len(texts) + a.batch - 1) // a.batch
    total = int(round(steps_per_epoch * a.epochs))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * max(0.0, 1 - s / total))
    step = 0
    if (work / "last" / "pos.json").exists():
        sd = torch.load(work / "last" / "weights.pt", map_location=dev)
        m.load_state_dict(sd, strict=False)
        opt.load_state_dict(torch.load(work / "last" / "opt.pt", map_location=dev))
        step = json.loads((work / "last" / "pos.json").read_text())["step"]
        for _ in range(step):
            sched.step()
        print("resumed at", step, flush=True)

    def evaluate(name):
        r, _ = B.score({t: embed(m, test_enc[t]) for t in TESTS})
        res["evals"][name] = r
        resp.write_text(json.dumps(res, indent=1))
        print(name, json.dumps({t: r[t]["overlap@10"][0] for t in TESTS}),
              "hit", round(r["synth_test"]["hit@10"][0], 4), flush=True)

    def save():
        (work / "last").mkdir(exist_ok=True)
        torch.save({n: p.detach().cpu() for n, p in m.named_parameters() if p.requires_grad}, work / "last" / "weights.pt")
        torch.save(opt.state_dict(), work / "last" / "opt.pt")
        (work / "last" / "pos.json").write_text(json.dumps({"step": step}))

    if step == 0 and "step0" not in res["evals"]:
        evaluate("step0")
    Yt = torch.from_numpy(Y).to(dev)
    lens = np.array([len(e) for e in enc])
    plan = {}
    stop = min(total, a.max_steps or total)
    t0, tot, n = time.time(), 0.0, 0
    while step < stop:
        ep, s = divmod(step, steps_per_epoch)
        if ep not in plan:
            plan = {ep: epoch_batches(lens, a.batch, ep + a.seed)}
        b = plan[ep][s]
        m.train()
        _, ids, mask = next(batches(enc, len(b), order=b))
        out = m(ids.to(dev), mask.to(dev))
        loss = (1 - (out * Yt[torch.as_tensor(b, device=dev)]).sum(1)).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        step += 1
        tot, n = tot + loss.item(), n + 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {tot / n:.5f} {(time.time() - t0) / n:.2f}s/step", flush=True)
            tot, n, t0 = 0.0, 0, time.time()
        if step % a.save_every == 0 or step == stop:
            save()
        if step % a.eval_every == 0 or step == stop:
            evaluate(f"step{step}")
    if step >= total:
        torch.save({n: t.cpu() for n, t in m.state_dict().items()}, work / "final.pt")
        print("done", flush=True)


if __name__ == "__main__":
    main()
