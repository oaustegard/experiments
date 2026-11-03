"""Teacher-forced next-token KL between two page exports on the published A8p_q8's own fact answers.

    python3 kl_check.py export/r2_A8p_q8 export/A8p_q8_page [export/base_q8_page ...]

Prints mean KL(first || each other) per position, so a rebuilt model can be compared against the size of
a known difference (A8p vs base, or int8 rounding)."""
import json, sys
from pathlib import Path
import numpy as np
from evalgen import Tok, prompt, session, experiment

HERE = Path(__file__).resolve().parent


def logits_all(sess, ids):
    T = len(ids); B = T - 1
    layers = sum(1 for i in sess.get_inputs() if i.name.endswith(".key"))
    feed = {"input_ids": np.array([ids] * B, np.int64), "attention_mask": np.ones((B, T), np.int64),
            "logits_index": np.arange(B, dtype=np.int64) * T + np.arange(B)}
    for l in range(layers):
        feed[f"past_key_values.{l}.key"] = np.zeros((B, 4, 0, 64), np.float32)
        feed[f"past_key_values.{l}.value"] = np.zeros((B, 4, 0, 64), np.float32)
    lg = sess.run(["logits"], feed)[0][:, 0, :].astype(np.float64)
    lg -= lg.max(-1, keepdims=True)
    return lg - np.log(np.exp(lg).sum(-1, keepdims=True))


def main():
    dirs = sys.argv[1:]
    tok = Tok(HERE / dirs[0] / "tokenizer.json")
    rows = [json.loads(l) for l in open(experiment("monad-chat") / "runs/facts_cpu.jsonl")][:6]
    seqs = [tok.encode(prompt(r["q"]) + "Query")[:160] for r in rows]
    # teacher-force each model on the first model's own greedy continuation
    from ort_gen import generate
    piece = [tok.decode([i]) for i in range(8192)]
    s0 = session(str(HERE / dirs[0]))
    g = generate(s0, seqs, 120, piece, stop=False)["tokens"]
    seqs = [s + t for s, t in zip(seqs, g)]
    ref = [logits_all(s0, s) for s in seqs]
    for d in dirs[1:]:
        s = session(str(HERE / d))
        kl = [float((np.exp(r) * (r - logits_all(s, q))).sum(-1).mean()) for r, q in zip(ref, seqs)]
        agree = [float((r.argmax(-1) == logits_all(s, q).argmax(-1)).mean()) for r, q in zip(ref, seqs)]
        print(d, "KL", round(float(np.mean(kl)), 5), "top1 agree", round(float(np.mean(agree)), 4), flush=True)


if __name__ == "__main__":
    main()
