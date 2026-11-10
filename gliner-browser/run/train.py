"""Train one GLiNER arm with the upstream library's model, processor, collator and Trainer.

    python3 run/train.py --name A4 --backbone jhu-clsp/ettin-encoder-17m --span-mode markerV0 --rnn 0 \
        --lr-enc 1.5e-4 --lr-other 3e-4 --batch 32 --steps 12000 --eval-every 1000 [--prefix-space 1] ...

Not upstream train.py. What that script drops or gets wrong, and what this one does instead:
- shuffle_types, random_drop, size_sup and label_smoothing are read by no library code path at
  2dc5de2. Here they are model-config keys (saved in gliner_config.json) and take effect with the
  shipped config_span.yaml values: shuffle_types true (the prompt's type order is shuffled per
  example), random_drop true (after the shuffle keep a uniform 1..len(types) prefix of the types,
  then cut to max_types, the rule of GLiNER 0.1.12 gliner/modules/base.py collate_fn), size_sup -1
  (no subsampling; >0 keeps a seeded random subset), label_smoothing 0 (passed to the Trainer).
- warmup is a 10% ratio, then the shipped cosine schedule. (upstream passes warmup_steps: 0.05,
  which transformers 4.x treats as 0.05 steps, i.e. no warmup.) Loss: focal with the shipped
  alpha 0.75, gamma 0, prob_margin 0, reduction sum, negatives 1.0, masking none; weight decay 0.1
  encoder / 0.01 other; max_grad_norm 10. bf16 only with --amp 1 (upstream hardcodes it on).
- every --eval-every steps the CrossNER dev splits are benched (bench.run_bench), a line goes to
  out/<name>/progress.jsonl and a checkpoint is saved; HF keeps the best-by-dev and the last.
  Best-by-dev = highest 5-set dev mean F1 over thresholds 0.3-0.6.
- --prefix-space 1 sets add_prefix_space on the ByteLevel pre-tokenizer of the tokenizer the
  library built, so every text word and every label word is tokenized in its G-dot form, at train
  and at inference (stored in the tokenizer and in gliner_config.json as prefix_space).
- resumes from the last checkpoint in out/<name>/ckpt (weights, optimizer, scheduler, RNG, best).
At the end: reload the best checkpoint with GLiNER.from_pretrained, bench dev+test, write
results/<name>.json, save the model to out/<name>/best/ (save_pretrained), print `ARM_DONE <name>`.
Any failure, including a batch the library's Trainer skipped for OOM or a non-finite loss, exits 1.
"""
import argparse
import json
import logging
import math
import random
import sys
import time
import traceback
import warnings
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bench  # noqa: E402

D = Path(__file__).resolve().parents[1]

# configs/config_span.yaml (training: section) at 2dc5de2; the keys upstream train.py drops are here too
SHIPPED = {"loss_alpha": 0.75, "loss_gamma": 0, "loss_prob_margin": 0, "label_smoothing": 0,
           "loss_reduction": "sum", "negatives": 1.0, "masking": "none", "weight_decay_encoder": 0.1,
           "weight_decay_other": 0.01, "max_grad_norm": 10.0, "scheduler_type": "cosine",
           "shuffle_types": True, "random_drop": True, "size_sup": -1}
WARMUP_RATIO = 0.1
DEMO_WORDS = [["John", "works", "at", "Google", "in", "Mountain", "View", "."]]
DEMO_LABELS = ["person", "organization name", "location"]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class TrainSampling:
    """Mixin over the library processor: batch_generate_class_mappings (processor.py:320-369 at
    2dc5de2) with shuffle_types and random_drop read from the config. Everything else is the
    library's code path unchanged; the mapping is only built for training batches (inference and
    evaluation pass a fixed label list)."""

    def batch_generate_class_mappings(self, batch_list, negatives=None, key="ner", sampled_neg=100):
        from gliner.data_processing.utils import get_negatives
        if negatives is None:
            negatives = get_negatives(batch_list, sampled_neg=sampled_neg, key=key)
        shuffle = bool(getattr(self.config, "shuffle_types", True))
        drop = bool(getattr(self.config, "random_drop", False))
        class_to_ids, id_to_classes = [], []
        for b in batch_list:
            max_neg_type_ratio = int(self.config.max_neg_type_ratio)
            neg_type_ratio = random.randint(0, max_neg_type_ratio) if max_neg_type_ratio else 0
            if f"{key}_negatives" in b:
                negs_i = b[f"{key}_negatives"]
            else:
                negs_i = negatives[: len(b[key]) * neg_type_ratio] if neg_type_ratio else []
            if f"{key}_labels" in b:
                types = b[f"{key}_labels"]
            else:
                types = list(set([el[-1] for el in b[key]] + negs_i))
                if shuffle:
                    random.shuffle(types)
                else:
                    types = sorted(types)       # set order is hash-seeded; sorted is the unshuffled order
                if drop and types:              # GLiNER 0.1.12 modules/base.py:95-99
                    types = types[: random.randint(1, len(types))]
                types = types[: int(self.config.max_types)]
            if key == "ner" and not b[key] and not types:
                types = [""]
            class_to_id = {k: v for v, k in enumerate(types, start=1)}
            class_to_ids.append(class_to_id)
            id_to_classes.append({v: k for k, v in class_to_id.items()})
        return class_to_ids, id_to_classes


def add_train_sampling(proc):
    base = type(proc)
    proc.__class__ = type(f"TrainSampling{base.__name__}", (base,),
                          {"batch_generate_class_mappings": TrainSampling.batch_generate_class_mappings})


def tokenization_check(model, backbone, prefix_space):
    """Ids the library's processor produces for one example, against an untouched upstream build."""
    from transformers import AutoTokenizer
    ref_tok = AutoTokenizer.from_pretrained(backbone)
    type(model)._set_tokenizer_spec_tokens(ref_tok)
    ref_tok.add_tokens(model._get_special_tokens(), special_tokens=True)
    ref = model.data_processor_class(model.config, ref_tok, None).tokenize_inputs(DEMO_WORDS, DEMO_LABELS)
    ours = model.data_processor.tokenize_inputs(DEMO_WORDS, DEMO_LABELS)
    tok = model.data_processor.transformer_tokenizer
    r_ids, o_ids = ref["input_ids"][0].tolist(), ours["input_ids"][0].tolist()
    wm = ours["words_mask"][0].tolist()
    first_sub = [tok.convert_ids_to_tokens(o_ids[i]) for i, w in enumerate(wm) if w > 0]
    chk = {"words": DEMO_WORDS[0], "labels": DEMO_LABELS, "upstream_ids": r_ids,
           "upstream_tokens": ref_tok.convert_ids_to_tokens(r_ids), "ids": o_ids,
           "tokens": tok.convert_ids_to_tokens(o_ids),
           # one first-subtoken mark per text word, 1..n in order, in both builds
           "words_mask_equal": [w for w in ref["words_mask"][0].tolist() if w > 0] == [w for w in wm if w > 0]
                               == list(range(1, len(DEMO_WORDS[0]) + 1)),
           "prefix_space": bool(prefix_space), "ids_equal_upstream": r_ids == o_ids}
    log(f"tokenization upstream : {chk['upstream_tokens']}")
    log(f"tokenization this arm : {chk['tokens']}")
    log(f"ids upstream {r_ids}")
    log(f"ids this arm {o_ids}  equal={r_ids == o_ids} (prefix_space={int(prefix_space)})")
    if not chk["words_mask_equal"]:
        raise RuntimeError("words_mask differs from upstream")
    if not prefix_space and r_ids != o_ids:
        raise RuntimeError("--prefix-space 0 does not reproduce upstream ids")
    if prefix_space and (r_ids == o_ids or not all(t.startswith("Ġ") for t in first_sub)):
        raise RuntimeError(f"--prefix-space 1 did not give G-dot word starts: {first_sub}")
    return chk


def read_progress(path):
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_statedict(model, path):
    sd = model.model.state_dict()
    lines = [f"{k}\t{list(v.shape)}\t{v.dtype}".replace("torch.", "") for k, v in sd.items()]
    lines.append(f"# {len(sd)} tensors, {sum(v.numel() for v in sd.values()):,} elements")
    path.write_text("\n".join(lines) + "\n")


def param_counts(model):
    groups = {}
    for n, p in model.model.named_parameters():
        g = n.split(".")[0]
        if g == "token_rep_layer" and ".embeddings.tok_embeddings" in n:
            g = "token_rep_layer.tok_embeddings"
        groups[g] = groups.get(g, 0) + p.numel()
    return {"total": sum(groups.values()), **groups}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--backbone", default="jhu-clsp/ettin-encoder-17m")
    ap.add_argument("--span-mode", default="markerV0", choices=["markerV0", "token_level"])
    ap.add_argument("--rnn", type=int, default=1, choices=[0, 1], help="num_rnn_layers (BiLSTM)")
    ap.add_argument("--hidden", type=int, default=None, help="default: backbone hidden size (no projection)")
    ap.add_argument("--lr-enc", type=float, default=1e-5)
    ap.add_argument("--lr-other", type=float, default=5e-5)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--dropout", type=float, default=0.4)
    ap.add_argument("--prefix-space", type=int, default=0, choices=[0, 1])
    ap.add_argument("--max-len", type=int, default=384)
    ap.add_argument("--max-width", type=int, default=12)
    ap.add_argument("--max-types", type=int, default=25)
    ap.add_argument("--max-neg-type-ratio", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--amp", type=int, default=0, choices=[0, 1], help="bf16 autocast in the Trainer")
    ap.add_argument("--eval-every", type=int, default=1000)
    ap.add_argument("--data", type=Path, default=D / "data/train/pile_mistral.json")
    ap.add_argument("--out", type=Path, default=D / "out")
    ap.add_argument("--results", type=Path, default=D / "results")
    ap.add_argument("--bench-limit", type=int, default=None, help="first N sentences per bench split")
    ap.add_argument("--bench-batch", type=int, default=12)
    ap.add_argument("--workers", type=int, default=1, help="dataloader workers")
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--crash-after-save", type=int, default=0, help="test hook: die right after this step's checkpoint")
    a = ap.parse_args()
    t_start = time.time()

    warnings.filterwarnings("ignore", message="Sentence of length")
    import transformers
    from transformers import AutoConfig, PrinterCallback, TrainerCallback
    from transformers.trainer_utils import get_last_checkpoint
    from gliner import GLiNER
    from gliner.training import Trainer as GLiNERTrainer
    transformers.logging.set_verbosity_error()

    cuda = torch.cuda.is_available()
    arm_dir = (a.out / a.name).resolve()
    ckpt_dir = arm_dir / "ckpt"
    progress_path = arm_dir / "progress.jsonl"
    arm_dir.mkdir(parents=True, exist_ok=True)
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    resume_step = int(Path(last).name.split("-")[-1]) if last else 0

    hidden = a.hidden or AutoConfig.from_pretrained(a.backbone).hidden_size
    model_cfg = {"model_name": a.backbone, "name": a.name, "span_mode": a.span_mode, "hidden_size": hidden,
                 "num_rnn_layers": a.rnn, "dropout": a.dropout, "max_width": a.max_width, "max_len": a.max_len,
                 "max_types": a.max_types, "max_neg_type_ratio": a.max_neg_type_ratio, "subtoken_pooling": "first",
                 "fuse_layers": False, "post_fusion_schema": "", "words_splitter_type": "whitespace",
                 "prefix_space": bool(a.prefix_space), "shuffle_types": SHIPPED["shuffle_types"],
                 "random_drop": SHIPPED["random_drop"], "size_sup": SHIPPED["size_sup"]}
    transformers.set_seed(a.seed)
    model = GLiNER.from_config(model_cfg)
    tok = model.data_processor.transformer_tokenizer
    if a.prefix_space:
        bench.set_prefix_space(tok, True)
    add_train_sampling(model.data_processor)
    tok_check = tokenization_check(model, a.backbone, a.prefix_space)
    params = param_counts(model)
    log(f"arm {a.name}: {type(model).__name__} params {params}")
    log(f"resume from {last}" if last else "fresh start")

    data = json.loads(a.data.read_text())
    if SHIPPED["size_sup"] > 0:
        data = random.Random(a.seed).sample(data, SHIPPED["size_sup"])
    log(f"train passages {len(data)} from {a.data}")

    # progress lines past the checkpoint we resume from were written by a run that died before saving
    prog = [p for p in read_progress(progress_path) if p["step"] <= resume_step]
    progress_path.write_text("".join(json.dumps(p) + "\n" for p in prog))
    sec_offset = prog[-1]["seconds"] if prog else 0.0
    t_train0 = time.time()

    oom = {"n": 0}

    class OOMCounter(logging.Handler):
        def emit(self, record):
            if "Skipping batch" in record.getMessage():
                oom["n"] += 1
    logging.getLogger("gliner.training.trainer").addHandler(OOMCounter())

    loss_log = []   # (step, mean loss over the steps since the previous log)
    timing = {"bench": 0.0}
    resume_check = {}

    class Watch(TrainerCallback):
        def __init__(self):
            self.t, self.s, self.b, self.dead = time.time(), None, 0.0, 0

        def on_step_end(self, args, state, control, **kw):
            if oom["n"]:
                raise RuntimeError(f"the Trainer skipped {oom['n']} batch(es) for OOM")
            if state.global_step >= state.max_steps:
                control.should_evaluate = True      # always bench the final step
            return control

        def on_train_begin(self, args, state, control, model=None, **kw):
            if not last:
                return
            # the Trainer has loaded the checkpoint by now (strict=False): prove every tensor came back
            from safetensors.torch import load_file
            saved = load_file(str(Path(last) / "model.safetensors"))
            live = model.model.state_dict()
            missing = [k for k in live if k not in saved]
            differ = [k for k in live if k in saved and not torch.equal(live[k].detach().cpu(), saved[k])]
            resume_check.update({"from": Path(last).name, "step": state.global_step, "tensors": len(live),
                                 "missing_in_file": missing, "differ": differ})
            log(f"resume check: step {state.global_step}, {len(live) - len(missing) - len(differ)}/{len(live)} "
                f"tensors identical to {Path(last).name}; missing {missing}; differ {differ}")
            with open(arm_dir / "resumes.jsonl", "a") as f:
                f.write(json.dumps({**resume_check, "at": time.strftime("%Y-%m-%dT%H:%M:%S")}) + "\n")
            if missing or differ:
                raise RuntimeError("resumed weights do not match the checkpoint")

        def on_save(self, args, state, control, **kw):
            if a.crash_after_save and state.global_step == a.crash_after_save:
                log(f"--crash-after-save: exiting after the checkpoint at step {state.global_step}")
                sys.stdout.flush()
                import os
                os._exit(3)     # simulate a dead VM: no cleanup, no final bench

        def on_log(self, args, state, control, logs=None, **kw):
            if logs and "loss" in logs:
                if not math.isfinite(logs["loss"]):
                    raise RuntimeError(f"non-finite loss {logs['loss']} at step {state.global_step}")
                # A head whose logits all fall below logit(1e-6) gets exactly zero gradient (the focal
                # loss clamps p at 1e-6) and never recovers; seen in smoke tests at lr-other 3e-4 with a
                # 3-step warmup. Two consecutive logs with a vanished gradient end the arm.
                gn = logs.get("grad_norm")
                self.dead = (self.dead + 1) if (gn is not None and gn < 1e-8) else 0
                if self.dead >= 2:
                    raise RuntimeError(f"collapsed head: grad_norm {gn} at step {state.global_step}")
                loss_log.append((state.global_step, logs["loss"]))
                now = time.time()
                prev = self.s if self.s is not None else resume_step
                # seconds per optimizer step since the previous log, dev-bench time excluded
                sps = (now - self.t - (timing["bench"] - self.b)) / max(1, state.global_step - prev)
                self.t, self.s, self.b = now, state.global_step, timing["bench"]
                log(f"step {state.global_step} loss {logs['loss']:.4f} lr {logs.get('learning_rate', 0):.3g} "
                    f"grad_norm {logs.get('grad_norm', float('nan')):.3g} {sps:.3f} s/step")

    class ArmTrainer(GLiNERTrainer):
        last_eval_step = resume_step

        def evaluate(self, eval_dataset=None, ignore_keys=None, metric_key_prefix="eval"):
            step = self.state.global_step
            t0 = time.time()
            py_state = random.getstate()
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if cuda else []):
                res = bench.run_bench(self.model, splits=("dev",), sets=bench.CROSSNER, limit=a.bench_limit,
                                      batch_size=a.bench_batch, log=None)
            random.setstate(py_state)
            timing["bench"] += time.time() - t0
            since =[(s, l) for s, l in loss_log if s > self.last_eval_step]
            w, prev = [], self.last_eval_step
            for s, l in since:
                w.append((s - prev, l))
                prev = s
            loss = sum(n * l for n, l in w) / sum(n for n, _ in w) if w else None
            line = {"step": step, "loss": loss, "dev_mean5": res["dev_mean5"],
                    "dev_best_threshold": res["dev_best_threshold"], "dev_best": res["dev_best_mean5"],
                    "dev_f1_at_0.5": {s: res["sets"][s]["dev"]["by_threshold"]["0.5"]["f1"] for s in bench.CROSSNER},
                    "lr": self._get_learning_rate(), "seconds": round(sec_offset + time.time() - t_train0, 1),
                    "bench_seconds": round(time.time() - t0, 1)}
            with open(progress_path, "a") as f:
                f.write(json.dumps(line) + "\n")
            self.last_eval_step = step
            log(f"EVAL step {step} loss {loss if loss is None else round(loss, 4)} dev_mean5 "
                + " ".join(f"{k}:{v * 100:.2f}" for k, v in res["dev_mean5"].items()) + f" ({line['bench_seconds']}s)")
            metrics = {"eval_dev_best": res["dev_best_mean5"],
                       **{f"eval_dev_f1_{k}": v for k, v in res["dev_mean5"].items()}}
            self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
            return metrics

    args = model.create_training_args(
        output_dir=str(ckpt_dir), learning_rate=a.lr_enc, weight_decay=SHIPPED["weight_decay_encoder"],
        others_lr=a.lr_other, others_weight_decay=SHIPPED["weight_decay_other"],
        focal_loss_alpha=SHIPPED["loss_alpha"], focal_loss_gamma=SHIPPED["loss_gamma"],
        focal_loss_prob_margin=SHIPPED["loss_prob_margin"], label_smoothing=SHIPPED["label_smoothing"],
        loss_reduction=SHIPPED["loss_reduction"], negatives=SHIPPED["negatives"], masking=SHIPPED["masking"],
        loss_type="focal", lr_scheduler_type=SHIPPED["scheduler_type"], warmup_ratio=WARMUP_RATIO,
        per_device_train_batch_size=a.batch, per_device_eval_batch_size=a.batch,
        max_grad_norm=SHIPPED["max_grad_norm"], max_steps=a.steps, save_strategy="steps", save_steps=a.eval_every,
        # save_total_limit=2 with a best metric keeps the best and the newest checkpoint (HF bubbles the
        # best to second-newest); 1 would make HF delete the last one when training ends
        eval_strategy="steps", eval_steps=a.eval_every, save_total_limit=2, metric_for_best_model="dev_best",
        greater_is_better=True, logging_steps=a.log_every, logging_first_step=True, use_cpu=not cuda,
        bf16=bool(a.amp), dataloader_num_workers=a.workers, report_to="none", seed=a.seed,
        remove_unused_columns=False, disable_tqdm=True, save_safetensors=True)
    trainer = ArmTrainer(model=model, args=args, train_dataset=data,
                         eval_dataset=["unused: evaluate() benches the CrossNER dev splits"],
                         data_collator=model._create_data_collator(), processing_class=tok, callbacks=[Watch()])
    trainer.remove_callback(PrinterCallback)    # Watch prints its own log lines
    if resume_step < a.steps:
        trainer.train(resume_from_checkpoint=last)
    else:
        log(f"checkpoint {last} is at max steps; skipping to the final bench")
    train_seconds = round(sec_offset + time.time() - t_train0, 1)

    last = get_last_checkpoint(str(ckpt_dir))
    state = json.loads((Path(last) / "trainer_state.json").read_text())
    best_step = state.get("best_global_step") or int(Path(last).name.split("-")[-1])
    best_dir = ckpt_dir / f"checkpoint-{best_step}"
    log(f"best step {best_step} (best_metric {state.get('best_metric')}), loading {best_dir}")
    del trainer, model
    device = "cuda" if cuda else "cpu"
    best = bench.load_model(best_dir, device)
    if bench.prefix_space_state(best.data_processor.transformer_tokenizer) != bool(a.prefix_space):
        raise RuntimeError("reloaded tokenizer lost the prefix-space setting")
    res = bench.run_bench(best, splits=("dev", "test"), limit=a.bench_limit, batch_size=a.bench_batch,
                          log=lambda m: log(m))
    out_best = arm_dir / "best"
    best.save_pretrained(str(out_best), safe_serialization=True)
    write_statedict(best, arm_dir / "statedict.txt")
    prog = read_progress(progress_path)
    at_best = next((p for p in prog if p["step"] == best_step), None)
    result = {
        "name": a.name, "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(a).items()},
        "model_config": model_cfg, "training": {**SHIPPED, "warmup_ratio": WARMUP_RATIO, "loss_type": "focal",
                                                "optimizer": "adamw_torch", "save_total_limit": 2,
                                                "pythonhashseed": __import__("os").environ.get("PYTHONHASHSEED")},
        "versions": bench.versions(), "params": params, "best_step": best_step,
        "best_dev_during_training": at_best and at_best["dev_best"], "best_dev_reloaded": res.get("dev_best_mean5"),
        "train_seconds": train_seconds, "wall_seconds_this_invocation": round(time.time() - t_start, 1),
        "resumed_from_step": resume_step, "resumes": read_progress(arm_dir / "resumes.jsonl"),
        "tokenization_check": tok_check, "progress": prog, "bench": res,
        "best_dir": str(out_best.relative_to(D)) if out_best.is_relative_to(D) else str(out_best)}
    a.results.mkdir(parents=True, exist_ok=True)
    (a.results / f"{a.name}.json").write_text(json.dumps(result, indent=1))
    log("test_mean7 " + " ".join(f"{k}:{v * 100:.2f}" for k, v in res["test_mean7"].items())
        + f" | dev-chosen t={res['dev_best_threshold']} test {res['test_mean7_at_dev_best'] * 100:.2f}")
    print(f"ARM_DONE {a.name}", flush=True)


if __name__ == "__main__":
    # The library builds each example's type list from set() before shuffling it, so the prompts
    # (and every number after them) depend on str hashing. Pin the hash seed for reproducible runs
    # and bit-identical resumes.
    import os
    if os.environ.get("PYTHONHASHSEED") is None:
        os.environ["PYTHONHASHSEED"] = "0"
        os.execv(sys.executable, [sys.executable, *sys.argv])
    try:
        main()
    except SystemExit:
        raise
    except BaseException:
        traceback.print_exc()
        sys.exit(1)
