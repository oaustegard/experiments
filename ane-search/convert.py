import sys, time, json, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
import coremltools as ct
from transformers import BertModel
from tokenizers import Tokenizer
from safetensors.torch import load_file
from ane_bert import load
HF = "/Users/austegard/Projects/experiments/leaf-wasm/out/hf"
tok = Tokenizer.from_file(f"{HF}/tokenizer.json"); tok.no_padding(); tok.no_truncation()
ref = BertModel.from_pretrained(HF, add_pooling_layer=False).eval(); d = load_file(f"{HF}/2_Dense/model.safetensors")
texts = ["Represent this sentence for searching relevant passages: which experiment measured Haiku decider accuracy",
         "The packed 68M model scored 0.772 on Typed Decisions, matching fp32. LoRA rank 8 spikes add 1.2M trainable parameters.",
         "short", "## Results\n\n| arm | hit@10 |\n|---|---|\n| Ettin-17m int4 | 0.918 |\n| leaf-ir int4 | 0.942 |\n" * 6]
def reference(t):
    ids = tok.encode(t).ids[:512]
    with torch.no_grad(): h = ref(input_ids=torch.tensor([ids])).last_hidden_state[0]
    v = h.mean(0) @ d["linear.weight"].T + d["linear.bias"]
    return ids, (v / v.norm()).numpy()
for S in [int(s) for s in sys.argv[1:]]:
    net, cfg = load(S, HF)
    worst = 1.0
    for t in texts:
        ids, r = reference(t); ids = ids[:S]
        if len(ids) < len(tok.encode(t).ids): _, r = None, None
        x = torch.zeros(1, S, dtype=torch.long); x[0, :len(ids)] = torch.tensor(ids); m = torch.zeros(1, S); m[0, :len(ids)] = 1
        with torch.no_grad(): v = net(x, m)[0].numpy()
        if r is not None: worst = min(worst, float(v @ r))
    print(f"S={S}: torch rewrite vs HF reference, worst cosine {worst:.6f}")
    tr = torch.jit.trace(net, (torch.zeros(1, S, dtype=torch.int32), torch.ones(1, S)))
    ml = ct.convert(tr, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16,
                    inputs=[ct.TensorType("input_ids", shape=(1, S), dtype=np.int32), ct.TensorType("pad_mask", shape=(1, S), dtype=np.float16)],
                    outputs=[ct.TensorType("embedding")])
    ml.save(f"model/leaf_S{S}.mlpackage")
    for name, cu in (("CPU", ct.ComputeUnit.CPU_ONLY), ("Neural Engine", ct.ComputeUnit.CPU_AND_NE)):
        mm = ct.models.MLModel(f"model/leaf_S{S}.mlpackage", compute_units=cu); worst = 1.0; ts = []
        for rep in range(40):
            for t in texts:
                ids, r = reference(t)
                if len(ids) > S: continue
                x = np.zeros((1, S), np.int32); x[0, :len(ids)] = ids; m = np.zeros((1, S), np.float16); m[0, :len(ids)] = 1
                t0 = time.perf_counter(); v = np.asarray(mm.predict({"input_ids": x, "pad_mask": m})["embedding"], np.float32)[0]; ts.append(time.perf_counter() - t0)
                worst = min(worst, float(v @ r))
        print(f"   Core ML {name:14s} median {np.median(ts[8:]) * 1000:.2f} ms, worst cosine vs reference {worst:.5f}")
