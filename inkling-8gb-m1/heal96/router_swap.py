# copy a GGUF, replacing only ffn_gate_inp rows of the kept experts with healed router weights (checkpoint + delta)
import json, sys, re, numpy as np, torch
from safetensors import safe_open
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
src, keep_path, ckpt, out = sys.argv[1:5]
keep = {int(k): v for k, v in json.load(open(keep_path)).items()}
ad = torch.load(ckpt, map_location="cpu", weights_only=False)["adapters"]
f = safe_open("/content/ink/model-00001-of-00001.safetensors", framework="pt", device="cpu")
r = GGUFReader(src)
w = GGUFWriter(out, r.fields["general.architecture"].contents())
for name, fl in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = fl.types[0]; sub = fl.types[-1] if vt == GGUFValueType.ARRAY else None
    w.add_key_value(name, fl.contents(), vt, sub_type=sub)
n = 0
for t in r.tensors:
    shape = [int(x) for x in t.shape]
    m = re.match(r"blk\.(\d+)\.ffn_gate_inp\.weight$", t.name)
    if m:
        li = int(m.group(1))
        rw = f.get_tensor(f"model.llm.layers.{li}.mlp.gate.weight").float()
        dk = [k for k in ad if re.search(rf"layers\.{li}\.mlp\.gate\.delta$", k)]
        if dk:
            rw = rw + ad[dk[0]].float()
        old = np.asarray(t.data, dtype=np.float32)
        new = rw[torch.tensor(sorted(keep[li]))].numpy().astype(np.float32)
        assert new.shape == old.shape, (new.shape, old.shape)
        w.add_tensor(t.name, np.ascontiguousarray(new), raw_dtype=GGMLQuantizationType.F32); n += 1
        continue
    raw = np.asarray(t.data).reshape(-1).view(np.uint8)
    w.add_tensor(t.name, raw.view(np.int8), raw_shape=list(reversed(shape)), raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("router rows replaced in", n, "layers ->", out)
