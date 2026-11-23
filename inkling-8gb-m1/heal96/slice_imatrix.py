# slice a llama.cpp GGUF imatrix to the kept experts per layer (expert dim is the last ggml dim of *_exps entries)
import json, sys, numpy as np
from gguf import GGUFReader, GGUFWriter, GGUFValueType, GGMLQuantizationType
src, keep_path, out = sys.argv[1:4]
keep = {int(k): v for k, v in json.load(open(keep_path)).items()}
r = GGUFReader(src)
w = GGUFWriter(out, arch=None) if False else GGUFWriter(out, "imatrix")
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = f.types[0]; sub = f.types[-1] if vt == GGUFValueType.ARRAY else None
    w.add_key_value(name, f.contents(), vt, sub_type=sub)
n = 0
for t in r.tensors:
    a = np.array(t.data, dtype=np.float32).reshape([int(x) for x in reversed(list(t.shape))])
    if "_exps." in t.name:
        li = int(t.name.split(".")[1])
        a = np.ascontiguousarray(a[keep[li]]); n += 1
    w.add_tensor(t.name, a, raw_dtype=GGMLQuantizationType.F32)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_tensors_to_file(); w.close()
print("sliced", n, "expert entries ->", out)
