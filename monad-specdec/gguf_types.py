"""Which precision does each official GGUF keep for each tensor?

Baguettotron sets tie_word_embeddings, so there is no separate output.weight —
`token_embd.weight` serves as both the embedding and the LM head. That single
tensor is what dominates an EAGLE draft step, which makes its quantization level
the thing that decides whether the cost ratio survives quantization.
"""
import json, struct, sys

GGML_TYPES = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 6: "Q5_0", 7: "Q5_1",
              8: "Q8_0", 9: "Q8_1", 10: "Q2_K", 11: "Q3_K", 12: "Q4_K",
              13: "Q5_K", 14: "Q6_K", 15: "Q8_K", 16: "IQ2_XXS", 17: "IQ2_XS",
              18: "IQ3_XXS", 19: "IQ1_S", 20: "IQ4_NL", 21: "IQ3_S",
              22: "IQ2_S", 23: "IQ4_XS", 24: "I8", 25: "I16", 26: "I32",
              27: "I64", 28: "F64", 29: "IQ1_M", 30: "BF16"}
SIMPLE = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f",
          7: "<?", 10: "<Q", 11: "<q", 12: "<d"}


def rd(f, fmt):
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def rstr(f):
    (n,) = rd(f, "<Q")
    return f.read(n).decode("utf-8", "replace")


def skip_val(f, t):
    if t in SIMPLE:
        rd(f, SIMPLE[t])
    elif t == 8:
        rstr(f)
    elif t == 9:
        (et,) = rd(f, "<I")
        (n,) = rd(f, "<Q")
        for _ in range(n):
            skip_val(f, et)
    else:
        raise ValueError(f"unknown kv type {t}")


def read_types(path):
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError("not a GGUF file")
        _ver, n_tensors, n_kv = rd(f, "<IQQ")
        for _ in range(n_kv):
            rstr(f)
            (t,) = rd(f, "<I")
            skip_val(f, t)
        out = {}
        for _ in range(n_tensors):
            name = rstr(f)
            (nd,) = rd(f, "<I")
            dims = rd(f, "<" + "Q" * nd)
            (tt,) = rd(f, "<I")
            rd(f, "<Q")
            out[name] = (GGML_TYPES.get(tt, str(tt)), list(dims))
    return out


if __name__ == "__main__":
    quants = sys.argv[1:] or ["Q8_0", "Q4_K_M", "Q4_0"]
    report = {}
    for q in quants:
        types = read_types(f"/tmp/specdec/Bag-{q}.gguf")
        hist = {}
        for t, _ in types.values():
            hist[t] = hist.get(t, 0) + 1
        report[q] = {
            "n_tensors": len(types),
            "type_histogram": hist,
            "token_embd_weight": types["token_embd.weight"][0],
            "has_separate_output_weight": "output.weight" in types,
            "sample_layer_tensor": types["blk.0.ffn_up.weight"][0],
        }
        print(q, json.dumps(report[q]), flush=True)
    report["note"] = ("token_embd.weight is the LM head under tied embeddings. "
                      "It stays Q8_0 in every 4-bit build, while layer tensors "
                      "drop to 4-bit.")
    json.dump(report, open("gguf_types.json", "w"), indent=2)
