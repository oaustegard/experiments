"""Strip the masked-LM head from a shipped encoder ONNX export so latency reflects the encoder a classifier uses.
Usage: python3 cut_heads.py model.onnx  -> writes enc_model.onnx next to it."""
import onnx, os, sys
from onnx import helper, TensorProto
for f in sys.argv[1:]:
    m = onnx.load(f, load_external_data=True); g = m.graph
    prod = {o: n for n in g.node for o in n.output}
    head = [n for n in g.node if n.name.startswith(("/head/", "/model/lm_head", "/lm_head", "/decoder"))]
    if not head: print(f, "no head nodes found; skipped"); continue
    qn = prod[head[0].input[0]]            # QuantizeLinear feeding the head
    enc_out = qn.input[0]                  # float encoder output
    drop = {id(n) for n in head} | {id(qn)}
    keep = [n for n in g.node if id(n) not in drop]
    needed, changed = {enc_out}, True
    while changed:
        changed = False
        for n in keep:
            if any(o in needed for o in n.output):
                for i in n.input:
                    if i not in needed: needed.add(i); changed = True
    keep = [n for n in keep if any(o in needed for o in n.output)]
    del g.node[:]; g.node.extend(keep)
    del g.output[:]; g.output.extend([helper.make_tensor_value_info(enc_out, TensorProto.FLOAT, ["b", "s", "h"])])
    init = [t for t in g.initializer if t.name in needed]; del g.initializer[:]; g.initializer.extend(init)
    out = os.path.join(os.path.dirname(f), "enc_" + os.path.basename(f)); onnx.save(m, out)
    print(f, "->", out, "output", enc_out, "nodes", len(keep), "MB", round(os.path.getsize(out) / 1e6, 1))
