import sys, json, time, numpy as np, torch, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, 'src')
import coremltools as ct
from dettin import Dettin
from ane_model import DettinANE
S = int(sys.argv[1])
m = Dettin.load('pack', device='cpu')
net = DettinANE(m.enc, m.head, S).eval()
ids = torch.zeros(1, S, dtype=torch.int32); pm = torch.ones(1, S)
tr = torch.jit.trace(net, (ids, pm))
t = time.time()
ml = ct.convert(tr, convert_to="mlprogram", minimum_deployment_target=ct.target.macOS14, compute_precision=ct.precision.FLOAT16,
                inputs=[ct.TensorType("input_ids", shape=(1, S), dtype=np.int32), ct.TensorType("pad_mask", shape=(1, S), dtype=np.float16)],
                outputs=[ct.TensorType("logits")])
ml.save(f"dettin17m_S{S}.mlpackage")
print(f"converted S={S} in {time.time() - t:.0f}s")
