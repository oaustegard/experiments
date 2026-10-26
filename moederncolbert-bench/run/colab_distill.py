# Colab GPU driver for distill.py: fetch 1.5M and 115M, encode NanoBEIR with 115M, then train.
# Upload mcb.py, encode.py, evaluate.py, distill.py to /content first. Args pass through to distill.py.
import os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "safetensors", "datasets"], check=True)
import torch
torch.backends.cuda.matmul.allow_tf32 = False
print(torch.cuda.get_device_name(0), flush=True)
root = "/content/models"
for m in ["1.5M", "115M"]:
    os.makedirs(f"{root}/{m}/1_Dense", exist_ok=True)
    b = f"https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-{m}/resolve/main"
    for f in ["config.json", "model.safetensors", "tokenizer.json", "1_Dense/model.safetensors"]:
        subprocess.run(["curl", "-sL", "-o", f"{root}/{m}/{f}", f"{b}/{f}"], check=True)
os.environ.update(MCB_DEVICE="cuda", MCB_ROOT=root, MCB_ENC="/content/enc", MCB_OUT="/content/distill", MCB_RES="/content/res")
t0 = time.time()
subprocess.run([sys.executable, "/content/encode.py", "115M"], check=True)
print("encode", time.time() - t0, flush=True)
subprocess.run([sys.executable, "/content/distill.py", *sys.argv[1:]], check=True)
print("total", time.time() - t0, flush=True)
