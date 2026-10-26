# Colab GPU driver: encode NanoBEIR with the 115M model using run/mcb.py (uploaded alongside).
import os, subprocess, sys, time
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "tokenizers", "safetensors", "datasets"], check=True)
import torch
torch.backends.cuda.matmul.allow_tf32 = False
print(torch.cuda.get_device_name(0), flush=True)
root = "/content/models"
for m in sys.argv[1:]:
    os.makedirs(f"{root}/{m}/1_Dense", exist_ok=True)
    b = f"https://huggingface.co/VAGOsolutions/SauerkrautLM-MoEdernColBERT-{m}/resolve/main"
    for f in ["config.json", "model.safetensors", "tokenizer.json", "1_Dense/model.safetensors"]:
        subprocess.run(["curl", "-sL", "-o", f"{root}/{m}/{f}", f"{b}/{f}"], check=True)
os.environ.update(MCB_DEVICE="cuda", MCB_ROOT=root, MCB_ENC="/content/enc")
t0 = time.time()
subprocess.run([sys.executable, "/content/encode.py", *sys.argv[1:]], check=True)
print("total", time.time() - t0, flush=True)
subprocess.run("cd /content && tar czf enc.tgz enc", shell=True, check=True)
