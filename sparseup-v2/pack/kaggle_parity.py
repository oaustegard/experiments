"""Build kaggle/job_parity_<name>.py: the browser parity check (check.py + web/check.html) on a Kaggle T4 box.

    python3 kaggle_parity.py --name smoke --mode smoke   # 3 int8 texts
    python3 kaggle_parity.py --name full --mode full
    python3 <claude-workspace>/scripts/kaggle_job.py push sparseup-parity-full kaggle/job_parity_full.py --accel gpu --internet

The job rebuilds the needed slice of the experiments repo from embedded files, downloads the model,
builds both packs, registers the NVIDIA Vulkan driver, and runs web/engine.js under Node on Dawn (run_node.mjs)
on a Tesla T4: int8 and q4 vectors for every fixture text, scored by check.py --source t4, plus T4 timing.
--mode q4swift instead runs check.py on SwiftShader for q4 (Kaggle grants no Vulkan; the Node path finds no driver).
Outputs land in /kaggle/working/out/.
"""
import argparse
import base64
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
FILES = ["_lib/wgsl/runtime.js", "_lib/wgsl/kernels.js", "embeddinggemma2-webgpu-tail/run/run_demo.py",
         "sparseup-v2/pack/pack.py", "sparseup-v2/pack/check.py", "sparseup-v2/pack/run_node.mjs", "sparseup-v2/pack/web/engine.js",
         "sparseup-v2/pack/web/sparseup.js", "sparseup-v2/pack/web/check.html", "sparseup-v2/pack/web/build.json",
         "sparseup-v2/pack/web/fixtures/parity.json"]
BINARY = ["sparseup-v2/pack/cache/parity_ref.npz"]

RUNNER = r'''
import base64, json, os, subprocess, sys
from pathlib import Path
R = Path("/kaggle/working/repo"); P = R / "sparseup-v2" / "pack"; OUT = Path("/kaggle/working/out")
OUT.mkdir(parents=True, exist_ok=True)
for rel, src in TEXT.items():
    (R / rel).parent.mkdir(parents=True, exist_ok=True); (R / rel).write_text(src)
for rel, b in BIN.items():
    (R / rel).parent.mkdir(parents=True, exist_ok=True); (R / rel).write_bytes(base64.b64decode(b))
def sh(cmd, **kw):
    print("$", cmd, flush=True); return subprocess.run(cmd, shell=True, text=True, **kw)
sh("nvidia-smi --query-gpu=name,driver_version --format=csv")
sh("pip install -q playwright 2>&1 | tail -1")   # check.py imports it
sh("ls /usr/share/vulkan/icd.d /etc/vulkan/icd.d 2>&1; ldconfig -p | grep -i -E 'GLX_nvidia|nvidia-glvk|libvulkan' | head")
sh("echo NVIDIA_DRIVER_CAPABILITIES=$NVIDIA_DRIVER_CAPABILITIES; ls /usr/local/nvidia/lib64 2>&1 | head -40")
libs = sorted(str(x) for d in ("/usr/local/nvidia", "/usr/lib/x86_64-linux-gnu", "/usr/lib64", "/usr/local/lib")
              if Path(d).exists() for x in Path(d).rglob("libGLX_nvidia.so*"))
print("GLX_nvidia candidates", libs, flush=True)
icd = Path("/usr/share/vulkan/icd.d/nvidia_icd.json")
if libs:
    icd.parent.mkdir(parents=True, exist_ok=True)
    icd.write_text(json.dumps({"file_format_version": "1.0.0", "ICD": {"library_path": libs[0], "api_version": "1.3"}}))
    os.environ["VK_ICD_FILENAMES"] = str(icd)
    print("wrote", icd, libs[0], flush=True)
from huggingface_hub import hf_hub_download
for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"]:
    hf_hub_download("Linkup-Platform/linkup-sparseup-embed-v2", f, local_dir=P / "hf")
for q in ("int8", "q4"):
    sh(f"{sys.executable} pack.py --hf hf --out packs/{q} --quant q4".replace("--quant q4", f"--quant {q}"), cwd=P, check=True)

if MODE == "q4swift":
    # Kaggle containers get compute,utility driver capabilities only (no Vulkan), so WebGPU runs on SwiftShader.
    sh("python3 -m playwright install --with-deps chromium 2>&1 | tail -2")
    env = dict(os.environ, CHROME_PATH="playwright")
    for start, n in ((0, 28), (28, 7)):
        sh(f"{sys.executable} -u check.py --pack q4 --start {start} --only {n} --no-timing", cwd=P, env=env)
    for f in sorted((P / "cache" / "wgsl_q4").glob("*.json")):
        (OUT / f"wgsl_q4_chunk_{f.name}").write_bytes(f.read_bytes())
    for f in (P / "results" / "parity_q4.json", P / "results" / "wgsl_q4_swiftshader.json"):
        if f.exists():
            (OUT / f.name).write_bytes(f.read_bytes())
else:
    sh("cd /tmp && curl -sSL https://nodejs.org/dist/v22.11.0/node-v22.11.0-linux-x64.tar.xz | tar xJ && ln -sf /tmp/node-v22.11.0-linux-x64/bin/* /usr/local/bin/ && node --version")
    sh("npm init -y >/dev/null && npm i webgpu@0.3.8 2>&1 | tail -2", cwd=P)
    sh("vulkaninfo --summary 2>&1 | head -20 || true")
    def run(pack, extra, tag):
        out = OUT / f"wgsl_{pack}_t4{tag}.json"
        r = sh(f"node run_node.mjs packs/{pack} web/fixtures/parity.json {out} {extra}", cwd=P)
        if out.exists():
            d = P / "cache" / f"wgsl_{pack}_t4"; d.mkdir(parents=True, exist_ok=True)
            (d / "000.json").write_bytes(out.read_bytes())
            sh(f"{sys.executable} check.py --pack {pack} --source t4", cwd=P)
            f = P / "results" / f"parity_{pack}_t4.json"
            if f.exists():
                (OUT / f.name).write_bytes(f.read_bytes())
        return r.returncode
    if MODE == "smoke":
        run("int8", "--only 3", "_smoke")
    else:
        for pack in ("int8", "q4"):
            run(pack, "--reps 20", "")
print("done", flush=True)
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="full")
    ap.add_argument("--mode", choices=["smoke", "full", "q4swift"], default="full")
    a = ap.parse_args()
    text = {f: (REPO / f).read_text() for f in FILES}
    binary = {f: base64.b64encode((REPO / f).read_bytes()).decode() for f in BINARY}
    out = HERE / "kaggle" / f"job_parity_{a.name}.py"
    out.parent.mkdir(exist_ok=True)
    out.write_text(f"TEXT = {text!r}\nBIN = {binary!r}\nMODE = {a.mode!r}\n" + RUNNER)
    print(out, out.stat().st_size)


if __name__ == "__main__":
    main()
