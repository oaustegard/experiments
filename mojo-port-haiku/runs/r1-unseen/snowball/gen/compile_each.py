"""Harness wrapper (not agent code): translate and compile each language alone."""
import json, subprocess, sys, pathlib, re
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import translate as T
rep = json.loads((pathlib.Path(__file__).parents[1] / "translate_report.json").read_text())
tmp = pathlib.Path("/tmp/sbl"); tmp.mkdir(exist_ok=True)
for lang in rep["ok"]:
    T.LANGS = [lang]; T.OUT = tmp / f"{lang}.mojo"; T.main()
def build(lang):
    p = subprocess.run(["mojo", "build", str(tmp / f"{lang}.mojo"), "--emit", "shared-lib", "-o", str(tmp / f"{lang}.so")], capture_output=True, text=True)
    errs = [l for l in (p.stdout + p.stderr).splitlines() if ": error:" in l]
    return lang, p.returncode, [re.sub(r"^.*?error: ", "", e) for e in errs][:3]
with ThreadPoolExecutor(3) as ex:
    res = list(ex.map(build, rep["ok"]))
rep["compiled"] = [l for l, rc, _ in res if rc == 0]
rep["compile_errors"] = {l: e for l, rc, e in res if rc != 0}
(pathlib.Path(__file__).parents[1] / "translate_report.json").write_text(json.dumps(rep, indent=1))
print(len(rep["compiled"]), "compiled:", rep["compiled"]); print(json.dumps(rep["compile_errors"], indent=1))
