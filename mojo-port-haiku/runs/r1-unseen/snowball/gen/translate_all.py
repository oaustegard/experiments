"""Harness wrapper (not agent code): run the agent's translator per language,
keep the languages it translates, record the ones it rejects."""
import json, sys, traceback, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent))
import translate as T
ok, failed = [], {}
for lang in T.LANGS:
    try:
        lg = T.Language(lang)
        lg.struct()
        ok.append(lang)
    except Exception as e:
        failed[lang] = f"{type(e).__name__}: {traceback.format_exc().strip().splitlines()[-3].strip()[:160]}"
T.LANGS = ok
T.main()
s = pathlib.Path(__file__).parents[1] / "msnowball/__init__.py"
src = s.read_text()
import re
src = re.sub(r"_LANGS = \[.*?\]", "_LANGS = %r" % ok, src)
s.write_text(src)
pathlib.Path(__file__).parents[1].joinpath("translate_report.json").write_text(json.dumps({"ok": ok, "failed": failed}, indent=1))
print(len(ok), "ok;", len(failed), "rejected:", json.dumps(failed, indent=1))
