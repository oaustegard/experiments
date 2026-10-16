"""Every corpus document, encoded by codec_v2.py, mangled the ways chat and mail
mangle links, decodes to the same text in Python and in Node (codec_v2.mjs +
node:zlib's raw inflate with the dictionary). Exit 1 on any mismatch."""
import json, subprocess, sys
from pathlib import Path
from urllib.parse import quote
import codec_v2 as c

HERE = Path(__file__).parent
docs = [d["text"] for k in ("train", "test") for d in json.loads((HERE / "data/corpus.json").read_text())[k]]
docs.append((HERE / "data/deck.tlo").read_text())

def mangles(p):
    yield "clean", p
    yield "percent-encoded twice (Webex)", quote(quote(p, safe=""), safe="")
    yield "curly quote glued on", p + "’"
    yield "wrapped at 76 by a mail client", "\n".join(p[i:i + 76] for i in range(0, len(p), 76))
    yield "zero-width spaces", p[:40] + "​" + p[40:]

cases, bad = [], 0
for i, doc in enumerate(docs):
    for qr in (False, True):
        p = c.encode(doc, qr=qr)
        assert all(ch.isalnum() or ch == "+" for ch in p), p
        for how, m in mangles(p):
            try:
                ok = c.decode(m) == doc
            except ValueError:
                ok = False
            if not ok:
                bad += 1
                print(f"python FAIL doc {i} qr={qr}: {how}")
            if not qr:
                cases.append({"doc": i, "how": how, "payload": m, "text": doc})
out = subprocess.run(["node", str(HERE / "test_node.mjs")], input=json.dumps(cases), capture_output=True, text=True)
print(out.stdout.strip() or out.stderr.strip())
total = len(docs) * 2 * 5
print(f"python: {total - bad}/{total} decodes correct")
# Why there is no trimming of glued letters: count wrong-but-clean decodes when one
# character is cut from a valid body (raw deflate has no checksum to refuse them).
wrong = 0
for doc in docs:
    p = c.encode(doc).replace("+", "")
    try:
        wrong += c.decode(p[:-1]) != doc
    except ValueError:
        pass
print(f"one character cut from the end: {wrong}/{len(docs)} decode to a wrong document without error")
sys.exit(1 if bad or out.returncode else 0)
