import os, re, json, hashlib, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.expanduser(os.environ.get("PSEARCH_ROOT", "~/Projects"))
IDX = os.path.join(HERE, "index")
S_DOC, S_QUERY = 256, 32
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
SKIP_DIRS = {"node_modules", ".git", ".venv", "venv", "site-packages", "build", "dist", "DerivedData", ".build", "Pods", "__pycache__",
             ".next", "target", "vendor", ".claude", "data", "out", "results", "prompts", "Core2-for-AWS-IoT-EduKit", "plugins", "env", "share", "suites", "fixtures"}
SKIP_FILES = re.compile(r"^(CMakeLists\.txt|LICENSE.*|_info\.txt|mail-log\.txt|requirements.*\.txt|install\.log)$", re.I)
EXTS = (".md", ".txt", ".rst")

class Embedder:
    """Core ML leaf-ir on the Neural Engine for documents (256 tokens), CPU for short queries (32 tokens)."""
    def __init__(self, docs=True):
        import coremltools as ct
        from tokenizers import Tokenizer
        self.tok = Tokenizer.from_file(os.path.join(HERE, "model", "tokenizer.json")); self.tok.no_padding(); self.tok.no_truncation()
        self.q = ct.models.MLModel(os.path.join(HERE, "model", f"leaf_S{S_QUERY}.mlpackage"), compute_units=ct.ComputeUnit.CPU_ONLY)
        self.d = ct.models.MLModel(os.path.join(HERE, "model", f"leaf_S{S_DOC}.mlpackage"), compute_units=ct.ComputeUnit.CPU_AND_NE) if docs else None
    def _run(self, model, ids, S):
        ids = ids[:S - 1] + [102] if len(ids) > S else ids          # keep [SEP] when truncating
        x = np.zeros((1, S), np.int32); x[0, :len(ids)] = ids; m = np.zeros((1, S), np.float16); m[0, :len(ids)] = 1
        v = np.asarray(model.predict({"input_ids": x, "pad_mask": m})["embedding"], np.float32)[0]
        return v / np.linalg.norm(v)          # the CPU path of the 32-token model returns the vector unnormalised
    def doc_ids(self, ids): return self._run(self.d, ids, S_DOC)
    def query(self, text):
        ids = self.tok.encode(QUERY_PREFIX + text).ids
        return self._run(self.q, ids, S_QUERY) if len(ids) <= S_QUERY or self.d is None else self._run(self.d, ids, S_DOC)

WORD = re.compile(r"[a-z0-9]+(?:[._-][a-z0-9]+)*")
def words(t):
    out = []
    for w in WORD.findall(t.lower()):
        out.append(w)
        if any(c in w for c in "._-"): out.extend(p for p in re.split(r"[._-]", w) if p)
    return out

def prose(text):
    """True for running text; False for data dumps (number tables, id lists, vectors saved as .txt)."""
    sample = text[:20000]; toks = sample.split()
    if len(toks) < 30: return False
    alpha = sum(t.isalpha() and len(t) > 1 for t in (x.strip(".,;:()[]*_`'\"") for x in toks)) / len(toks)
    return alpha > 0.5
