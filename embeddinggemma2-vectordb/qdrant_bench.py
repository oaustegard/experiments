#!/usr/bin/env python3
"""Qdrant sweeps for the EmbeddingGemma 2 vector-DB quantization test (part 3).

Base: first N shards of the Quora corpus (fp32 768d, normalized).
Truth: exact dot-product top-10 on fp32 d=768 in numpy. Reduced-dim
collections (d=128, d=512) are scored against the same d=768 truth.

Collections (Dot, HNSW m=32, ef_construct=200):
  fp32_d768   no quantization
  fp32_d128   first 128 dims, renormalized
  int8_d768   scalar int8, always_ram quantized, originals on_disk
  bq2_d768    binary two_bits, query_encoding scalar8bits
  bq2_d512    as bq2_d768, first 512 dims renormalized
  bq1_d768    binary one_bit (default query encoding)
Quantized collections are searched with rescore off and on (oversampling 4).

Per point: recall@10, QPS (sequential single client, best of 3 runs),
p50/p99 latency (ms) from the best run.

RAM-cap mode (--ram-cap-mb): cgroup v1 memory child, Qdrant started inside
it with memory.limit_in_bytes set. Runs fp32 d=768 (originals on_disk) and
bq2_d768 (rescore off/on). Also records whether page cache is charged.
Results of that mode are merged into the output JSON under "ram_cap".

Usage:
  python qdrant_bench.py --smoke                  # 2 shards, 100 held-out queries
  python qdrant_bench.py                          # full sweep (run on an idle box)
  python qdrant_bench.py --ram-cap-mb 0           # RAM-cap mode, auto cap (0 = auto)
  python qdrant_bench.py --ram-cap-mb 120         # RAM-cap mode, explicit cap in MB
"""
import argparse, atexit, datetime, json, math, os, shutil, signal, socket, subprocess, sys, time, urllib.request
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient, models as qm

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
QD_BIN = "/tmp/qd/n/qdrant"
CG_ROOT = Path("/sys/fs/cgroup/memory")
DIM = 768
K = 10
EFS = [16, 32, 64, 128, 256, 512]
REPS = 3
WARMUP = 20
OVERSAMPLE = 4.0
HNSW = qm.HnswConfigDiff(m=32, ef_construct=200)
FP32_MB = None  # set in prepare()

CONFIGS = [
    dict(name="fp32_d768", dim=768, quant=None, rescore=[None]),
    dict(name="fp32_d128", dim=128, quant=None, rescore=[None]),
    dict(name="fp32_d256", dim=256, quant=None, rescore=[None]),
    dict(name="fp32_d512", dim=512, quant=None, rescore=[None]),
    dict(name="int8_d768", dim=768, quant="int8", rescore=[False, True]),
    dict(name="bq2_d768", dim=768, quant="bq2", rescore=[False, True]),
    dict(name="bq2_d512", dim=512, quant="bq2", rescore=[False, True]),
    dict(name="bq1_d768", dim=768, quant="bq1", rescore=[False, True]),
]

_servers = []


def log(msg):
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def stop_all():
    for s in list(_servers):
        s.stop()


atexit.register(stop_all)
signal.signal(signal.SIGTERM, lambda *a: sys.exit(1))


def in_cgroup_hook(cg_name):
    """preexec hook: move the child into a memory cgroup before exec."""
    procs = CG_ROOT / cg_name / "cgroup.procs"

    def hook():
        os.setsid()
        with open(procs, "w") as f:
            f.write(str(os.getpid()))
    return hook


class Server:
    """A qdrant process we started. Always stopped by stop() or atexit."""

    def __init__(self, storage, tag, cgroup=None):
        self.storage = Path(storage)
        self.tag = tag
        self.cgroup = cgroup
        self.http = free_port()
        self.grpc = free_port()
        while self.grpc == self.http:
            self.grpc = free_port()
        self.proc = None

    def start(self):
        self.storage.mkdir(parents=True, exist_ok=True)
        (DATA / "logs").mkdir(parents=True, exist_ok=True)
        logf = open(DATA / "logs" / f"qdrant_{self.tag}.log", "a")
        env = dict(os.environ,
                   QDRANT__STORAGE__STORAGE_PATH=str(self.storage),
                   QDRANT__SERVICE__HOST="127.0.0.1",
                   QDRANT__SERVICE__HTTP_PORT=str(self.http),
                   QDRANT__SERVICE__GRPC_PORT=str(self.grpc),
                   QDRANT__TELEMETRY_DISABLED="true",
                   QDRANT__LOG_LEVEL="WARN")
        pre = in_cgroup_hook(self.cgroup) if self.cgroup else os.setsid
        self.proc = subprocess.Popen([QD_BIN], env=env, cwd=str(DATA), stdout=logf,
                                     stderr=subprocess.STDOUT, preexec_fn=pre)
        _servers.append(self)
        self.wait_ready()

    def wait_ready(self, timeout=120):
        url = f"http://127.0.0.1:{self.http}/readyz"
        t0 = time.time()
        while time.time() - t0 < timeout:
            if self.proc.poll() is not None:
                raise RuntimeError(f"qdrant exited rc={self.proc.returncode}; see data/logs/qdrant_{self.tag}.log")
            try:
                with urllib.request.urlopen(url, timeout=2) as r:
                    if r.status == 200:
                        return
            except Exception:
                pass
            time.sleep(0.5)
        raise RuntimeError("qdrant did not become ready")

    def client(self):
        return QdrantClient(host="127.0.0.1", port=self.http, grpc_port=self.grpc,
                            prefer_grpc=True, timeout=600)

    def rss_mb(self):
        return self.rss_detail()["rss"]

    def rss_detail(self):
        """VmRSS split into anonymous (not evictable) and file-backed (mmap, evictable) MB."""
        out = {}
        for line in open(f"/proc/{self.proc.pid}/status"):
            for key, name in (("VmRSS:", "rss"), ("RssAnon:", "anon"), ("RssFile:", "file")):
                if line.startswith(key):
                    out[name] = round(int(line.split()[1]) / 1024, 1)
        return out

    def stop(self):
        if self.proc is not None and self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(60)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        if self in _servers:
            _servers.remove(self)


def qdrant_version():
    out = subprocess.run([QD_BIN, "--version"], capture_output=True, text=True).stdout.strip()
    return out.split()[-1] if out else None


def load_data(shards, heldout_nq, nq):
    """Return base (N,768) fp32, queries (nq,768) fp32, exact top-K truth (nq,K)."""
    parts = []
    for i in range(shards):
        f = DATA / "quora" / "shards" / f"docs_{i:05d}.npy"
        if not f.exists():
            raise FileNotFoundError(f"missing shard {f} (still encoding?)")
        parts.append(np.load(f).astype(np.float32))
    base = np.concatenate(parts)
    del parts
    assert base.shape[1] == DIM, base.shape
    if heldout_nq:
        Q = base[-nq:].copy()
        base = base[:-nq].copy()
    else:
        qf = DATA / "quora" / "q_dense.npy"
        if not qf.exists():
            raise FileNotFoundError(f"missing {qf} (query encode not finished)")
        Q = np.load(qf).astype(np.float32)[:nq]
        if len(Q) < nq:
            raise ValueError(f"only {len(Q)} queries available, need {nq}")
    truth = exact_top_k(base, Q)
    return base, Q, truth


def exact_top_k(base, Q, chunk=100):
    out = np.empty((len(Q), K), dtype=np.int64)
    for i in range(0, len(Q), chunk):
        S = Q[i:i + chunk] @ base.T
        idx = np.argpartition(-S, K, axis=1)[:, :K]
        order = np.argsort(-np.take_along_axis(S, idx, 1), axis=1)
        out[i:i + chunk] = np.take_along_axis(idx, order, 1)
    return out


def truncate(X, d):
    if d == X.shape[1]:
        return np.ascontiguousarray(X)
    Y = X[:, :d].astype(np.float32)
    Y /= np.linalg.norm(Y, axis=1, keepdims=True)
    return np.ascontiguousarray(Y)


def quant_config(kind):
    if kind is None:
        return None
    if kind == "int8":
        return qm.ScalarQuantization(scalar=qm.ScalarQuantizationConfig(
            type=qm.ScalarType.INT8, always_ram=True))
    if kind == "bq2":
        return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(
            always_ram=True, encoding=qm.BinaryQuantizationEncoding.TWO_BITS,
            query_encoding=qm.BinaryQuantizationQueryEncoding.SCALAR8BITS))
    if kind == "bq1":
        return qm.BinaryQuantization(binary=qm.BinaryQuantizationConfig(
            always_ram=True, encoding=qm.BinaryQuantizationEncoding.ONE_BIT))
    raise ValueError(kind)


def build(client, name, X, quant, disk_vectors=False, timeout=3600):
    """Create, upload and wait for the index. Returns build record."""
    if client.collection_exists(name):
        client.delete_collection(name)
    on_disk = quant is not None or disk_vectors
    client.create_collection(
        name,
        vectors_config=qm.VectorParams(size=X.shape[1], distance=qm.Distance.DOT, on_disk=on_disk),
        hnsw_config=HNSW,
        # default threshold (20000 KB) leaves small-dim segments unindexed (brute force);
        # 1000 KB makes every collection HNSW-indexed regardless of dimension
        optimizers_config=qm.OptimizersConfigDiff(indexing_threshold=1000),
        quantization_config=quant_config(quant),
    )
    n = len(X)
    t0 = time.time()
    B = 2000
    for lo in range(0, n, B):
        hi = min(n, lo + B)
        client.upsert(name, points=qm.Batch(ids=list(range(lo, hi)), vectors=X[lo:hi].tolist()), wait=True)
    upload_s = time.time() - t0
    while True:
        info = client.get_collection(name)
        if info.status == qm.CollectionStatus.GREEN and info.points_count == n:
            break
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{name}: index not green after {timeout}s")
        time.sleep(2)
    indexed = info.indexed_vectors_count or 0
    if indexed < 0.99 * n:
        log(f"  WARNING {name}: only {indexed}/{n} vectors in HNSW index")
    rec = dict(dim=X.shape[1], quant=quant or "none", on_disk_vectors=on_disk, points=n,
               indexed_vectors=indexed, upload_s=round(upload_s, 1), build_total_s=round(time.time() - t0, 1))
    log(f"  built {name}: n={n} d={X.shape[1]} indexed={indexed} total={rec['build_total_s']}s")
    return rec


def recall_at_k(ids_per_query, truth):
    hits = sum(len(set(a) & set(b.tolist())) for a, b in zip(ids_per_query, truth))
    return hits / (len(truth) * K)


def sweep(client, name, Q, truth, quant, rescore, efs, reps=REPS):
    qlist = [q.tolist() for q in Q]
    pts = []
    for ef in efs:
        if quant:
            sp = qm.SearchParams(hnsw_ef=ef, quantization=qm.QuantizationSearchParams(
                rescore=rescore, oversampling=OVERSAMPLE))
        else:
            sp = qm.SearchParams(hnsw_ef=ef)

        def run_once(qs):
            lats, ids = [], []
            t0 = time.perf_counter()
            for q in qs:
                t = time.perf_counter()
                r = client.query_points(name, query=q, limit=K, search_params=sp,
                                        with_payload=False, with_vectors=False)
                lats.append(time.perf_counter() - t)
                ids.append([p.id for p in r.points])
            return time.perf_counter() - t0, lats, ids

        run_once(qlist[:min(WARMUP, len(qlist))])  # warm-up, not timed
        runs = [run_once(qlist) for _ in range(reps)]
        best_total, best_lats, _ = min(runs, key=lambda r: r[0])
        recall = recall_at_k(runs[0][2], truth)
        lat_ms = np.array(best_lats) * 1000
        pt = dict(collection=name, dim=Q.shape[1], quant=quant or "none",
                  rescore=rescore, hnsw_ef=ef, recall_at_10=round(recall, 4),
                  qps=round(len(qlist) / best_total, 1),
                  p50_ms=round(float(np.percentile(lat_ms, 50)), 3),
                  p99_ms=round(float(np.percentile(lat_ms, 99)), 3), nq=len(qlist), reps=reps)
        pts.append(pt)
        log(f"    {name} rescore={rescore} ef={ef}: recall@10={pt['recall_at_10']} "
            f"qps={pt['qps']} p50={pt['p50_ms']} p99={pt['p99_ms']}")
    return pts


def load_doc(path):
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception:
            return {}
    return {}


def save_doc(path, doc):
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2))
    tmp.replace(path)


def run_sweep(args, out):
    base, Q, truth = load_data(args.shards, args.heldout, args.nq)
    efs = args.efs
    doc = load_doc(out)
    doc["meta"] = dict(
        date=datetime.datetime.now().isoformat(timespec="seconds"),
        shards=args.shards, n_base=len(base), nq=len(Q), heldout_queries=args.heldout,
        efs=efs, reps=REPS, warmup=WARMUP, oversample=OVERSAMPLE, k=K,
        hnsw=dict(m=32, ef_construct=200), transport="grpc",
        truth="exact fp32 d=768 dot product, numpy", qdrant_version=qdrant_version(),
        loadavg_at_start=[round(x, 2) for x in os.getloadavg()], smoke=args.smoke,
    )
    doc["builds"], doc["points"] = {}, []
    storage = DATA / "qdrant" / f"sweep_{os.getpid()}"
    srv = Server(storage, tag="sweep")
    srv.start()
    try:
        client = srv.client()
        for cfg in CONFIGS:
            name = cfg["name"]
            log(f"collection {name}")
            X = truncate(base, cfg["dim"])
            Qd = truncate(Q, cfg["dim"])
            rec = build(client, name, X, cfg["quant"])
            rec["rss_after_build_mb"] = srv.rss_mb()
            doc["builds"][name] = rec
            for rescore in cfg["rescore"]:
                doc["points"].extend(sweep(client, name, Qd, truth, cfg["quant"], rescore, efs))
                save_doc(out, doc)
            client.delete_collection(name)
        client.close()
    finally:
        srv.stop()
        shutil.rmtree(storage, ignore_errors=True)
    save_doc(out, doc)
    log(f"wrote {out}")


# ---------- RAM-cap mode ----------

def cg_read(cg, name):
    return int((CG_ROOT / cg / name).read_text().strip())


def cg_stat(cg):
    st = {}
    for line in (CG_ROOT / cg / "memory.stat").read_text().splitlines():
        k, v = line.split()
        st[k] = int(v)
    mb = 2 ** 20
    return dict(usage_mb=round(cg_read(cg, "memory.usage_in_bytes") / mb, 1),
                max_usage_mb=round(cg_read(cg, "memory.max_usage_in_bytes") / mb, 1),
                limit_mb=round(cg_read(cg, "memory.limit_in_bytes") / mb, 1),
                failcnt=cg_read(cg, "memory.failcnt"),
                cache_mb=round(st.get("cache", 0) / mb, 1),
                rss_mb=round(st.get("rss", 0) / mb, 1),
                mapped_file_mb=round(st.get("mapped_file", 0) / mb, 1))


def set_limit(cg, mb):
    (CG_ROOT / cg / "memory.limit_in_bytes").write_text(str(int(mb * 2 ** 20)))


def run_in_cg(cg, argv):
    subprocess.run(argv, preexec_fn=in_cgroup_hook(cg), check=True)


def probe_page_cache(cg, cap_mb=64, file_mb=256):
    """Does page cache count against a v1 memory cgroup here? Write then read a file
    larger than the cap, both inside the cgroup, and see whether usage stays under it."""
    set_limit(cg, cap_mb)
    f = DATA / "pc_probe.bin"
    res = {"cap_mb": cap_mb, "file_mb": file_mb}
    try:
        run_in_cg(cg, ["dd", "if=/dev/zero", f"of={f}", "bs=1M", f"count={file_mb}", "status=none"])
        run_in_cg(cg, ["sync"])
        res["after_write"] = cg_stat(cg)
        run_in_cg(cg, ["sh", "-c", f"cat {f} > /dev/null"])
        res["after_read"] = cg_stat(cg)
        res["page_cache_counted"] = res["after_read"]["cache_mb"] <= cap_mb + 8 and res["after_read"]["failcnt"] > 0
    finally:
        f.unlink(missing_ok=True)
        fe = CG_ROOT / cg / "memory.force_empty"
        if fe.exists():
            fe.write_text("0")  # drop this cgroup's page cache so it does not leak into the next phase
    return res


def run_ram_cap(args, out):
    base, Q, truth = load_data(args.shards, args.heldout, args.nq)
    cg = f"qdbench-{os.getpid()}"
    cg_path = CG_ROOT / cg
    doc = load_doc(out)
    try:
        cg_path.mkdir()
    except OSError as e:
        doc["ram_cap"] = dict(status="unavailable", reason=f"{type(e).__name__}: {e}")
        save_doc(out, doc)
        log(f"cgroup unavailable: {e}")
        return
    store = DATA / "qdrant" / f"cap_store_{os.getpid()}"
    cap = dict(status="ran", cgroup=cg_path.as_posix())
    try:
        cap["page_cache_probe"] = probe_page_cache(cg)
        log(f"page cache probe: counted={cap['page_cache_probe'].get('page_cache_counted')}")
        set_limit(cg, 2 ** 40)  # uncapped for the build and baseline

        srv = Server(DATA / "qdrant" / f"cap_empty_{os.getpid()}", tag="cap_empty")
        srv.start()
        cap["baseline_empty_rss_mb"] = srv.rss_mb()
        srv.stop()
        shutil.rmtree(DATA / "qdrant" / f"cap_empty_{os.getpid()}", ignore_errors=True)
        log(f"baseline empty RSS {cap['baseline_empty_rss_mb']} MB")

        # build the two collections (originals on_disk) without the cap
        srv = Server(store, tag="cap_build")
        srv.start()
        client = srv.client()
        cap["builds"] = {}
        cap["builds"]["fp32_d768_disk"] = build(client, "fp32_d768_disk", base, None, disk_vectors=True)
        cap["builds"]["bq2_d768"] = build(client, "bq2_d768", base, "bq2")
        client.close()
        srv.stop()

        srv = Server(store, tag="cap_idle")
        srv.start()
        idle = srv.rss_detail()
        cap["loaded_idle_rss_mb"], cap["loaded_idle_anon_mb"], cap["loaded_idle_file_mb"] = idle["rss"], idle["anon"], idle["file"]
        srv.stop()
        log(f"loaded idle RSS {idle['rss']} MB (anon {idle['anon']}, file {idle['file']})")

        fp32_mb = base.nbytes / 2 ** 20
        # auto cap: anonymous RSS of the loaded server + 64 MB. File-backed mmap pages (the
        # on_disk fp32 originals) are evictable under the cap and are not counted here.
        cap_mb = args.ram_cap_mb or math.ceil(idle["anon"] + 64)
        cap["cap_mb"] = cap_mb
        cap["fp32_vectors_mb"] = round(fp32_mb, 1)
        cap["cap_excludes_fp32"] = cap_mb < cap["baseline_empty_rss_mb"] + fp32_mb
        if not cap["cap_excludes_fp32"]:
            log(f"WARNING cap {cap_mb} MB does not exclude fp32 vectors (baseline + {fp32_mb:.0f} MB)")

        srv = Server(store, tag="cap_run", cgroup=cg)
        set_limit(cg, 2 ** 40)
        srv.start()  # started in the cgroup; the loaded collections are read at startup
        client = srv.client()
        cap["runs"] = []
        plan = [("fp32_d768_disk", None, None), ("bq2_d768", "bq2", False), ("bq2_d768", "bq2", True)]
        for coll, quant, rescore in plan:
            for phase, lim in (("uncapped_control", 2 ** 40), ("capped", cap_mb)):
                if phase == "uncapped_control" and args.no_cap_control:
                    continue
                set_limit(cg, lim)
                log(f"  {phase}: {coll} rescore={rescore} limit={'none' if lim > 2**30 else str(lim)+'MB'}")
                pts = sweep(client, coll, truncate(Q, DIM), truth, quant, rescore, args.efs)
                cap["runs"].append(dict(phase=phase, collection=coll, rescore=rescore,
                                        limit_mb=None if lim > 2 ** 30 else lim,
                                        points=pts, cgroup=cg_stat(cg), rss_mb=srv.rss_mb()))
                save_doc(out, dict(doc, ram_cap=cap))
        client.close()
        srv.stop()
    except Exception as e:
        cap["status"] = f"error: {type(e).__name__}: {e}"
        log(cap["status"])
        raise
    finally:
        stop_all()
        shutil.rmtree(store, ignore_errors=True)
        try:
            set_limit(cg, 2 ** 40)
            (cg_path / "memory.force_empty").write_text("0")
        except OSError:
            pass
        try:
            cg_path.rmdir()
        except OSError as e:
            log(f"could not remove cgroup {cg_path}: {e}")
        doc["ram_cap"] = cap
        save_doc(out, doc)
        log(f"wrote {out} (ram_cap)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shards", type=int, default=10, help="number of 10k-doc shards in the base (default 10 = 100k)")
    ap.add_argument("--nq", type=int, default=1000, help="number of queries (default 1000)")
    ap.add_argument("--smoke", action="store_true", help="2 shards, 100 held-out base vectors as queries, ef 32/128")
    ap.add_argument("--ram-cap-mb", type=int, default=None,
                    help="run RAM-cap mode instead of the sweep; the cap in MB, 0 = auto (loaded idle RSS + 64 MB)")
    ap.add_argument("--no-cap-control", action="store_true", help="RAM-cap mode: skip the uncapped control pass")
    ap.add_argument("--out", default=None, help="output JSON (default results_qdrant.json; smoke: data/results_qdrant_smoke.json)")
    args = ap.parse_args()
    if args.smoke:
        args.shards, args.nq, args.heldout = 2, 100, True
        args.efs = [32, 128]
    else:
        args.heldout = False
        args.efs = EFS
    if args.ram_cap_mb is not None and args.ram_cap_mb < 0:
        ap.error("--ram-cap-mb must be >= 0")
    if args.out:
        out = Path(args.out)
    elif args.smoke:
        out = DATA / "results_qdrant_smoke.json"
    else:
        out = HERE / "results_qdrant.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not Path(QD_BIN).exists():
        ap.error(f"qdrant binary not found: {QD_BIN}")
    if args.ram_cap_mb is not None:
        if not CG_ROOT.exists():
            ap.error("cgroup v1 memory controller not found")
        run_ram_cap(args, out)
    else:
        run_sweep(args, out)


if __name__ == "__main__":
    main()
