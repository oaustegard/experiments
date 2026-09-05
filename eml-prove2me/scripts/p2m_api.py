"""Minimal Prove2Me client for this session: token from credentials.json, JSON helpers, multipart verify."""
import json, time, urllib.request, urllib.error, uuid, os
W = "/home/user/claude-workspace/.spokes/prove2me_workspace"; B = "https://prove2.me/api/v1"
_tok = {"v": None, "exp": 0}
def token():
    if _tok["v"] and time.time() < _tok["exp"] - 120: return _tok["v"]
    cred = json.load(open(f"{W}/credentials.json"))
    st, r = call("/agent/refresh", {"api_key": cred["api_key"]}, auth=False)
    assert st == 200, (st, r)
    _tok["v"], _tok["exp"] = r["access_token"], r.get("expires_at", time.time() + 3000); return _tok["v"]
def call(path, body=None, method=None, auth=True):
    data = json.dumps(body).encode() if body is not None else None
    h = {"Content-Type": "application/json", "User-Agent": "muninn-raven"}
    if auth: h["Authorization"] = f"Bearer {token()}"
    req = urllib.request.Request(B + path, data=data, headers=h, method=method or ("POST" if data is not None else "GET"))
    for attempt in range(4):
        try:
            r = urllib.request.urlopen(req, timeout=120); raw = r.read(); return r.status, (json.loads(raw) if raw else {})
        except urllib.error.HTTPError as e:
            break
        except (urllib.error.URLError, OSError) as e:
            if attempt == 3: raise
            time.sleep(3 * (attempt + 1))
    try:
        raise e
    except urllib.error.HTTPError as e:
        raw = e.read().decode(); 
        try: return e.code, json.loads(raw)
        except Exception: return e.code, raw[:500]
def verify(theorem_id, path, explanation=None, proof_type=None):
    bnd = uuid.uuid4().hex; parts = []
    def field(name, val): parts.append(f"--{bnd}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode())
    field("theorem_id", theorem_id)
    if proof_type: field("proof_type", proof_type)
    if explanation: field("explanation", explanation)
    parts.append(f"--{bnd}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"solution.lean\"\r\nContent-Type: text/plain\r\n\r\n".encode() + open(path, "rb").read() + b"\r\n")
    parts.append(f"--{bnd}--\r\n".encode())
    body = b"".join(parts)
    req = urllib.request.Request(B + "/verify", data=body, headers={"Content-Type": f"multipart/form-data; boundary={bnd}", "User-Agent": "muninn-raven", "Authorization": f"Bearer {token()}"})
    try:
        r = urllib.request.urlopen(req, timeout=120); return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode()[:500]
def poll_job(job_id, every=5, limit=600):
    t0 = time.time()
    while time.time() - t0 < limit:
        st, r = call(f"/publish-jobs/{job_id}")
        if r.get("status") in ("PUBLISHED", "FAILED", "ERROR"): return r
        time.sleep(every)
    return {"status": "TIMEOUT"}
def poll_sub(sub_id, every=5, limit=900):
    t0 = time.time()
    while time.time() - t0 < limit:
        st, r = call(f"/verify?submission_id={sub_id}")
        if r.get("status") not in (None, "PENDING", "QUEUED", "RUNNING", "COMPILING", "Pending", "Running"): return r
        time.sleep(every)
    return {"status": "TIMEOUT"}
