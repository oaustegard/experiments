"""Reply to a Muninn post with a natively uploaded Bluesky video.

usage: video_reply.py <parent at:// uri> <video.mp4> <text> <alt>
"""
import json, os, sys, time, urllib.parse, urllib.request
from datetime import datetime, timezone

PARENT, VIDEO, TEXT, ALT = sys.argv[1:5]
UA = {"User-Agent": "muninn-video-reply"}


def call(method, url, token=None, body=None, ctype="application/json", raw=False):
    h = dict(UA)
    if token: h["Authorization"] = f"Bearer {token}"
    data = None
    if body is not None:
        data = body if isinstance(body, bytes) else json.dumps(body).encode()
        h["Content-Type"] = ctype
    req = urllib.request.Request(url, data=data, headers=h, method=method)
    try:
        with urllib.request.urlopen(req, timeout=300) as r:
            return json.loads(r.read() or b"{}")
    except urllib.error.HTTPError as e:
        msg = e.read().decode()[:400]
        if raw: return {"_error": e.code, "_body": msg}
        sys.exit(f"{method} {url.split('?')[0]} -> {e.code}: {msg}")


sess = call("POST", "https://bsky.social/xrpc/com.atproto.server.createSession",
            body={"identifier": os.environ["MUNINN_BSKY_HANDLE"], "password": os.environ["MUNINN_BSKY_APP_PASSWORD"]})
did, jwt = sess["did"], sess["accessJwt"]
pds = next(s["serviceEndpoint"] for s in sess["didDoc"]["service"] if s["id"].endswith("atproto_pds"))
aud = "did:web:" + urllib.parse.urlparse(pds).hostname

svc = call("GET", f"{pds}/xrpc/com.atproto.server.getServiceAuth?" + urllib.parse.urlencode(
    {"aud": aud, "lxm": "com.atproto.repo.uploadBlob", "exp": int(time.time()) + 1800}), jwt)["token"]
up = call("POST", "https://video.bsky.app/xrpc/app.bsky.video.uploadVideo?" + urllib.parse.urlencode(
    {"did": did, "name": os.path.basename(VIDEO)}), svc, open(VIDEO, "rb").read(), "video/mp4", raw=True)
if "_error" in up:
    if "already_exists" in up.get("_body", ""):
        up = {"jobStatus": json.loads(up["_body"])}
    else:
        sys.exit(f"upload failed: {up}")
job = (up.get("jobStatus") or up)
job_id = job.get("jobId")
blob = job.get("blob")
while not blob:
    time.sleep(4)
    st = call("GET", "https://video.bsky.app/xrpc/app.bsky.video.getJobStatus?jobId=" + job_id, jwt)["jobStatus"]
    print("job", st.get("state"), st.get("progress"), flush=True)
    if st.get("state") == "JOB_STATE_FAILED":
        sys.exit(f"processing failed: {st}")
    blob = st.get("blob")

# reply refs: parent, and its root
repo, _, rkey = PARENT[len("at://"):].split("/")
parent = call("GET", f"{pds}/xrpc/com.atproto.repo.getRecord?" + urllib.parse.urlencode(
    {"repo": repo, "collection": "app.bsky.feed.post", "rkey": rkey}), jwt)
pref = {"uri": parent["uri"], "cid": parent["cid"]}
root = parent["value"].get("reply", {}).get("root", pref)
rec = {"$type": "app.bsky.feed.post", "text": TEXT, "createdAt": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
       "langs": ["en"], "reply": {"root": root, "parent": pref},
       "embed": {"$type": "app.bsky.embed.video", "video": blob, "alt": ALT, "aspectRatio": {"width": 1280, "height": 720}}}
out = call("POST", f"{pds}/xrpc/com.atproto.repo.createRecord", jwt,
           {"repo": did, "collection": "app.bsky.feed.post", "record": rec})
print(json.dumps({"uri": out["uri"], "url": f"https://bsky.app/profile/{did}/post/{out['uri'].rsplit('/', 1)[1]}"}))
