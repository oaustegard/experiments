import json, math, sys, zlib, zstandard
HERE=sys.argv[1]; V1=open(sys.argv[2],'rb').read()
c=json.load(open(f"{HERE}/corpus.json")); train=[d["text"].encode() for d in c["train"]]; test=[d["text"].encode() for d in c["test"]]
big=b"\x00".join(train)
def content_of(zd):
    b=zd.as_bytes()
    for p in range(8,len(b)-64):
        if all(b[q:q+8] in big for q in range(p,p+64,8)): return b[p:]
def dfl(raw,zdict=None,strategy=zlib.Z_DEFAULT_STRATEGY):
    o=zlib.compressobj(9,zlib.DEFLATED,-15,9,strategy,**({"zdict":zdict} if zdict else {})); return o.compress(raw)+o.flush()
def both(raw,d): return min(len(dfl(raw,d)),len(dfl(raw,d,zlib.Z_FIXED)))
b64=lambda n: math.ceil(n*8/6)
def report(name,f):
    s=sum(b64(f(r)) for r in test); print(f"{name:52} {s:6} {100*(s-base)/base:+6.1f}%")
base=sum(b64(min(len(dfl(r,V1)),len(dfl(r)),len(r))) for r in test)
print(f"{'v1 (dynamic Huffman only)':52} {base:6}")
report("v1 dict, encoder also tries fixed Huffman", lambda r: both(r,V1))
out={}
for sz in (8192,16384,32768):
    zd=zstandard.train_dictionary(sz,train*4); ct=content_of(zd); out[sz]=ct
    report(f"deflate + zstd-trained content {len(ct)//1024} KB", lambda r: both(r,ct))
    report(f"deflate + zstd content {len(ct)//1024} KB + v1 last", lambda r: both(r,(ct+V1)[-32768:]))
    cc=zstandard.ZstdCompressor(level=22,dict_data=zd,write_checksum=False,write_content_size=False,write_dict_id=False)
    report(f"zstd 22 + trained dict {sz//1024} KB", lambda r: len(cc.compress(r)))
json.dump({k:v.hex() for k,v in out.items()},open(f"{HERE}/zcontent.json","w"))
