import json, math, sys, zlib, gzip, qrcode
from qrcode.util import QRData, MODE_ALPHA_NUM, MODE_8BIT_BYTE
HERE=sys.argv[1]; V1=open(sys.argv[2],'rb').read()
Z=bytes.fromhex(json.load(open(f"{HERE}/zcontent.json"))["32768"]); D2=(Z+V1)[-32768:]
open(f"{HERE}/dict-v2.bin","wb").write(D2)
print("dictionary bytes:", len(V1), "->", len(D2), "| gzipped in a JS bundle:", len(gzip.compress(V1,9)), "->", len(gzip.compress(D2,9)))
def dfl(raw,d=None):
    o=zlib.compressobj(9,zlib.DEFLATED,-15,9,**({"zdict":d} if d else {})); return o.compress(raw)+o.flush()
def best(raw,d): return min(len(dfl(raw,d)),len(dfl(raw)),len(raw))
# out-of-distribution: documents nobody trained on
ood={"this session's deck": open(sys.argv[3],'rb').read(),
     "a plain-prose letter": ("Dear Ms. Alvarez,\n\nThank you for the quick turnaround on the lease renewal. "
       "I have read the draft and have two small changes: the pet clause should name both cats, and the "
       "renewal date should be the first of March rather than the fifteenth.\n\nI can sign on Thursday "
       "afternoon if that suits you.\n\nWith thanks,\nOskar").encode(),
     "Norwegian text": ("# Handleliste\n\n- [ ] Brød og smør\n- [ ] Melk, egg og ost\n- [ ] Epler til matpakken\n\n"
       "## Middag på fredag\n\nVi lager fiskesuppe med laks, torsk og reker. Husk fløte og dill.").encode()}
for n,r in ood.items():
    a,b=best(r,V1),best(r,D2); print(f"OOD {n:26} raw {len(r):5}  v1 {a:5}  v2 {b:5}  {100*(b-a)/a:+.1f}%")
# QR versions (error correction L) for the deck under three alphabets
import base64
r=ood["this session's deck"]; c=dfl(r,D2)
b64="y"+base64.urlsafe_b64encode(c).decode().rstrip("=")
def b62(data):
    A="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz"; out=""
    for i in range(0,len(data),32):
        blk=data[i:i+32]; n=int.from_bytes(blk,"big"); L=43 if len(blk)==32 else math.ceil(len(blk)*8/math.log2(62)); s=""
        for _ in range(L): n,m=divmod(n,62); s=A[m]+s
        out+=s
    return out
b32=base64.b32encode(c).decode().rstrip("=")
pre="https://tinylittleoffice.com/deck/#/v2/deck/"
def qv(segs):
    q=qrcode.QRCode(error_correction=qrcode.constants.ERROR_CORRECT_L)
    for data,mode in segs: q.add_data(QRData(data.encode(),mode=mode))
    q.make(fit=True); return q.version
for name,p,mode in (("base64url (today)",b64,MODE_8BIT_BYTE),("base62",b62(c),MODE_8BIT_BYTE),("base32 upper, alphanumeric QR segment",b32,MODE_ALPHA_NUM)):
    v=qv([(pre,MODE_8BIT_BYTE),(p,mode)]); print(f"QR deck {name:40} chars {len(pre)+len(p):5}  QR version {v} ({17+4*v}x{17+4*v} modules)")
# dictionary size against link length and bundle cost (the RESULTS.md table)
c2=json.load(open(f"{HERE}/corpus.json")); test=[d["text"].encode() for d in c2["test"]]
b=lambda n: math.ceil(n*8/6)
base=sum(b(best(r,V1)) for r in test)
for k in ("8192","16384","32768"):
    D=(bytes.fromhex(json.load(open(f"{HERE}/zcontent.json"))[k])+V1)[-int(k):]
    s=sum(b(min(len(dfl(r,D)),len(dfl(r)))) for r in test)
    print(f"dict {int(k)//1024:2} KB: links {100*(s-base)/base:+.1f}% | bundle +{(len(gzip.compress(D,9))-len(gzip.compress(V1,9)))/1024:.1f} KB gz")
