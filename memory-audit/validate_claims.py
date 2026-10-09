"""Validate claims/bNN.jsonl against batches/bNN.txt: JSON shape, mem ids in batch, quote is verbatim."""
import json, re, sys, glob, os
KINDS={"path","skill","tool","network","limit","open_item","repo_state","other"}
def norm(s): return re.sub(r'\s+',' ',s).strip()
def check(i):
    bt=open(f'batches/b{i:02d}.txt').read()
    mems={}
    for m in re.finditer(r'^=== MEMORY (\w{8}) [^\n]*\n(.*?)(?=^=== MEMORY |\Z)',bt,re.M|re.S):
        mems[m.group(1)]=norm(m.group(2))
    p=f'claims/b{i:02d}.jsonl'
    if not os.path.exists(p): return None
    ok=[];bad=[]
    for n,line in enumerate(open(p)):
        if not line.strip(): continue
        try: c=json.loads(line)
        except Exception as e: bad.append((n,'json')); continue
        if set(c)>= {"mem","quote","claim","kind","check"} and c['mem'] in mems and c['kind'] in KINDS:
            if norm(c['quote']).rstrip('.…') in mems[c['mem']] or norm(c['quote'])[:60] in mems[c['mem']]:
                ok.append(c)
            else: bad.append((n,'quote'))
        else: bad.append((n,'shape/mem'))
    return len(mems),ok,bad
if __name__=='__main__':
    idx=[int(a) for a in sys.argv[1:]] or range(64)
    tot=0
    for i in idx:
        r=check(i)
        if r is None: print(f'b{i:02d} missing'); continue
        n,ok,bad=r; tot+=len(ok)
        print(f'b{i:02d} mems={n} ok={len(ok)} bad={len(bad)} {bad[:5]}')
    print('total ok',tot)
