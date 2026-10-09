"""Build verify chunks from finished, validated claim files not yet chunked. Skips claude.ai-env claims."""
import json, glob, os, re, sys
from validate_claims import check
D=os.getcwd()
done=set(json.load(open('chunked.json'))) if os.path.exists('chunked.json') else set()
P=open('verify_prompt.txt').read()
os.makedirs('vchunks',exist_ok=True); os.makedirs('verdicts',exist_ok=True); os.makedirs('vprompts',exist_ok=True)
new=[];used=[];skipped=0
for p in sorted(glob.glob('claims/b*.jsonl')):
    i=int(p[8:10])
    if i in done: continue
    n,ok,bad=check(i); used.append(i)
    for c in ok:
        if c.get('env')=='claude.ai': skipped+=1; continue
        c['batch']=i; new.append(c)
def key(c):
    m=re.search(r'oaustegard/([\w.-]+)',c['claim']+' '+c['check'])
    return (m.group(1) if m else 'zz_'+c['kind'], c['mem'])
new.sort(key=key)
start=len(glob.glob('vchunks/v*.jsonl'))
SZ=25
for k in range(0,len(new),SZ):
    j=start+k//SZ
    with open(f'vchunks/v{j:03d}.jsonl','w') as f:
        for c in new[k:k+SZ]: f.write(json.dumps({x:c.get(x,'') for x in ('mem','created','quote','claim','kind','check','env')})+'\n')
    n=len(new[k:k+SZ])
    open(f'vprompts/p{j:03d}.txt','w').write(P.format(IN=f'{D}/vchunks/v{j:03d}.jsonl',OUT=f'{D}/verdicts/v{j:03d}.jsonl',N=n))
json.dump(sorted(done|set(used)),open('chunked.json','w'))
print('batches',len(used),'claims',len(new),'skipped claude.ai',skipped,'chunks',start,'..',start+(len(new)-1)//SZ)
