import json,glob,os
D=os.getcwd(); P=open('recheck_prompt.txt').read()
os.makedirs('rchunks',exist_ok=True); os.makedirs('rechecks',exist_ok=True); os.makedirs('rprompts',exist_ok=True)
done=set(json.load(open('rchunked.json'))) if os.path.exists('rchunked.json') else set()
from validate_verdicts import check
rows=[];used=[]
for p in sorted(glob.glob('verdicts/v*.jsonl')):
    j=int(p[10:13])
    if j in done: continue
    r=check(j)
    if r is None or r[0]=='badjson': continue
    used.append(j)
    rows+=[{k:o.get(k,'') for k in ('mem','claim','verdict','probe','evidence','correction')} for o in r[0] if o.get('verdict') in ('stale','contradicted')]
start=len(glob.glob('rchunks/r*.jsonl')); SZ=25
for k in range(0,len(rows),SZ):
    i=start+k//SZ; part=rows[k:k+SZ]
    with open(f'rchunks/r{i:03d}.jsonl','w') as f:
        for o in part: f.write(json.dumps(o)+'\n')
    open(f'rprompts/q{i:03d}.txt','w').write(P.format(IN=f'{D}/rchunks/r{i:03d}.jsonl',OUT=f'{D}/rechecks/r{i:03d}.jsonl',N=len(part)))
json.dump(sorted(done|set(used)),open('rchunked.json','w'))
print('verdict files',len(used),'rows',len(rows),'chunks',start,'..',start+max(0,len(rows)-1)//SZ)
