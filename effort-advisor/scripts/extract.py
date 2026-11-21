import json,collections,re,os
os.environ['PYTHONHASHSEED']='0'
T=[json.loads(l) for l in open('data/turns.jsonl')]
byfile=collections.defaultdict(list)
for t in T: byfile[t['file']].append(t)
for f in byfile: byfile[f].sort(key=lambda t:t['idx'])
def sig(name,inp):
    if not isinstance(inp,dict): return str(inp)[:120]
    for k in ('command','file_path','path','pattern','url','query','prompt','description','skill','notebook_path'):
        if k in inp and inp[k]: 
            return re.sub(r'\s+',' ',str(inp[k]))[:120]
    return json.dumps(inp,sort_keys=True)[:120]
def errtext(c):
    if isinstance(c,list):
        c=' '.join(x.get('text','') if isinstance(x,dict) else str(x) for x in c)
    return re.sub(r'\s+',' ',str(c))[:160]
out={}
for f,ts in byfile.items():
    recs=[]
    for l in open(f):
        try: r=json.loads(l)
        except: continue
        recs.append(r)
    starts={t['uuid']:i for i,t in enumerate(ts)}
    cur=None; calls=collections.OrderedDict(); per={t['id']:[] for t in ts}
    idmap={}
    for r in recs:
        if r.get('isSidechain'): continue
        u=r.get('uuid')
        if r.get('type')=='user' and u in starts:
            cur=ts[starts[u]]['id']; continue
        if cur is None: continue
        m=r.get('message') or {}
        c=m.get('content')
        if not isinstance(c,list): continue
        if r['type']=='assistant':
            for b in c:
                if b.get('type')=='tool_use':
                    d={'h':hash(json.dumps(b.get('input'),sort_keys=True))%10**9 if b.get('input') else 0,'tool':b['name'],'sig':sig(b['name'],b.get('input')),'err':False,'etext':'','ts':r.get('timestamp')}
                    if b['name'] in('Edit','Write','NotebookEdit','MultiEdit'): d['file']=(b.get('input') or {}).get('file_path')
                    per[cur].append(d); idmap[b['id']]=d
        elif r['type']=='user':
            for b in c:
                if b.get('type')=='tool_result' and b.get('tool_use_id') in idmap:
                    d=idmap[b['tool_use_id']]
                    d['err']=bool(b.get('is_error'))
                    if d['err']: d['etext']=errtext(b.get('content'))
    out.update(per)
json.dump(out,open('data/calls.json','w'))
tt={t['id']:t for t in T}
bad=[(i,tt[i]['n_tools'],len(c)) for i,c in out.items() if tt[i]['n_tools']!=len(c)]
print(len(out),len(bad),bad[:10])
print(sum(len(c) for c in out.values()))
