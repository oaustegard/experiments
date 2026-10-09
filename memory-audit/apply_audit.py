"""Fold verdicts + rechecks into per-memory corrections. --apply supersedes; default is a dry run."""
import json, glob, sys, collections, re
TAG='audited-2026-10-09'
EXCLUDE={'6a000334','b31b3cad'}  # claude.ai-container bash timeouts, checked against the wrong machine
mems={}
for l in open('memories.jsonl'):
    r=json.loads(l); mems[r['id'][:8]]=r
def load(pat):
    out=[]
    for p in sorted(glob.glob(pat)):
        for l in open(p):
            l=l.strip()
            if not l: continue
            try: out.append(json.loads(l))
            except Exception: pass
    return out
verdicts=load('verdicts/v*.jsonl'); rechecks=load('rechecks/r*.jsonl')
rk={(o.get('mem'),o.get('claim')):o for o in rechecks}
final=collections.defaultdict(list); stats=collections.Counter(); unsure=[]
for v in verdicts:
    stats['v_'+str(v.get('verdict'))]+=1
    if v.get('verdict') not in ('stale','contradicted'): continue
    r=rk.get((v.get('mem'),v.get('claim')))
    if r is None: stats['r_missing']+=1; continue
    s=r.get('recheck'); stats['r_'+str(s)]+=1
    if s=='confirmed': corr=v.get('correction') or r.get('correction')
    elif s=='confirmed_bad_correction': corr=r.get('correction')
    elif s=='unsure': unsure.append((v,r)); continue
    else: continue
    if v['mem'] in mems and corr:
        final[v['mem']].append({'claim':v['claim'],'correction':corr.strip(),'kind':v['verdict'],'probe':(r.get('probe') or v.get('probe') or '')[:160]})
def header(items):
    lines=[f"[AUDIT 2026-10-09: {len(items)} statement(s) below no longer hold. Read the original with these corrections; each was checked twice against live repos/services.]"]
    for it in items:
        lines.append(f"- WAS: {it['claim']} NOW: {it['correction']}")
    return '\n'.join(lines)+'\n\n'
plan=[]
for m,items in final.items():
    orig=mems[m]
    if orig['summary'].startswith('[AUDIT 2026-10-09') or m in EXCLUDE: continue
    tags=list(dict.fromkeys((orig['tags'] or [])+[TAG]+(['audit-contradicted'] if any(i['kind']=='contradicted' for i in items) else [])))
    plan.append({'id':orig['id'],'type':orig['type'],'priority':int(orig['priority'] or 0),'tags':tags,'summary':header(items)+orig['summary'],'n':len(items)})
json.dump(plan,open('apply_plan.json','w'),indent=1)
json.dump([{'mem':v['mem'],'claim':v['claim'],'verdict':v['verdict'],'recheck_evidence':r.get('evidence','')} for v,r in unsure],open('unsure.json','w'),indent=1)
print(dict(stats)); print('memories to supersede',len(plan),'claims',sum(p['n'] for p in plan),'unsure',len(unsure))
if '--apply' in sys.argv:
    sys.path.insert(0,'/mnt/skills/user/remembering')
    from scripts.memory import supersede
    done=json.load(open('applied.json')) if __import__('os').path.exists('applied.json') else {}
    for p in plan:
        if p['id'] in done: continue
        try:
            new=supersede(p['id'],p['summary'],p['type'],tags=p['tags'],priority=p['priority'],drift_class='narrowing' if p['type']=='procedure' else None)
            done[p['id']]=str(new)
        except Exception as e:
            print('FAIL',p['id'][:8],type(e).__name__,str(e)[:120]); done[p['id']]=None
        json.dump(done,open('applied.json','w'))
    print('applied',sum(1 for v in done.values() if v),'failed',sum(1 for v in done.values() if not v))
