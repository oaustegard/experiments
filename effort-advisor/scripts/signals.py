import json,re,statistics as st,collections
T=[json.loads(l) for l in open('data/turns.jsonl')]
C=json.load(open('data/calls.json'))
def norm(c): return c['tool']+'|'+re.sub(r'\d+','N',c['sig'])[:70]
POLL=re.compile(r'\bsleep\s+\d|gh run (view|watch|list)|gh pr checks|gh api .*(runs|status)|until |while .*do|tail -f|\bwait\b|poll',re.I)
POLLT={'TaskOutput','BashOutput','Monitor'}
def ispoll(c): return c['tool'] in POLLT or (c['tool']=='Bash' and bool(POLL.search(c['sig'])))
def fire(calls):
    ev={}
    n=len(calls); E=[c['err'] for c in calls]
    # a
    run=0
    for i in range(n):
        run=run+1 if E[i] else 0
        if run>=3: ev['err_streak']=(i,f"3+ consecutive errors ending: {calls[i]['tool']} {calls[i]['etext'][:80]}"); break
    # b0 exact
    H=[(c['tool'],c['h']) for c in calls]
    for i in range(n):
        if H[i][1]==0: continue
        w=H[max(0,i-9):i+1]
        if w.count(H[i])>=3: ev['repeat_exact']=(i,f"{calls[i]['tool']} {calls[i]['sig'][:80]} x{w.count(H[i])}"); break
    # b
    N=[norm(c) for c in calls]
    for i in range(n):
        w=N[max(0,i-9):i+1]
        if w.count(N[i])>=3: ev['repeat_near']=(i,f"{N[i][:90]} x{w.count(N[i])}"); break
    # c
    for i in range(n):
        if sum(E[max(0,i-9):i+1])>=5: ev['err_density']=(i,f">=5 errors in 10 calls, latest: {calls[i]['etext'][:80]}"); break
    # d
    edits=collections.defaultdict(list)
    for i,c in enumerate(calls):
        if 'file' in c and c['file']:
            edits[c['file']].append(i)
            L=edits[c['file']]
            if len(L)>=5 and any(calls[j]['err'] and calls[j]['tool']=='Bash' for j in range(L[0],i)):
                ev['edit_thrash']=(i,f"{c['file'][-60:]} edited {len(L)}x with bash failures between"); break
    # e2 relaxed: >=4 poll calls within 8
    for i in range(n):
        if sum(ispoll(c) for c in calls[max(0,i-7):i+1])>=4: ev['poll_relaxed']=(i,f"4+ poll in 8: {calls[i]['sig'][:80]}"); break
    # e
    run=0
    for i in range(n):
        run=run+1 if ispoll(calls[i]) else 0
        if run>=4: ev['poll_loop']=(i,f"4+ consecutive poll calls: {calls[i]['sig'][:80]}"); break
    # f same error text repeated >=3 in 10
    ET=[re.sub(r'\d+','N',c['etext'])[:50] if c['err'] else None for c in calls]
    for i in range(n):
        if ET[i] and ET[max(0,i-9):i+1].count(ET[i])>=3: ev['same_error']=(i,f"same error x3: {ET[i]}"); break
    return ev
CORR=re.compile(r"^\s*(no[,. !]|nope|stop|wait|wrong|that'?s not|not what|you (didn'?t|forgot|missed|should)|why did|undo|revert|actually|don'?t|didn'?t|doesn'?t work|still (broken|not|fail)|it'?s (broken|still)|try again|that (didn'?t|is wrong|isn'?t))",re.I)
rows=[];ep=[]
for t in T:
    c=C[str(t['id'])] if str(t['id']) in C else C[t['id']]
    ev=fire(c)
    corr=bool(t.get('next_prompt') and t.get('next_kind')=='human' and CORR.search(t['next_prompt']))
    rows.append(dict(id=t['id'],kind=t['kind'],n=len(c),ev=ev,intr=bool(t['interrupted']),corr=corr,dur=t['dur_s']))
    for s,(i,e) in ev.items(): ep.append(dict(turn_id=t['id'],kind=t['kind'],signal=s,call_index=i,total_calls=len(c),evidence=e))
with open('data/stuck.jsonl','w') as f:
    for e in ep: f.write(json.dumps(e)+'\n')
json.dump(rows,open('data/rows.json','w'))
SIG=['err_streak','repeat_exact','repeat_near','err_density','edit_thrash','poll_loop','poll_relaxed','same_error']
durs=sorted(r['dur'] for r in rows if r['dur']); p90=durs[int(.9*len(durs))]
print('p90 dur',p90)
def rep(sub,label):
    big=[r for r in sub if r['n']>=10]
    print(f"\n## {label}: turns={len(sub)} with>=10 calls={len(big)} interrupted={sum(r['intr'] for r in sub)} corr={sum(r['corr'] for r in sub)} long={sum(r['dur']>=p90 for r in sub)}")
    print('signal|fired|%all|%>=10|med first idx|med total|med remaining frac|intr%|corr%|long%')
    for s in SIG+['ANY']:
        F=[r for r in sub if (s=='ANY' and r['ev']) or s in r['ev']]
        if not F: print(s,0); continue
        fi=[ (min(v[0] for v in r['ev'].values()) if s=='ANY' else r['ev'][s][0])+1 for r in F]
        rem=[1-f/r['n'] for f,r in zip(fi,F)]
        remc=[r['n']-f for f,r in zip(fi,F)]
        print(f"{s}|{len(F)}|{100*len(F)/len(sub):.1f}|{100*sum(r['n']>=10 for r in F)/max(1,len(big)):.1f}|{st.median(fi)}|{st.median(r['n'] for r in F)}|{st.median(rem):.2f} (median calls left {st.median(remc)})|{100*sum(r['intr'] for r in F)/len(F):.0f}|{100*sum(r['corr'] for r in F)/len(F):.0f}|{100*sum(r['dur']>=p90 for r in F)/len(F):.0f}")
    base=[r for r in sub if not r['ev'] and r['n']>=10]
    print('baseline (no signal, >=10 calls):',len(base),'intr%',round(100*sum(r['intr'] for r in base)/max(1,len(base))),'corr%',round(100*sum(r['corr'] for r in base)/max(1,len(base))),'long%',round(100*sum(r['dur']>=p90 for r in base)/max(1,len(base))))
rep([r for r in rows if r['kind']=='human'],'HUMAN'); rep(rows,'ALL')
