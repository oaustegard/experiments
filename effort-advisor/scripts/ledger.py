import json,subprocess,collections,re
T=[json.loads(l) for l in open('data/turns.jsonl')]
R=json.load(open('data/rows.json'))
E=[json.loads(l) for l in open('data/stuck.jsonl')]
G='/Users/austegard/Projects/claude-workspace'
fl=[json.loads(l) for l in subprocess.run(['git','-C',G,'show','origin/failure-mining:flags.jsonl'],capture_output=True,text=True).stdout.splitlines()]
s8=lambda f:(re.search(r'Z-([0-9a-f]{8})-',f.split('/')[-1]) or [None,None])[1]
sess=collections.defaultdict(list)
for t in T:
    k=s8(t['file']) if t['source']=='archive' else None
    if k: sess[k].append(t['id'])
print('archive sessions in turns',len(sess))
STR={'err_streak','repeat_exact','err_density','edit_thrash','poll_loop','poll_relaxed','same_error'}
fired=collections.defaultdict(set)
for e in E:
    if e['signal'] in STR: fired[e['turn_id']].add(e['signal'])
def sfire(k,sig=None): 
    return any((fired[i] if sig is None else fired[i]&{sig}) for i in sess[k])
base=[k for k in sess]; print('base rate session fires strict:',sum(sfire(k) for k in base),len(base))
cats={'repeated_failure','stall_or_wait','tool_misuse'}
fk=[f for f in fl if f['category'] in cats and f.get('kept')]
allf=[f for f in fl if f['category'] in cats]
for name,F in (('kept/verified',fk),('all',allf)):
    ks=sorted({f['session_id'] for f in F}); inn=[k for k in ks if k in sess]
    print(name,'flags',len(F),'sessions',len(ks),'in corpus',len(inn),'fire',sum(sfire(k) for k in inn))
    for c in cats:
        kc=sorted({f['session_id'] for f in F if f['category']==c and f['session_id'] in sess})
        print(' ',c,len(kc),sum(sfire(k) for k in kc))
    for s in sorted(STR):
        print('  ',s,sum(sfire(k,s) for k in inn))
# turn-level: sessions with fires but unflagged in other categories
flagged_any={f['session_id'] for f in fl if f.get('kept')}
nf=[k for k in sess if k not in flagged_any]; print('unflagged sessions fire',sum(sfire(k) for k in nf),len(nf))
# fire with repeat_near too
nearfire={e['turn_id'] for e in E if e['signal']=='repeat_near'}
ks=sorted({f['session_id'] for f in fk if f['session_id'] in sess})
print('incl repeat_near',sum(any(i in nearfire or fired[i] for i in sess[k]) for k in ks),len(ks))
print('flag turns wasted median',sorted(f['turns_wasted'] for f in fk)[len(fk)//2])
