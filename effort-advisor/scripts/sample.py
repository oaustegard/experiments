import json,random,sys
E=[json.loads(l) for l in open('data/stuck.jsonl')]
C=json.load(open('data/calls.json'))
strict=[e for e in E if e['signal']!='repeat_near']
near=[e for e in E if e['signal']=='repeat_near']
random.seed(42)
S=random.sample(strict,min(25,len(strict)));N=random.sample(near,5)
json.dump({'strict':S,'near':N},open('data/sample.json','w'))
print(len(strict))
which=sys.argv[1]; lo,hi=int(sys.argv[2]),int(sys.argv[3])
for k,e in enumerate((S if which=='s' else N)[lo:hi],lo):
    cs=C[str(e['turn_id'])]; i=e['call_index']
    print(f"\n=== [{which}{k}] turn {e['turn_id']} {e['signal']} at {i+1}/{e['total_calls']} :: {e['evidence'][:100]}")
    for j in range(max(0,i-8),min(len(cs),i+4)):
        c=cs[j]; print(f"{'>' if j==i else ' '}{j+1:3} {'ERR' if c['err'] else '   '} {c['tool'][-22:]:22} {c['sig'][:90]!r} {c['etext'][:70] if c['err'] else ''}")
