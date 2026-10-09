import json,sys,glob,collections
V={"holds","stale","contradicted","unverifiable"}
def check(j):
    inp=[json.loads(l) for l in open(f'vchunks/v{j:03d}.jsonl')]
    try: out=[json.loads(l) for l in open(f'verdicts/v{j:03d}.jsonl') if l.strip()]
    except FileNotFoundError: return None
    except Exception as e: return ('badjson',str(e))
    probs=[]
    if len(out)!=len(inp): probs.append(f'count {len(out)}/{len(inp)}')
    ids=collections.Counter(c['mem'] for c in inp)
    for o in out:
        if o.get('verdict') not in V: probs.append('verdict:'+str(o.get('verdict')))
        if o.get('mem') not in ids: probs.append('mem:'+str(o.get('mem')))
        if not o.get('probe') or not str(o.get('evidence','')).strip(): probs.append('noevidence:'+str(o.get('mem')))
        if o.get('verdict') in ('stale','contradicted') and not o.get('correction'): probs.append('nocorr:'+o['mem'])
    return out,probs
if __name__=='__main__':
    js=[int(a) for a in sys.argv[1:]] or [int(p[10:13]) for p in sorted(glob.glob('verdicts/v*.jsonl'))]
    tot=collections.Counter()
    for j in js:
        r=check(j)
        if r is None: print(f'v{j:03d} missing'); continue
        if r[0]=='badjson': print(f'v{j:03d} BADJSON',r[1]); continue
        out,probs=r; c=collections.Counter(o.get('verdict') for o in out); tot+=c
        print(f'v{j:03d}',dict(c),probs[:6])
    print('TOTAL',dict(tot))
