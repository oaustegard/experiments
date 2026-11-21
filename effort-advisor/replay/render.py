import json,re,os,collections
from build import select,ROOT,load
INSTR="Below is a conversation between a user and an AI coding assistant, then the user's newest message. Write the assistant's reply to that newest message. You have no tools: answer from the conversation alone. Reply with the answer only."
SR=re.compile(r'<system-reminder[^>]*>.*?</system-reminder>',re.S)
RED=[re.compile(p) for p in [r'sk-[A-Za-z0-9_-]{16,}',r'gh[pousr]_[A-Za-z0-9]{20,}',r'github_pat_[A-Za-z0-9_]{20,}',r'hf_[A-Za-z0-9]{20,}',r'xox[abprs]-[A-Za-z0-9-]{10,}',r'AKIA[0-9A-Z]{16}',r'AIza[0-9A-Za-z_-]{30,}',r'eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}(\.[A-Za-z0-9_-]+)?',r'(?i)(bearer\s+)[A-Za-z0-9._~+/-]{20,}',r'(?i)((?:api[_-]?key|token|secret|password|passwd|authorization)["\']?\s*[:=]\s*["\']?)[A-Za-z0-9._~+/-]{16,}',r'\b[A-Fa-f0-9]{40,}\b',r'-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----']]
def red(t):
    for i,p in enumerate(RED):
        t=p.sub(lambda m:(m.group(1) if p.groups and m.group(1) and i in(8,9) else '')+'[REDACTED]',t)
    return t
def clean(t): return red(SR.sub('',t)).strip()
def clip(t,n=500): 
    t=clean(t if isinstance(t,str) else json.dumps(t,ensure_ascii=False))
    return t if len(t)<=n else t[:n]+f' …[clipped {len(t)-n} chars]'
def tr_text(c):
    if isinstance(c,str): return c
    return '\n'.join(b.get('text','') if b.get('type')=='text' else f"[{b.get('type')}]" for b in c)
def render(rec,n=500):
    out=[]; m=rec.get('message') or {}; c=m.get('content'); t=rec['type']
    if t=='user':
        if isinstance(c,str):
            s=clean(c); 
            if s: out.append(('USER: ' if not rec.get('isMeta') else 'USER (injected context): ')+(s if not rec.get('isMeta') else clip(s,n)))
        else:
            for b in c or []:
                if b.get('type')=='text':
                    s=clean(b['text'])
                    if s: out.append('USER: '+(s if not rec.get('isMeta') else clip(s,n)))
                elif b.get('type')=='tool_result': out.append('TOOL RESULT: '+clip(tr_text(b.get('content') or ''),n))
                elif b.get('type')=='image': out.append('USER: [image]')
    elif t=='assistant':
        for b in c or []:
            if b.get('type')=='text':
                s=clean(b['text'])
                if s: out.append('ASSISTANT: '+s)
            elif b.get('type')=='tool_use': out.append(f"ASSISTANT TOOL CALL {b.get('name')}: "+clip(b.get('input'),n))
    return out
def is_human_start(x):
    if x.get('type')!='user' or x.get('isMeta') or x.get('isSidechain'): return False
    c=(x.get('message') or {}).get('content')
    return isinstance(c,str) or (c and all(b.get('type')!='tool_result' for b in c))
chosen,cands,cache=select(50)
ACTION=('Edit','Write','Artifact','SendUserFile','NotebookEdit','update_trigger','create_trigger','send_message')
items=[];dropped=[]
for x in chosen:
    recs=cache[x['file']]; idx=next(i for i,r in enumerate(recs) if r.get('uuid')==x['uuid'])
    parts=[]
    for r in recs[:idx]:
        if r.get('type') in('user','assistant') and not r.get('isSidechain'): parts+=render(r)
    # keep last ~60K chars on message boundaries
    kept=[];tot=0
    for p in reversed(parts):
        if tot+len(p)>60000 and kept: break
        kept.append(p[-60000:]);tot+=len(p)
    kept.reverse()
    trunc=len(kept)<len(parts)
    # in-turn tool activity
    inturn=[]
    for r in recs[idx+1:]:
        if is_human_start(r): break
        if r.get('type') in('user','assistant') and not r.get('isSidechain'):
            inturn+=[p for p in render(r,2000) if p.startswith(('TOOL RESULT','ASSISTANT TOOL CALL'))]
    names0=[p.split(':')[0].replace('ASSISTANT TOOL CALL ','') for p in inturn if p.startswith('ASSISTANT TOOL CALL')]
    if any(n.endswith(ACTION) for n in names0): dropped.append((x['id'],x['_model'],x['_effort'],'answer reports side-effecting tool actions made in the turn: '+','.join(names0),x['prompt'][:60])); continue
    if str(x['id'])=='789': dropped.append((x['id'],x['_model'],x['_effort'],'not a real question (persona line; original answer is about container boot state)',x['prompt'][:60])); continue
    if len(items)>=40 and x['_stratum']=='medium': continue
    note=[]
    if trunc: note.append('context truncated to last ~60K chars')
    if not parts: note.append('first turn of session: no prior conversation')
    names=[p.split(':')[0].replace('ASSISTANT TOOL CALL ','') for p in inturn if p.startswith('ASSISTANT TOOL CALL')]
    if inturn: note.append('original turn used tools ('+','.join(names)+'); their calls/results (clipped 2000) appended after the newest message')
    body=INSTR+'\n\n=== PRIOR CONVERSATION'+(' (earlier part omitted)' if trunc else '')+' ===\n\n'+('\n\n'.join(kept) if kept else '(none)')+'\n\n=== NEWEST USER MESSAGE ===\n\n'+clean(x['_prompt_full'])
    if inturn: body+='\n\n=== TOOL OUTPUT ALREADY GATHERED FOR THIS MESSAGE (you ran these; use the results, do not mention being unable to run tools) ===\n\n'+'\n\n'.join(inturn)
    body+='\n\n=== END. Write the assistant\'s reply to the newest user message. ===\n'
    pf=f"{ROOT}/replay/prompts/{x['id']}.md"; open(pf,'w').write(body)
    items.append(dict(id=x['id'],session=x['session'],model=x['_model'],effort=x['_effort'],stratum=x['_stratum'],n_tools=int(x['n_tools']),tools_in_turn=names,prompt=x['prompt'][:300],prompt_file=pf,original_answer=x['final_text'],prev_final=x.get('prev_final'),next_prompt=x.get('next_prompt'),context_chars=len(body),source=x['source'],idx=x['idx'],orig_out_tokens=x.get('out_tokens'),orig_dur_s=x.get('dur_s'),note='; '.join(note)))
with open(ROOT+'/replay/items.jsonl','w') as f:
    for i in items: f.write(json.dumps(i,ensure_ascii=False)+'\n')
print(len(cands),len(items)); print('DROPPED',*dropped,sep='\n'); print(collections.Counter((i['model'],i['effort']) for i in items)); print(collections.Counter(i['session'][:8] for i in items).most_common(4))
for i in items: print(i['id'],i['effort'],i['model'][7:],i['context_chars'],i['tools_in_turn'],repr(i['prompt'][:60]),'|',len(i['original_answer']))
