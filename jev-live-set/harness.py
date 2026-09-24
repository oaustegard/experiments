import sys, json
from playwright.sync_api import sync_playwright
page_html = open('jev-live-set.html').read()
MOCK = r"""<script>
window.__reqs=[];
window.claude={use:(name)=>name==='mcp'?Promise.resolve({callTool:async(server,tool,input)=>{
  window.__reqs.push(input); await new Promise(r=>setTimeout(r,300+Math.random()*300));
  const out={}; for (const [k,q] of Object.entries(input.questions)){ const ks=Object.keys(q.criteria); const w=ks.map(()=>Math.random()**3); const s=w.reduce((a,b)=>a+b,0);
    const pr=Object.fromEntries(ks.map((k2,i)=>[k2,w[i]/s])); const ch=ks[w.indexOf(Math.max(...w))]; out[k]={type:'choice',choice:ch,probabilities:pr,confidence:.5}; }
  return {content:[{type:'text',text:JSON.stringify(out)+'\n[usage: jev-1.13.0 input 1500, output 900 tokens, key BYOK]'}], payload: JSON.stringify(out)+'\n[usage: jev-1.13.0 input 1500, output 900 tokens, key BYOK]'};
}}):Promise.resolve(null)};
</script>"""
mode = sys.argv[1]
html = ('<!doctype html><html><head><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1"></head><body>' + (MOCK if mode=='jev' else '') + page_html + '</body></html>')
open('test.html','w').write(html)
with sync_playwright() as p:
    b = p.chromium.launch(executable_path='/opt/pw-browsers/chromium', args=['--autoplay-policy=no-user-gesture-required'])
    pg = b.new_page(viewport={'width':1280,'height':1500})
    msgs=[]; pg.on('console', lambda m: msgs.append(m.type+': '+m.text[:300])); pg.on('pageerror', lambda e: msgs.append('PAGEERR '+str(e)[:400]))
    pg.goto('file://'+__import__('os').path.abspath('test.html')); pg.wait_for_timeout(800)
    if mode=='selftest':
        r = pg.evaluate("""() => { const J=window.__jevlive; const errs=[]; let n=0, haps=0;
          for (const sid of Object.keys(J.STYLES)) for (const L of J.LAYERS) for (const o of J.STYLES[sid].layers[L])
            for (let tonic=0; tonic<12; tonic++) for (const mode of ['minor','major']) { const key={tonic,mode};
              for (const pr of J.PROGS[mode]) for (const r of pr[0].split(' ')) { const ch=J.chordOf(r,key); const code=J.fillTemplate(o.code,ch,key); n++;
                try { const pat=J.compile(code); const hs=pat.queryArc(5,6).filter(h=>h.hasOnset()); haps+=hs.length;
                  for (const h of hs){ const v=h.value; if (v.note!==undefined){ const m=StrudelLib.core.noteToMidi(String(v.note)); if(!isFinite(m)) throw new Error('bad note '+v.note);} }
                  if (!hs.length && o.code!=='silence' && !/degradeBy/.test(o.code)) errs.push('nohaps '+o.id+' '+code);
                } catch(e){ errs.push(o.id+' | '+code+' | '+e.message); } } }
          return {n, haps, errs: [...new Set(errs)].slice(0,20), nerr: errs.length}; }""")
        print(json.dumps(r, indent=1))
    else:
        pg.click('#play'); pg.wait_for_timeout(int(sys.argv[2]) if len(sys.argv)>2 else 12000)
        if len(sys.argv)>3: 
            pg.fill('#steertext', sys.argv[3]); pg.click('#steerform button[type=submit]'); pg.wait_for_timeout(6000)
        info = pg.evaluate("""() => { const S=window.__jevlive.S; return {playing:S.playingBar, sched:S.scheduledUpTo, decided:S.decidedUpTo, calls:S.calls, late:S.late, lat:S.lat.slice(-5), phrases:[...S.phrases.values()].map(p=>[p.p,p.section,p.style,p.key.tonic+p.key.mode,p.by]), 
           bars:[...S.bars.values()].slice(-3).map(b=>[b.b,b.by,b.err||'',Object.values(b.haps).reduce((a,h)=>a+h.length,0)]), log:[...document.querySelectorAll('#log div')].slice(0,8).map(d=>d.textContent), conn:document.getElementById('conntext').textContent, note:document.getElementById('steernote').textContent} }""")
        print(json.dumps(info, indent=1))
        if mode=='jev':
            reqs = pg.evaluate("window.__reqs")
            json.dump(reqs, open('reqs.json','w'))
            print('requests', len(reqs), 'question sets', [sorted(r['questions'].keys()) for r in reqs[:3]])
        pg.screenshot(path='shot-%s.png'%mode, full_page=True)
    print('\n'.join(msgs[:25]))
    b.close()
