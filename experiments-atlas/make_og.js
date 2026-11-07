// Renders web/og.png (1200x630) from web/index.html. Run: NODE_PATH=$(npm root -g) node make_og.js
const {chromium}=require('playwright');
(async()=>{const b=await chromium.launch();const p=await b.newPage({viewport:{width:1200,height:630},deviceScaleFactor:1});
await p.goto('file://'+__dirname+'/web/index.html',{waitUntil:'networkidle'});
await p.addStyleTag({content:`
.controls,.chips,aside,.zoom,.hint,.sub{display:none!important}
html,body{height:630px;overflow:hidden}
.wrap{max-width:none;padding:0;margin:0;display:block;height:630px;position:relative}
header{position:absolute;left:56px;top:64px;width:420px;z-index:2}
h1{font-size:56px;line-height:1.05}
.stage{display:block}
.map{position:absolute;left:380px;top:0;width:820px;height:630px;max-height:none;aspect-ratio:auto;border:0;border-radius:0;background:transparent}
.og{position:absolute;left:56px;bottom:56px;width:380px;z-index:2;color:var(--muted);font-size:20px;line-height:1.4}
.og b{display:block;color:var(--fg);font:650 44px var(--f-display);margin-bottom:6px}
`});
await p.evaluate(()=>{const d=document.createElement('div');d.className='og';d.innerHTML='<b>217 experiments</b>in 12 clusters, from embeddings and quantization to WebGPU, agents and film.';document.querySelector('.wrap').appendChild(d);document.getElementById('exp').click()});
await p.waitForTimeout(1200);await p.evaluate(()=>document.getElementById('zf').click());await p.waitForTimeout(900);
await p.screenshot({path:__dirname+'/web/og.png'});await b.close()})();
