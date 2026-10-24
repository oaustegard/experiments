// Pull the inline blocks out of index.html for checking.
const fs = require('fs');
const html = fs.readFileSync(__dirname + '/../web/index.html', 'utf8');
const get = (re) => { const m = html.match(re); if (!m) throw new Error('missing ' + re); return m[1]; };
module.exports = {
  skeleton: get(/\/\/ BEGIN skeleton\n([\s\S]*?)\/\/ END skeleton/),
  bones: get(/<script type="text\/plain" id="wgsl-bones">([\s\S]*?)<\/script>/),
  composite: get(/<script type="text\/plain" id="wgsl-composite">([\s\S]*?)<\/script>/),
  module: get(/<script type="module">([\s\S]*?)<\/script>/),
};
if (require.main === module) {
  const out = __dirname + '/out'; fs.mkdirSync(out, { recursive: true });
  for (const [k, v] of Object.entries(module.exports)) fs.writeFileSync(`${out}/${k}.${k === 'bones' || k === 'composite' ? 'wgsl' : (k === 'module' ? 'mjs' : 'js')}`, v);
  console.log('wrote', fs.readdirSync(out).join(' '));
}
