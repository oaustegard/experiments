// Count captured frames per shot into src/shots.json for the composition.
const fs = require('fs'), path = require('path');
const dir = path.resolve(__dirname, '../public/shots'), out = {};
for (const g of fs.readdirSync(dir)) {
  const p = path.join(dir, g); if (!fs.statSync(p).isDirectory()) continue;
  const n = fs.readdirSync(p).filter(f => /^f\d{4}\.jpg$/.test(f)).length; if (n) out[g] = n;
}
fs.writeFileSync(path.resolve(__dirname, '../src/shots.json'), JSON.stringify(out, null, 1));
console.log(out);
