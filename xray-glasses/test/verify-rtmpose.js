// Decodes the RTMPose heads in rtm_raw.json (from run-rtmpose.py) with the page's own decodeSimcc,
// maps crop coordinates back to the image the way runAnimal() does, and compares with AP-10K ground
// truth. Tries both output orders and reports the mean keypoint error in pixels per image.
// Writes rtm_keypoints.json (page-order decode) for render-cpu.js.
const fs = require('fs');
const path = require('path');
const src = require('./extract.js');
const XR = (() => { const module = { exports: {} }; new Function('module', src.skeleton)(module); return module.exports; })();
const D = __dirname;
const raw = JSON.parse(fs.readFileSync(path.join(D, 'rtm_raw.json'), 'utf8'));
const gt = JSON.parse(fs.readFileSync(path.join(D, 'ap10k_gt.json'), 'utf8'));
const K = 17, BINS = 512;  // RTM_K, RTM_BINS in web/index.html
const out = { categories: gt.categories, runs: [] };
for (const r of raw.runs) {
  const [sx, sy, s] = r.crop, k = r.size / s;
  const g = gt.annotations.find(a => a.file === r.file).keypoints;
  const err = (xh, yh) => {
    const { pts, conf } = XR.decodeSimcc(Float32Array.from(xh), Float32Array.from(yh), K, BINS);
    const vp = pts.map(p => [sx + p[0] / k, sy + p[1] / k]);
    const d = [];
    for (let i = 0; i < K; i++) if (g[i * 3 + 2] > 0) d.push(Math.hypot(vp[i][0] - g[i * 3], vp[i][1] - g[i * 3 + 1]));
    return { vp, conf, mean: d.reduce((a, b) => a + b, 0) / d.length, max: Math.max(...d), n: d.length };
  };
  const page = err(r.output_0, r.output_1), swapped = err(r.output_1, r.output_0);
  // split ratio: decode at ratio 2 vs 1 (bins/1) and see which lands on the GT
  const diag = Math.hypot(...gt.annotations.find(a => a.file === r.file).keypoints.slice(0, 2));
  console.log(`${r.file}: page order (output_0=x, output_1=y) mean ${page.mean.toFixed(1)} px, max ${page.max.toFixed(1)} px over ${page.n} visible keypoints; swapped mean ${swapped.mean.toFixed(1)} px; crop side ${s.toFixed(0)} px (1 crop px = ${(1 / k).toFixed(2)} image px); scores ${Math.min(...page.conf).toFixed(2)}-${Math.max(...page.conf).toFixed(2)}`);
  out.runs.push({ file: r.file, keypoints: page.vp.map(p => [+p[0].toFixed(1), +p[1].toFixed(1)]), conf: page.conf.map(c => +c.toFixed(3)), mean_err_px: +page.mean.toFixed(2), max_err_px: +page.max.toFixed(2), swapped_mean_err_px: +swapped.mean.toFixed(2) });
}
fs.writeFileSync(path.join(D, 'rtm_keypoints.json'), JSON.stringify(out));
console.log('wrote rtm_keypoints.json');
