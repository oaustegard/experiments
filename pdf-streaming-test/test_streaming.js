// Playwright test: verify the PDF Text Extractor streams pages in and that
// varying `concurrency` produces valid output on the same PDF.
//
// We serve the tool page from http://127.0.0.1:4001/ (Python http.server rooted
// at /home/user/oaustegard.github.io). Any browser request to cdnjs or to the
// PDF URL we pass in is intercepted by Playwright and answered from
// experiments/pdf-streaming-test/vendor/, so the whole run works offline.

const fs = require('fs');
const path = require('path');
const { chromium } = require('/opt/node22/lib/node_modules/playwright');

const BASE = 'http://127.0.0.1:4001/web-utilities/pdf-text-extractor.html';
const VENDOR = path.resolve(__dirname, 'vendor');

const routes = {
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.min.js': {
    body: fs.readFileSync(path.join(VENDOR, 'pdf.min.js')),
    contentType: 'application/javascript',
  },
  'https://cdnjs.cloudflare.com/ajax/libs/pdf.js/3.11.174/pdf.worker.min.js': {
    body: fs.readFileSync(path.join(VENDOR, 'pdf.worker.min.js')),
    contentType: 'application/javascript',
  },
  'https://arxiv.org/pdf/test': {
    body: fs.readFileSync(path.join(VENDOR, 'test.pdf')),
    contentType: 'application/pdf',
  },
  'https://arxiv.org/pdf/big': {
    body: fs.readFileSync(path.join(VENDOR, 'big.pdf')),
    contentType: 'application/pdf',
  },
  'https://arxiv.org/pdf/bigger': {
    body: fs.readFileSync(path.join(VENDOR, 'bigger.pdf')),
    contentType: 'application/pdf',
  },
};

async function runOne(browser, pdfKey, concurrency) {
  const ctx = await browser.newContext();
  const page = await ctx.newPage();
  await page.route('**/*', async (route) => {
    const url = route.request().url();
    if (routes[url]) {
      return route.fulfill({ status: 200, body: routes[url].body, headers: { 'content-type': routes[url].contentType } });
    }
    return route.continue();
  });

  const errors = [];
  page.on('pageerror', e => errors.push('pageerror: ' + e.message));
  page.on('console', m => { if (m.type() === 'error') errors.push('console.error: ' + m.text()); });

  const pdfUrl = `https://arxiv.org/pdf/${pdfKey}`;
  const url = `${BASE}#url=${encodeURIComponent(pdfUrl)}&concurrency=${concurrency}&format=markdown`;

  const snapshots = [];
  const start = Date.now();
  await page.goto(url, { waitUntil: 'domcontentloaded' });

  // Poll every 150ms until we see the ✓ line or a warning.
  let sawSuccess = false;
  while ((Date.now() - start) < 180000) {
    const s = await page.evaluate(() => ({
      progText: document.getElementById('progress')?.textContent || '',
      outLen: document.getElementById('text-output')?.textContent.length || 0,
      pagesRendered: (document.getElementById('text-output')?.textContent.match(/^## Page \d+/gm) || []).length,
    }));
    snapshots.push({ t: Date.now() - start, ...s });
    if (s.progText.startsWith('✓')) { sawSuccess = true; break; }
    if (s.progText.startsWith('⚠')) break;
    await new Promise(r => setTimeout(r, 150));
  }

  const finalOut = await page.$eval('#text-output', el => el.textContent);
  const pagesInOutput = (finalOut.match(/^## Page \d+/gm) || []).map(s => parseInt(s.match(/\d+/)[0], 10));
  const orderedOK = pagesInOutput.every((v, i, a) => i === 0 || v > a[i - 1]);

  await ctx.close();
  return {
    pdfKey, concurrency,
    elapsedMs: Date.now() - start,
    success: sawSuccess,
    finalProg: (await page.$eval('#progress', el => el.textContent).catch(() => '')) || snapshots[snapshots.length - 1]?.progText || '',
    finalOutLen: finalOut.length,
    pagesInOutput,
    orderedOK,
    errors: errors.slice(0, 10),
    // A sparse timeline of "pages rendered so far" to prove streaming happens.
    growth: snapshots.filter((s, i, arr) => i === 0 || s.pagesRendered !== arr[i - 1].pagesRendered).map(s => ({ t: s.t, r: s.pagesRendered, o: s.outLen })),
  };
}

(async () => {
  const browser = await chromium.launch({ executablePath: '/opt/pw-browsers/chromium', headless: true });

  const results = [];
  for (const pdfKey of ['test', 'big', 'bigger']) {
    for (const c of [1, 2, 4, 8]) {
      process.stdout.write(`\n=== pdf=${pdfKey} concurrency=${c} ===\n`);
      try {
        const r = await runOne(browser, pdfKey, c);
        console.log(`  elapsed=${(r.elapsedMs / 1000).toFixed(2)}s success=${r.success} outLen=${r.finalOutLen} pages=${r.pagesInOutput.length} ordered=${r.orderedOK}`);
        console.log(`  final: ${r.finalProg.slice(0, 200)}`);
        if (r.errors.length) console.log('  errors:', r.errors.join(' | '));
        console.log('  growth (t_ms:pages):', r.growth.map(g => `${g.t}:${g.r}`).join(' '));
        results.push(r);
      } catch (e) {
        console.error('  ERROR:', e.message);
        results.push({ pdfKey, concurrency: c, err: e.message });
      }
    }
  }

  console.log('\n\n=== SUMMARY ===');
  console.log('pdf     conc  elapsed  pages  ordered');
  for (const r of results) {
    if (!r.elapsedMs) { console.log(`${r.pdfKey.padEnd(7)} ${String(r.concurrency).padEnd(5)} ERR: ${r.err}`); continue; }
    console.log(`${r.pdfKey.padEnd(7)} ${String(r.concurrency).padEnd(5)} ${(r.elapsedMs / 1000).toFixed(2)}s   ${String(r.pagesInOutput.length).padEnd(5)}  ${r.orderedOK}`);
  }

  await browser.close();
})();
