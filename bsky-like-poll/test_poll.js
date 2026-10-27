// Drives index.html in headless Chromium with the network mocked: signs in,
// builds a 3-option poll, checks the records written to the PDS, then counts
// results on a mocked thread where one account liked two options and one
// option's likes span two pages. Run: node test_poll.js
const { chromium } = require('playwright');
const path = require('path');
const assert = require('assert');

const DID = 'did:plc:testuser000000000000000';
const PDS = 'https://pds.example.com';
const json = (route, body, status = 200) => route.fulfill({ status, contentType: 'application/json', body: JSON.stringify(body) });

(async () => {
  const browser = await chromium.launch();
  const page = await browser.newPage();
  const writes = [];
  let n = 0;
  await page.route('**/*', async route => {
    const url = new URL(route.request().url());
    const nsid = url.pathname.split('/').pop();
    if (url.protocol === 'file:') return route.continue();
    if (nsid === 'com.atproto.identity.resolveHandle') return json(route, { did: DID });
    if (url.host === 'plc.directory') return json(route, { id: DID, service: [{ id: '#atproto_pds', type: 'AtprotoPersonalDataServer', serviceEndpoint: PDS }] });
    if (nsid === 'com.atproto.server.createSession') {
      const b = JSON.parse(route.request().postData());
      assert.equal(b.identifier, DID); assert.equal(b.password, 'abcd-efgh-ijkl-mnop');
      return json(route, { did: DID, handle: 'tester.bsky.social', accessJwt: 'A', refreshJwt: 'R' });
    }
    if (nsid === 'com.atproto.repo.createRecord') {
      assert.equal(route.request().headers().authorization, 'Bearer A');
      const b = JSON.parse(route.request().postData());
      writes.push(b);
      const rkey = b.rkey || `rk${n++}`;
      return json(route, { uri: `at://${DID}/${b.collection}/${rkey}`, cid: `cid-${rkey}` });
    }
    if (nsid === 'app.bsky.feed.getPostThread') {
      const mk = (rkey, text, replies = []) => ({ post: { uri: `at://${DID}/app.bsky.feed.post/${rkey}`, author: { did: DID }, record: { text } }, replies });
      const other = { post: { uri: 'at://did:plc:other/app.bsky.feed.post/x', author: { did: 'did:plc:other' }, record: { text: '1️⃣ fake' } }, replies: [] };
      const t = mk('rk0', 'Lunch?', [other, mk('rk1', '1️⃣ Tacos\n\n♥ like to vote', [mk('rk2', '2️⃣ Pho\n\n♥ like to vote', [mk('rk3', '3️⃣ Salad\n\n♥ like to vote')])])]);
      return json(route, { thread: t });
    }
    if (nsid === 'app.bsky.feed.getLikes') {
      const uri = url.searchParams.get('uri'), cursor = url.searchParams.get('cursor');
      const L = ds => ds.map(d => ({ actor: { did: d } }));
      if (uri.endsWith('rk1')) return cursor ? json(route, { likes: L(['c', 'd']) }) : json(route, { likes: L(['a', 'b']), cursor: 'p2' });
      if (uri.endsWith('rk2')) return json(route, { likes: L(['a', 'e']) });
      return json(route, { likes: [] });
    }
    throw new Error('unexpected request ' + url);
  });
  page.on('pageerror', e => { console.error('pageerror', e); process.exitCode = 1; });

  await page.goto('file://' + path.join(__dirname, 'index.html'));
  await page.fill('#handle', '@Tester');
  await page.fill('#password', 'abcd-efgh-ijkl-mnop');
  await page.click('#loginBtn');
  await page.waitForSelector('#builder:not(.hidden)');
  assert.equal(await page.textContent('#whoami'), '@tester.bsky.social');

  assert(await page.isDisabled('#postBtn'), 'post disabled while empty');
  await page.fill('#question', 'Lunch?');
  const ins = await page.$$('#opts input');
  await ins[0].fill('Tacos'); await ins[1].fill('Pho');
  await page.click('#addOpt');
  await (await page.$$('#opts input'))[2].fill('Salad');
  assert.equal(await page.$$eval('#preview .post', ps => ps.length), 4);
  assert(!(await page.isDisabled('#postBtn')));
  await page.screenshot({ path: path.join(__dirname, 'shot-builder.png'), fullPage: true });

  await page.click('#postBtn');
  await page.waitForSelector('#postMsg.ok');
  assert.equal(writes.length, 5, 'root + 3 options + threadgate');
  const [root, o1, o2, o3, gate] = writes;
  assert.equal(root.collection, 'app.bsky.feed.post');
  assert(root.record.text.startsWith('Lunch?\n\n🗳️ Poll'));
  assert(!root.record.reply);
  assert.equal(o1.record.text, '1️⃣ Tacos\n\n♥ like to vote');
  assert.deepEqual(o1.record.reply, { root: { uri: `at://${DID}/app.bsky.feed.post/rk0`, cid: 'cid-rk0' }, parent: { uri: `at://${DID}/app.bsky.feed.post/rk0`, cid: 'cid-rk0' } });
  assert.equal(o3.record.reply.parent.uri, `at://${DID}/app.bsky.feed.post/rk2`, 'options chain');
  assert.equal(o3.record.reply.root.uri, `at://${DID}/app.bsky.feed.post/rk0`);
  assert.equal(gate.collection, 'app.bsky.feed.threadgate');
  assert.equal(gate.rkey, 'rk0');
  assert.deepEqual(gate.record.allow, []);
  assert.equal(await page.inputValue('#pollUrl'), 'https://bsky.app/profile/tester.bsky.social/post/rk0');

  await page.click('#resultsForm button');
  await page.waitForSelector('#resultsOut table');
  const rows = await page.$$eval('#resultsOut tbody tr', trs => trs.map(tr => [...tr.cells].map(c => c.textContent.trim().replace(/\s+/g, ' '))));
  console.log(rows, await page.textContent('#resultsMsg'));
  assert.deepEqual(rows.map(r => r[2]), ['4', '2', '0'], 'raw likes incl. page 2');
  assert.deepEqual(rows.map(r => r[1].split(' ')[0]), ['3', '1', '0'], 'multi-liker a dropped');
  assert(rows[0][0].includes('Tacos'));
  assert.match(await page.textContent('#resultsMsg'), /4 votes from 5 accounts\. 1 account liked more than one option/);
  await page.screenshot({ path: path.join(__dirname, 'shot-results.png'), fullPage: true });
  await browser.close();
  console.log(process.exitCode ? 'FAIL' : 'PASS');
})().catch(e => { console.error(e); process.exit(1); });
