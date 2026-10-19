// Correctness gate: the model card's text-search example must reproduce the card's printed scores
// (the same check embeddinggemma2-webgpu-tail/run/ref_real.py runs in Python), and a 512-dim
// prefix must come back unit length. Exit 1 on any miss.
process.env.NO_LISTEN = '1';
const { embed, ready } = (await import('./server.cjs')).default;
await ready;

const QUERY = 'Which planet is known as the Red Planet?';
const DOCS = [
  "Venus is often called Earth's twin because of its similar size and proximity.",
  'Mars, known for its reddish appearance, is often referred to as the Red Planet.',
  'Jupiter, the largest planet in our solar system, has a prominent red spot.',
  'Saturn, famous for its rings, is sometimes mistaken for the Red Planet.',
];
const CARD = [0.684, 0.854, 0.752, 0.783];
const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);

const q = (await embed(QUERY, 'query')).embedding;
const scores = [];
for (const d of DOCS) scores.push(dot(q, (await embed(d, 'document')).embedding));
const cardOk = scores.every((s, i) => Math.abs(s - CARD[i]) <= 6e-4);

const q512 = (await embed(QUERY, 'query', 512)).embedding;
const norm512 = Math.sqrt(dot(q512, q512));
const normOk = q512.length === 512 && Math.abs(norm512 - 1) < 1e-5;

console.log(JSON.stringify({ scores: scores.map((s) => +s.toFixed(4)), card: CARD, cardOk, norm512, normOk }));
process.exit(cardOk && normOk ? 0 : 1);
