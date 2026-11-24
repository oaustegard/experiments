// toPaddleResult (web/pipeline.js) on a fixed input: field names, shapes, ordering, clipping, score threshold.
// No GPU, no network.   node web/test/check_paddle_result.mjs      (exit 0 when every check passes)
import assert from "node:assert/strict";
import { toPaddleResult } from "../pipeline.js";

const lines = [
  { box: [[10, 5], [110, 5], [110, 25], [10, 25]], text: "Hello", score: 0.98, detScore: 0.9 },
  { box: [[12, 40], [300.4, 41], [299, 63], [11, 62]], text: "second line", score: 0.4, detScore: 0.8 },
  { box: [[-3, 90], [205, 90], [205, 130], [-3, 131]], text: "edge", score: 0.7, detScore: 0.7 },
];
const r = toPaddleResult(lines, { width: 200, height: 128 });
assert.deepEqual(Object.keys(r), ["dt_polys", "rec_texts", "rec_scores", "rec_polys", "rec_boxes"]);
assert.deepEqual(r.rec_texts, ["Hello", "second line", "edge"]);
assert.deepEqual(r.rec_scores, [0.98, 0.4, 0.7]);
assert.deepEqual(r.dt_polys[0], [[10, 5], [110, 5], [110, 25], [10, 25]]);
assert.deepEqual(r.dt_polys[1], [[12, 40], [200, 41], [200, 63], [11, 62]]); // 300.4 and 299 clipped to the width, 40.x rounded
assert.deepEqual(r.dt_polys[2], [[0, 90], [200, 90], [200, 128], [0, 128]]); // clipped on every side
assert.deepEqual(r.rec_polys, r.dt_polys);
assert.deepEqual(r.rec_boxes, [[10, 5, 110, 25], [11, 40, 200, 63], [0, 90, 200, 128]]);
for (const p of r.dt_polys) { assert.equal(p.length, 4); for (const q of p) assert.ok(q.length === 2 && q.every(Number.isInteger)); }
for (const b of r.rec_boxes) assert.ok(b.length === 4 && b[0] <= b[2] && b[1] <= b[3]);

// text_rec_score_thresh: dt_polys keeps every box, rec_* drop the low line, in the same order
const t = toPaddleResult(lines, { width: 200, height: 128, minScore: 0.5 });
assert.equal(t.dt_polys.length, 3);
assert.deepEqual(t.rec_texts, ["Hello", "edge"]);
assert.deepEqual(t.rec_scores, [0.98, 0.7]);
assert.deepEqual(t.rec_polys, [r.dt_polys[0], r.dt_polys[2]]);
assert.deepEqual(t.rec_boxes, [[10, 5, 110, 25], [0, 90, 200, 128]]);

// no clipping without a size
assert.deepEqual(toPaddleResult([lines[1]]).dt_polys[0][1], [300, 41]);

// empty page: empty arrays (PaddleOCR gives an empty rec_boxes too)
assert.deepEqual(toPaddleResult([], { width: 10, height: 10 }), { dt_polys: [], rec_texts: [], rec_scores: [], rec_polys: [], rec_boxes: [] });
console.log("toPaddleResult: ok");
