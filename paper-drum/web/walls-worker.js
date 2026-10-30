// Runs walls.js off the main thread, so finding the walls never delays the audio blocks. The page posts an
// ImageBitmap of the frame (already drawn upright through a 2D canvas) with the settings; the worker answers with
// copies of the wall bits, the mask, the region labels, the pads and the per-cell motion.
import { Walls } from "./walls.js";
let W = null;
onmessage = ({ data: { nx, ny, bitmap, opts } }) => {
  const t0 = performance.now();
  if (!W || W.nx !== nx || W.ny !== ny) W = new Walls(nx, ny);
  const changed = W.frame(bitmap, opts);
  bitmap.close();
  const out = { nx, ny, changed, motion: W.motion.slice(), sizes: W.sizes, pads: W.pads, edge: W.edge || [], ms: performance.now() - t0 };
  if (changed) Object.assign(out, { wall: W.wall.slice(), mask: W.mask.slice(), label: W.label.slice() });
  postMessage(out, [out.motion.buffer, ...(changed ? [out.wall.buffer, out.mask.buffer, out.label.buffer] : [])]);
};
