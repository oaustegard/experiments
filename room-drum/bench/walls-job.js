// GPU Bridge job: the CPU side of the page on the same machine, in the bridge's Worker. walls.js arrives as an
// input string. A synthetic 640 x 480 frame (moving discs and bars) stands in for the camera, since the worker has
// none. Times Walls.frame (draw, luma, Sobel, hysteresis, region labels) and the field overlay draw per grid.
const { src, grids = [], frames = 30 } = ctx.inputs.meta;
const { Walls } = await import(URL.createObjectURL(new Blob([src], { type: "text/javascript" })));
const cam = new OffscreenCanvas(640, 480), cg = cam.getContext("2d");
function paint(t) {
  cg.fillStyle = "#777"; cg.fillRect(0, 0, 640, 480);
  cg.fillStyle = "#222"; cg.beginPath(); cg.arc(220 + 60 * Math.sin(t / 5), 240, 120, 0, 2 * Math.PI); cg.fill();
  cg.fillStyle = "#ddd"; cg.fillRect(420, 60 + 10 * (t % 7), 40, 300); cg.fillRect(80, 400, 480, 18);
}
const view = new OffscreenCanvas(1280, 960), vg = view.getContext("2d");
const rows = [];
for (const [gi, [nx, ny]] of grids.entries()) {
  ctx.progress(gi / grids.length, `${nx}x${ny}`);
  const W = new Walls(nx, ny), ms = [], drawMs = [];
  const ov = new OffscreenCanvas(nx, ny), og = ov.getContext("2d"), id = og.createImageData(nx, ny);
  for (let f = 0; f < frames; f++) {
    paint(f);
    let t = performance.now(); W.frame(cam, { src: "edge", thr: 0.5 }); ms.push(performance.now() - t);
    t = performance.now();
    const d = id.data; for (let i = 0; i < nx * ny; i++) { d[4 * i] = W.wall[i] * 230; d[4 * i + 3] = 150; }
    og.putImageData(id, 0, 0); vg.drawImage(ov, 0, 0, 1280, 960); drawMs.push(performance.now() - t);
  }
  ms.sort((a, b) => a - b); drawMs.sort((a, b) => a - b);
  const r = { nx, ny, cells: nx * ny, wallsMedianMs: +ms[frames >> 1].toFixed(2), wallsP90Ms: +ms[Math.floor(frames * 0.9)].toFixed(2),
    drawMedianMs: +drawMs[frames >> 1].toFixed(2), walls: W.wall.reduce((a, b) => a + b, 0), regions: W.sizes.length };
  rows.push(r); ctx.log(JSON.stringify(r));
}
return { rows, frames };
