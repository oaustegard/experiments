// node run/gpu_check.mjs -> results/gpu_check.json: storage-buffer shader, baked shader and the CPU mirror agree
import fs from "node:fs";
import { open } from "../../run/browser.mjs";
const { page, close } = await open("texture-collage/splat/run/gpu_check.html");
const r = await page.evaluate(() => window.run());
console.log(JSON.stringify(r, null, 1));
fs.writeFileSync("results/gpu_check.json", JSON.stringify(r, null, 1));
await close();
