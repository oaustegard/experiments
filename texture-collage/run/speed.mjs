import { open } from "./browser.mjs";
const { page, close } = await open("texture-collage/run/_bank.html");
for (const [S, steps] of [[64, 12]]) {
  const r = await page.evaluate(([S, steps]) => window.gen(23, S, steps, 1), [S, steps]);
  console.log(S, steps, r.ms.toFixed(0), "ms");
}
await close();
