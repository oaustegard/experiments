// node run/check_state.mjs : web/state.js must reproduce run/state.py on every saved account
// (data/state_fixture.json, written by the snippet in RESULTS.md Method). Exits 1 on any difference.
import { readFileSync } from "node:fs";
import { accountState, postState } from "../web/state.js";
const fx = JSON.parse(readFileSync(new URL("../data/state_fixture.json", import.meta.url)));
let bad = 0, n = 0;
for (const f of fx) {
  for (const [k, s] of [["s12", accountState(f.a, 12)], ["s6", accountState(f.a, 6)]]) {
    n++; if (s !== f[k]) { if (bad++ < 3) console.log("DIFF", f.a.profile.handle, k, "\nPY:", JSON.stringify(f[k]).slice(0, 400), "\nJS:", JSON.stringify(s).slice(0, 400)); }
  }
  f.posts.forEach((p, i) => { n++; if (postState(f.a.items[i]) !== p) { if (bad++ < 3) console.log("POST DIFF", JSON.stringify(p), JSON.stringify(postState(f.a.items[i]))); } });
}
console.log(`${n - bad}/${n} identical`);
process.exit(bad ? 1 : 0);
