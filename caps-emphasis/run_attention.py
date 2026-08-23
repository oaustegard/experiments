"""Q4: does attention mass on the emphasised span scale with its token count or
with its case?"""
import sys, os, json
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import caps_lib as C
from conditions import KEYWORD_FRAMES, MATCHED

# span text to locate inside the rendered prompt, per condition
SPANS = {}
for key, lo, up, kw in KEYWORD_FRAMES:
    SPANS[f"{key}_lower"] = (lo, kw)
    SPANS[f"{key}_caps"] = (up, kw.upper() if key != "forbidden" else "FORBIDDEN")
for name, tmpl, span in MATCHED:
    SPANS[name] = (tmpl, span)


def main():
    items = json.load(open(os.path.join(HERE, "items_screened.json")))["kept"][:12]
    out = {}
    for cname, (tmpl, span) in SPANS.items():
        per_layer_tot, per_layer_pt, lens, misses = [], [], [], 0
        for it in items:
            user = tmpl.format(W=it["word"]) + " " + it["question"]
            prompt = C.build_prompt(user, it["prefix"])
            r = C.attention_to_span(prompt, span)
            if r is None:
                misses += 1
                continue
            tot, pt, n = r
            per_layer_tot.append(tot)
            per_layer_pt.append(pt)
            lens.append(n)
        if not per_layer_tot:
            out[cname] = dict(error="span never resolved", misses=misses)
            continue
        L = len(per_layer_tot[0])
        out[cname] = dict(
            span=span, span_tokens=C.mean(lens), n_items=len(lens), misses=misses,
            layer_total=[C.mean(r[l] for r in per_layer_tot) for l in range(L)],
            layer_per_token=[C.mean(r[l] for r in per_layer_pt) for l in range(L)],
            total_all_layers=C.mean(sum(r) for r in per_layer_tot),
            per_token_all_layers=C.mean(sum(r) for r in per_layer_pt),
        )
        print(f"  {cname:16} span={span[:26]!r:30} ntok={C.mean(lens):4.1f} "
              f"sum_mass={out[cname]['total_all_layers']:7.3f} "
              f"per_tok={out[cname]['per_token_all_layers']:7.3f}")
    C.dump(os.path.join(HERE, "attention.json"), out)


if __name__ == "__main__":
    main()
