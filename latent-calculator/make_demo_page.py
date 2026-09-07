"""Render the side-by-side demo page from results/*.json.

    python3 make_demo_page.py [--model smol] [--demo results/demo_smol.json]
                              [--stream-arm stream] [--out demo_body.html]

Then build with composing-html:
    python3 /mnt/skills/user/composing-html/scripts/build.py build freeform \
        --set title='Latent Calculator' --set body_html=@demo_body.html \
        --out demo.html
"""
import argparse
import html
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
R = os.path.join(HERE, "results")

MODEL_NAMES = {"smol": "SmolLM2-135M", "monad": "Pleias Monad 56.7M"}


def load(name):
    with open(os.path.join(R, name)) as f:
        return json.load(f)


def esc(s):
    return html.escape(str(s))


def badge(ok):
    return ('<span class="badge badge--ok">right</span>' if ok
            else '<span class="badge badge--err">wrong</span>')


def pct(x):
    return f"{100 * x:.0f}%"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="smol")
    ap.add_argument("--demo", default=None)
    ap.add_argument("--stream-arm", default="stream-left")
    ap.add_argument("--out", default=os.path.join(HERE, "demo_body.html"))
    a = ap.parse_args()
    m = a.model
    demo = json.load(open(a.demo or os.path.join(R, f"demo_{m}.json")))
    learned = load(f"{m}_{a.stream_arm}_learned_attn.json")
    oracle = load(f"{m}_{a.stream_arm}_oracle.json")
    single = load(f"{m}_residual_oracle.json")
    none = load(f"{m}_none_oracle.json")
    text = load(f"{m}_text_oracle.json")
    qh = load(f"query_head_{m}_attn.json")
    qh_mlp = load(f"query_head_{m}.json")
    probe = load(f"probe_{m}.json")
    k = learned["k"]
    name = MODEL_NAMES.get(m, m)
    li, l5 = learned["splits"]["test_in"], learned["splits"]["test_len5"]
    oi = oracle["splits"]["test_in"]
    ni, ti = none["splits"]["test_in"], text["splits"]["test_in"]
    summ = demo["summary"]
    n = demo["n"]

    out = []
    w = out.append

    w('<section>')
    w('<p>A frozen language model, a calculator, and no tokens between them. '
      f'{esc(name)} never sees the tool: a small trained head reads the '
      f'operands out of its layer-{k} activations, a Python calculator does the '
      'arithmetic, and a small trained encoder writes the result back into the '
      f'residual stream after layer {k}, once per answer step. Every model weight '
      'is frozen. Nothing is appended to the prompt. The model then answers as '
      'if it knew.</p>')
    w('<p>Below: the same twelve prompts through the frozen model alone, through '
      'the usual text route (the result pasted into the prompt as tokens), and '
      'through the latent port. Then the numbers over 2,000 prompts.</p>')
    w('</section>')

    # stat row
    w('<section><div class="eyebrow">Twelve prompts, three routes</div>')
    w('<div class="grid grid--3">')
    for arm, label, sub in [("none", "Frozen model", "no tool"),
                            ("text", "Tool result as text", "pasted into the prompt"),
                            ("stream", "Latent port", "injected after layer %d" % k)]:
        s = summ[arm]
        w(f'<div class="card"><div class="eyebrow">{esc(label)}</div>'
          f'<div class="bignum">'
          f'{int(round(s["accuracy"] * n))}<span style="color:var(--g500)"> / {n}</span></div>'
          f'<div class="muted">{esc(sub)} · {s["mean_tokens"]:.0f} tokens · '
          f'{s["mean_ms"]:.0f} ms per answer</div></div>')
    w('</div></section>')

    # rows
    w('<section><div class="eyebrow">Side by side</div>')
    w('<p>Random rows from the in-distribution test split (operand lengths 1-4 '
      'and 6 digits). "Query" is what the trained head read out of the '
      'activations; the calculator ran on exactly that.</p>')
    w('<div style="overflow-x:auto"><table>')
    w('<thead><tr><th>Prompt</th><th>Answer</th><th>Frozen model</th>'
      '<th>Tool as text</th><th>Latent port</th><th>Query read from layer %d</th></tr></thead><tbody>' % k)
    for r in demo["rows"]:
        arms = r["arms"]
        cells = []
        for arm in ("none", "text", "stream"):
            g = arms[arm]
            gen = g["gen"] if len(g["gen"]) <= 40 else g["gen"][:37] + "…"
            cells.append(f'<td><code>{esc(gen)}</code> {badge(g["correct"])}</td>')
        q = r.get("query") or ""
        calc = r.get("calculator") or ""
        w(f'<tr><td>{esc(r["prompt"])}</td><td><code>{esc(r["gold"])}</code></td>'
          + "".join(cells)
          + f'<td><code>{esc(q)}</code> → <code>{esc(calc)}</code></td></tr>')
    w('</tbody></table></div></section>')

    # accuracy tables
    w('<section><div class="eyebrow">Over 2,000 prompts</div>')
    w('<p>Exact match of the generated answer. In distribution means unseen '
      'operand values at training lengths; held-out length means at least one '
      '5-digit operand, a length never trained on.</p>')
    w('<div class="grid grid--2">')
    w('<div><table><thead><tr><th>Route</th><th>In distribution</th><th>Held-out length</th>'
      '<th>Tool tokens</th><th>ms / answer</th></tr></thead><tbody>')
    rows = [
        ("Frozen model", none, ""),
        ("Tool result as text (answer anywhere in output)", text, "contains"),
        ("Single vector, oracle operands (phase 1)", single, ""),
        ("Latent port, oracle operands", oracle, ""),
        ("<strong>Latent port, learned query</strong>", learned, ""),
    ]
    for label, res, mode in rows:
        s_in, s_5 = res["splits"]["test_in"], res["splits"]["test_len5"]
        key = "contains_match" if mode == "contains" else "exact_match"
        w(f'<tr><td>{label}</td><td>{pct(s_in[key])}</td><td>{pct(s_5[key])}</td>'
          f'<td>{s_in["tool_tokens"]:.1f}</td><td>{res["cpu_ms_per_answer_bs1"]:.0f}</td></tr>')
    w('</tbody></table></div>')
    w('<div><table><thead><tr><th>Learned query, by operator</th><th>In distribution</th>'
      '<th>Held-out length</th></tr></thead><tbody>')
    for op in ("add", "sub", "mul", "cmp"):
        w(f'<tr><td><code>{op}</code></td><td>{pct(li["by_op"][op]["acc"])}</td>'
          f'<td>{pct(l5["by_op"][op]["acc"])}</td></tr>')
    w('</tbody></table>')
    w('<table><thead><tr><th>By longest operand</th>' +
      "".join(f'<th>{L} digits</th>' for L in sorted(li["by_max_len"], key=int)) +
      '</tr></thead><tbody><tr><td>in distribution</td>' +
      "".join(f'<td>{pct(li["by_max_len"][L]["acc"])}</td>' for L in sorted(li["by_max_len"], key=int)) +
      '</tr></tbody></table></div>')
    w('</div></section>')

    # how it works
    w('<section><div class="eyebrow">How the port works</div><ul class="bullets">')
    w(f'<li><strong>Asking.</strong> A cross-attention head with 13 learned queries reads '
      f'the operator and twelve digit slots out of the layer-{k} activations of every prompt '
      f'token ({qh["params"] / 1e3:.0f}k parameters). It recovers the full query on '
      f'{pct(qh["test_in"]["exact"])} of in-distribution prompts. Reading one vector at the '
      f'last token, the phase-1 design, managed {pct(qh_mlp["test_in"]["exact"])}: the digits '
      'live at their own tokens, not at the question mark.</li>')
    w('<li><strong>Computing.</strong> Plain Python on the decoded query. Exact.</li>')
    w(f'<li><strong>Reading.</strong> An encoder turns the result (sign, digits, or a '
      'comparison word) plus the answer-step index into one vector of the model\'s hidden '
      f'size and adds it to the residual stream after layer {k}, at the query position and '
      'again at every answer step. The digits are stored most-significant first, so answer step j '
      'reads slot j. The frozen upper layers turn that into the next token. '
      f'Trained by the model\'s own next-token loss; {pct(oi["exact_match"])} exact with '
      'true operands.</li>')
    w('<li><strong>Cost.</strong> Zero tokens in either direction. The port\'s own compute is '
      'a few matrix products per step; the wall-clock difference in the tables is mostly '
      'that the frozen model rambles for 16 tokens and the ported one stops after the '
      'answer.</li>')
    w('</ul></section>')

    # failures
    w('<section><div class="eyebrow">Where it fails</div><ul class="bullets">')
    w(f'<li>Twelve-digit products. Multiplying two 6-digit numbers lands at '
      f'{pct(li["by_op"]["mul"]["acc"])}; the wrong answers have the leading digits right and '
      'the last few off. Addition and subtraction, whose results stay under eight digits, are at '
      f'{pct(li["by_op"]["add"]["acc"])} and {pct(li["by_op"]["sub"]["acc"])}.</li>')
    w(f'<li>Lengths it never saw. On prompts with a 5-digit operand the reading side holds up '
      f'({pct(oracle["splits"]["test_len5"]["exact_match"])} with true operands) but the query head '
      f'is right only {pct(l5["calculator_exact"])} of the time, so the whole pipeline lands at '
      f'{pct(l5["exact_match"])}. The head counts digit positions from the end of the prompt, and '
      'an unseen operand length shifts the count.</li>')
    w('<li>The text route scores zero on exact match here because a 135M base model wraps '
      'the pasted result in prose or repeats it; the "contains" column is its fair number.</li>')
    w('</ul></section>')

    # colophon-ish
    w('<section><p class="muted">'
      f'{esc(name)}, frozen, fp32, 4 CPU cores. Trained parts: query head '
      f'{qh["params"] / 1e3:.0f}k, result encoder about 1M. Port at layer {k} of '
      f'{len(probe["layers"]) - 1}, chosen by a linear-probe sweep. 20,000 synthetic training '
      'prompts over twelve templates. Code, predictions and results: '
      '<code>oaustegard/experiments/latent-calculator</code>.</p></section>')

    with open(a.out, "w") as f:
        f.write("\n".join(out))
    print(f"wrote {a.out} ({sum(len(x) for x in out)} chars)")


if __name__ == "__main__":
    main()
