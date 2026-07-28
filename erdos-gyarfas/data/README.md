# Data provenance

Two of these files are Klas Markström's, downloaded from his site; two are this
session's output; one is a graph found here. They are kept side by side because
the point of the `*_mine` files is that they agree with the `*_markstrom` ones.

| file | origin |
|---|---|
| `order24_markstrom.g6` | [`cubic_no4_8_n24.g6`](http://abel.math.umu.se/~klasm/Data/cubicavoidcycle/cubic_no4_8_n24.g6) — Markström |
| `order26_markstrom.g6` | [`cubic_no4_8_n26.g6`](http://abel.math.umu.se/~klasm/Data/cubicavoidcycle/cubic_no4_8_n26.g6) — Markström |
| `order24_mine.g6` | this session, `../reproduce.sh 24` |
| `order26_mine.g6` | this session, `../reproduce.sh 26` |
| `near_miss_order26.g6` | this session — cubic, spectrum `{3,5,6,7,9,…,26}`, misses exactly 4 and 8 |

Markström's index page is
[`Data/cubicavoid.html`](http://abel.math.umu.se/~klasm/Data/cubicavoid.html).
Note the **plain HTTP** — the host does not serve HTTPS, which is why tools that
silently upgrade the scheme cannot fetch any of it. That page is also the only
citable source for `f(4) ≥ 54`:

> "I have looked for cubic graphs with no cycles of lengths 4,8,16. If have found
> no such graphs and have searched all N<=52"

which Exoo (arXiv:1403.5636) could only cite as "an unpublished result of
Markström". His files hold **3-connected** cubic graphs; the counts here are over
all connected cubic graphs, and at orders 24 and 26 the two coincide.

## Verifying the agreement

Canonical labelling, so isomorphic graphs compare equal:

```sh
nauty-labelg -q < order26_markstrom.g6 | sort > /tmp/theirs
nauty-labelg -q < order26_mine.g6      | sort > /tmp/mine
diff /tmp/theirs /tmp/mine && echo "identical"
```

Result at both orders: 4/4 and 23/23, zero differences either way.
