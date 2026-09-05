# A numeric select for RASP

One attention head of head_dim 2 gives RASP a `select_at(addr)` primitive that
attends to the position whose index equals a computed integer, with no one-hot
position table anywhere in the weights. Three programs that a categorical
compiler reaches only through such a table compile to 11, 13 and 19 residual
dimensions here, and the same weights, compiled without ever being shown a
length, run correctly at n = 3 and at n = 1000.

## The primitive

`rasp_ns.py` carries a RASP subset — `tokens`, `indices`, `map`,
`sequence_map`, categorical `select`/`aggregate` — plus `select_at(addr)`,
where `addr` is a numeric s-op giving each query position an integer target.
`aggregate(select_at(addr), v)` returns `v[addr[i]]`, or 0 out of range.

`compile_ns.py` turns such a program into explicit numpy matrices for a
standard attention-plus-MLP transformer. Position j writes the key
`(2j, -j²)`; the query for address x is `(x, 1)`; the score is
`2xj - j² = x² - (j - x)²`. This key geometry comes from Percepta (Tzamos et
al., *Can LLMs Be Computers?*, percepta.ai, March 2026, repo
`Percepta-Core/transformer-vm`); the `x² - (j - x)²` rewrite and the gap
statement below come from Oskar Austegard's LAC line
(`oaustegard/llm-as-computer`), where `torchlean-lac/` proved the same
identity in Lean. Out-of-range reads are zeroed by an MLP range gate,
`v = relu(1 - relu(-x) - relu(x - L + 1))`, with L recovered from a
uniform-attention head whose mean over `indices` is `(L-1)/2`. For comparison
the compiler also emits Tracr-style categorical selects, with one-hot key and
query subspaces, a predicate matrix, and a BOS column carrying the default.

## Three programs

`programs.py` builds `y[i] = x[2i + 1]`, `y[i] = x[x[i]]` and
`y[i] = x[i - x[0]]`. Compiled models match the interpreter exactly (max abs
error 0.0) at n = 8, 32, 128, and again at 3, 257 and 1000.

The same gather written with a categorical select over positions needs
`categorize` on both the index and the address. That build costs 55 residual
dimensions and 6,760 parameters at n_max = 8 and 391 dimensions and 354,632
parameters at n_max = 64, and it returns wrong answers for every n above its
n_max. The arithmetic build costs 13 dimensions and 425 parameters and carries
no n_max at all.

## Measured margin and beta thresholds

`margin.py` checks every integer address at n = 8, 32, 128 and 512. The
winner/runner-up score gap is exactly 1.0 in all cases. Sweeping
`softmax(beta · score)` against average-hard attention on a 0.25 grid gives the
smallest beta at which softmax agrees with hard argmax to within 0.5.

| program | n = 8 | n = 32 | n = 128 |
|---|---|---|---|
| `y[i] = x[2i + 1]` | 3.50 | 3.50 | 3.75 |
| `y[i] = x[x[i]]` | 3.00 | 4.75 | 6.25 |
| `y[i] = x[i - x[0]]` | 9.25 | 9.00 | 9.50 |

Length raises the threshold only where the magnitude of the read values rises
with it. The pointer chase reads a token drawn from `0..n-1` and climbs from
3.00 to 6.25; the other two read a fixed 0-9 alphabet and stay flat within one
grid step over a 16x change in length. The shift program pays about 6 extra
units of beta for its range gate, whose bound M multiplies any error in the 0/1
flag. At n = 32 the thresholds run 6.75, 9.00, 11.50 and 16.00 for M of 10²,
10³, 10⁴ and 10⁶. `gap_vs_beta.png` plots the error curves.

## Delta against prior art

RASP (Weiss, Goldberg, Yahav 2021) and Tracr (Lindner et al., arXiv 2301.05062)
define `select` as a predicate over categorical keys and queries and compile it
to one-hot subspaces, so a computed position costs one dimension per reachable
position. ALTA (arXiv 2410.18077) compiles the same categorical shape. B-RASP
with positions (Strobl et al., TACL 2025, arXiv 2404.02040) reaches a computed
integer address through an equality predicate over bounded integer vectors
under average-hard attention. RASP-L (arXiv 2310.16028) rules index arithmetic
out of the language. Giannou et al. (arXiv 2301.13196, Lemma 2) obtain margin 1
from ±1 binary position codes in log n dimensions. The addition here is a
select whose predicate is arithmetic rather than symbolic: two key dimensions
regardless of length, a gap of exactly 1, and a compiled program with no
compile-time length bound.

## Limits

A numpy toy with hand-built weights and no learning. One head, integer
addresses only, since the range gate's ReLU indicator is exact on integers and
interpolates otherwise. Elementwise maps compile only when they fit an affine
form on an integer probe grid. Aggregate values stay numerical. Float32 weights read exactly at
n = 128 and hit a precision ceiling once `j²` passes `2^24`, measured and
proved in `torchlean-lac/`: exact through address 4096 and wrong from 4097.
