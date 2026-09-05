## Motivation

Odrzywolek ("All elementary functions from a single operator", arXiv:2603.21852, 2026) shows that the binary operator $\mathrm{eml}(x, y) = e^{x} - \ln y$ together with the constant $1$ generates every elementary function: $e = \mathrm{eml}(1,1)$, $e^{x} = \mathrm{eml}(x, 1)$, $\ln x = \mathrm{eml}(1, \mathrm{eml}(\mathrm{eml}(1, x), 1))$, and from these subtraction, negation, multiplication and powers. A closed tree over this grammar names a real constant, and the paper's Table 4 records how many nodes each standard function costs. The same question for constants, how many $\mathrm{eml}$ nodes the number $2$, or $\ln 2$, or $\pi$ needs, is the analogue of Mahler and Popken's integer complexity ($\|n\|$, the fewest $1$s needed to write $n$ with $+$ and $\times$; OEIS A005245), and it is where the operator's cost model becomes measurable.

Symbolic-regression engines built on this operator (`oaustegard/eml-sr`) must construct every constant they use from the leaf $1$, so the minimal sizes bound what a tree of a given depth can express. Exhaustive enumeration on the real branch to size 20 (327,841,920 distinct values) and on the complex principal branch to size 18 has produced a table of minimal sizes and witnesses (eml-sr, `benchmarks/eml_complexity.md`, 2026-09-04). Every witness in that table was checked numerically at 40 digits and symbolically; none of it is machine-checked, and the lower bounds rest on the correctness of a numerical enumeration with 11-digit observational dedupe.

## Setting

An **EML tree** is a full binary tree whose leaves are the constant $1$ and whose internal nodes are $\mathrm{eml}$. Writing $\mathbf{1}$ for the leaf and $\mathrm{node}(a, b)$ for an internal node with children $a, b$, its **evaluation** is

$$\mathrm{eval}(\mathbf{1}) = 1, \qquad \mathrm{eval}(\mathrm{node}(a, b)) = \exp(\mathrm{eval}\, a) - \ln(\mathrm{eval}\, b),$$

with the real exponential and the real logarithm, and its **size** is the number of internal nodes. A tree is **valid** when every logarithm it takes is of a positive real: at every node $\mathrm{node}(a, b)$, $\mathrm{eval}\, b > 0$, recursively. A real $c$ is **attained at size $n$** when some valid tree of size exactly $n$ evaluates to $c$; the **EML complexity** of $c$ is $n$ when $c$ is attained at $n$ and at no smaller size.

The enumeration behind the table finds, on the real branch, the following minimal sizes: $e$ at $1$, $e-1$ and $e^{e}$ at $2$, $0$ at $3$, $e-2$ at $7$, $-1$ at $8$, $2$ at $9$, $\ln 2$ at $12$, $3$ at $14$, $1/2$ at $17$, $-3$ at $20$, $4$ at $21$. Integer cost grows linearly at five nodes per unit through a ladder $e - (k+1) = \ln(\exp(e - k)) - 1$; the first witness that multiplies rather than subtracts appears on the complex branch at size 36.

## Formalization targets

### Witnesses

For each constant in the table, the statement that it is attained at its tabulated size:

$$\exists\, t,\ \mathrm{valid}(t) \wedge \mathrm{size}(t) = n \wedge \mathrm{eval}(t) = c.$$

These are the milestones `attains_e` through `attains_four`. Each is proved by exhibiting the enumeration's tree; the proofs are short and are already available.

### Lower bounds

$$\forall\, m < 21,\ \neg\big(\exists\, t,\ \mathrm{valid}(t) \wedge \mathrm{size}(t) = m \wedge \mathrm{eval}(t) = 4\big).$$

### Goal

$$\mathrm{Complexity}(4, 21): \quad \text{$4$ is attained at $21$ and at no smaller size.}$$

The goal is the cell the enumeration singled out: on the complex branch $4$ costs $19$ through $\exp(e - 4)$, but $e - 3 < 0$ has no real logarithm, so the real branch needs a different route, and $21$ is what it found. A weaker stable statement, $\mathrm{Complexity}(2, 9)$, is included as a milestone because its lower bound is a finite search over $2{,}056$ trees of size at most $8$.

## Significance

The witnesses turn a numerical table into checked facts: a real closed tree of $21$ nodes evaluates exactly to $4$, with every logarithm taken of a positive real. Downstream, the eml-sr compiler can be measured against these sizes (it currently overpays by a factor of $1.3$ to $3.7$ on arithmetic-built constants), and any symbolic-regression search over this grammar can read off which constants a given depth can and cannot express.

The lower bounds are where formalization adds something the enumeration cannot. The enumeration dedupes values to $11$ significant digits and rejects non-finite intermediates; a proof that no smaller tree exists must handle every tree of the smaller sizes exactly. That includes the trees the enumeration discarded as numerically degenerate. None of the table's rows has a machine-checked proof before this mission; the witness proofs exist as Lean files and will be submitted against the milestones.

## Difficulty

The witnesses are routine once the evaluation identities $\ln(e^{x}) = x$ and $e^{\ln x} = x$ (for $x > 0$) are applied bottom-up with the positivity facts supplied; the only arithmetic needed is $2 < e < 3$, bounds on $\ln 2$, and $7 < e^{e}$.

The lower bounds are not routine. The obvious approach, enumerate every valid tree of size below $n$ and check that none evaluates to $c$, runs into two problems. First, the evaluation is a real number, not a decidable object: showing $\mathrm{eval}(t) \neq 4$ for a specific $t$ requires an interval bound tight enough to separate it from $4$, and trees such as $e - \ln(\exp(e - \ln(\cdots)))$ evaluate to values within $10^{-11}$ of integers only by coincidence, so the bounds must be honest. Second, the number of trees of size $m$ is the Catalan number $C_m$, so size $20$ alone has $6.5 \times 10^{9}$ trees; a proof for $\mathrm{Complexity}(4, 21)$ needs either a structural argument that most trees cannot reach $4$ (for instance, that a valid tree's value determines a finite set of parent values, the inverse map $b = \exp(\exp a - c)$ used by the enumeration's joins) or a certified enumeration far beyond what `decide` handles. $\mathrm{Complexity}(2, 9)$ is the tractable first case.

## Formalization scope

The Lean development fixes the following conventions:

- Trees are the inductive type `EmlComplexity.Tree` with constructors `one` and `node`; `size` counts `node`s; `eval` uses `Real.exp` and `Real.log`.
- Mathlib's `Real.log` is total, with $\ln x = \ln|x|$ for $x \ne 0$ and $\ln 0 = 0$. A statement about `eval` alone would therefore admit trees that take logarithms of negative reals or of zero; `valid` rules them out, and every target quantifies over valid trees. A tree whose value coincides with the target only through junk logarithms does not count.
- The real branch only. The complex principal branch, where $\ln(-1) = i\pi$ and several constants are cheaper, is a separate development.
- `Attains c n` is existence of a valid tree of size exactly $n$; `Complexity c n` is `Attains c n` together with `∀ m < n, ¬ Attains c m`.

A complete development needs no library beyond `Mathlib.Analysis.SpecialFunctions.Log.Basic` for the definitions and `Mathlib.Analysis.Complex.ExponentialBounds` for the numeric facts. The definition file is reusable for any question about closed EML trees, and contributions of structural lemmas (the three-node logarithm, the five-node subtraction, monotonicity of size under these constructions) are welcome, as are witnesses for constants not in the current table.

## Selected references

- A. Odrzywolek, *All elementary functions from a single operator*, arXiv:2603.21852, 2026. https://arxiv.org/abs/2603.21852
- K. Mahler and J. Popken, *On a maximum problem in arithmetic*, Nieuw Archief voor Wiskunde (3) 1 (1953), 1–15.
- OEIS A005245, *Integer complexity: the number of 1's required to build n using + and ×*. https://oeis.org/A005245
- O. Austegard, *EML complexity of constants*, oaustegard/eml-sr, `benchmarks/eml_complexity.md`, 2026. https://github.com/oaustegard/eml-sr/blob/main/benchmarks/eml_complexity.md
