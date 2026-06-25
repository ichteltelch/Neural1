# `PolyInteraction`

Stateless utility that expands a feature vector into all monomial products (polynomial feature interactions) of a chosen range of orders, plus the matching back-propagation of gradients and the combinatorics needed to size the output.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

Given `n` input features `x₀…x_{n−1}`, polynomial interaction expansion produces every product of `order` factors drawn **with repetition** from those features — i.e. every degree-`order` monomial — for each order from `minOrder` to `maxOrder`. This lets a *linear* classifier fit nonlinear (polynomial) decision boundaries: you expand the inputs, then feed the expanded vector to logistic regression / a linear layer.

The class is `double`-typed (with `float` overloads also present in this file — see Gotchas) and purely static. It provides:
- `apply(…)` — the forward expansion.
- `diffApply(…)` — the reverse-mode gradient (back-propagation) of `apply`.
- `simplexNumber` / `simplexNumberSum` — the count of monomials, used to size output arrays and offsets.

Only *distinct* monomials are produced (combinations with repetition, not all ordered tuples): the recursion restricts each successive factor to indices `≤` the previous one, so `x₀x₁` is emitted once, not as both `x₀x₁` and `x₁x₀`.

## Output sizing — `simplexNumber(n, order)`, `simplexNumberSum(n, minOrder, maxOrder)`

The number of degree-`order` monomials in `n` variables is the "multiset coefficient" `C(n+order−1, order)` — a *simplex number*. `simplexNumber(n, order)` computes it (with fast paths for orders 0–3, then an incremental product with `gcd` reduction at each step to avoid `long` overflow). `simplexNumberSum` sums it over an order range, again accumulating as a reduced fraction (`numer/denom`, asserting `denom==1` at the end) so it stays exact for large `n`. There is a one-arg-max overload (`simplexNumberSum(n, maxOrder)`, implicitly from order 1) and a commented-out naive loop version. `main` cross-checks the two implementations for `n=27`, orders 1–3.

## Forward expansion — `apply(…)`

Range form: `apply(n, minOrder, maxOrder, in, inOffset, out, outOffset)` loops `order` from `minOrder` to `maxOrder`, appending each order's block to `out` and returning the total count (`= simplexNumberSum`).

Single-order form: `apply(n, order, in, inOffset, out, outOffset)` delegates to the private recursive `apply(…, multiplier)`:
- `order ≤ 0` → emit the single constant `multiplier` (the empty product), return 1.
- `order == 1` → emit `in[inOffset+i]·multiplier` for each `i`, return `n`.
- otherwise → for each `i` in `0…n−1`, recurse with `order−1`, `n` capped at `i+1`, and `multiplier·in[inOffset+i]`. Capping at `i+1` enforces non-increasing factor indices, which is exactly what yields combinations-with-repetition (each monomial once).

So the products are emitted in a fixed, deterministic order determined by this recursion; `diffApply` must mirror it exactly.

## Gradient back-propagation — `diffApply(…)`

Reverse-mode AD for `apply`: given upstream gradients `dout` w.r.t. each output monomial, accumulate gradients `din` w.r.t. each input feature. It walks the same recursion. For a monomial `m = ∏ x_{i_k}`, `∂m/∂x_j = (number of times j appears)·(m / x_j)` — computed without dividing by `x_j` by splitting the product into "the factor we are differentiating" and "the rest":

- The two-block body (`order ≥ 2`): the **first** block (`…, multiplier, dinOffset+i`) handles the case where the current factor `i` is the variable being differentiated — it threads a `varIndex` so the remaining factors are multiplied in and credited to `din[varIndex]`. The **second** block (`…, multiplier·in[inOffset+i]`) continues normally, treating `x_i` as a constant factor of the rest. Together they cover the product rule across all factor positions.
- `order == 1`, no `varIndex`: `din[dinOffset+i] += dout[doutOffset+i]·multiplier`.
- `order == 1`, with `varIndex`: `din[varIndex] += dout[doutOffset+i]·in[i]·multiplier` (the remaining factor folded in). Note this reads `in[i]` (not `in[inOffset+i]`) — see INCONGRUENCIES.
- `order ≤ 0` throws `IllegalArgumentException` (the constant term has no gradient path here).

Returns the number of output gradients consumed (= the same count `apply` produced), keeping offsets in lockstep.

## Gotchas / dead code

- This file (`PolyInteraction`) contains a **full `float` copy** of `apply`/`diffApply` alongside the `double` versions, duplicating `PolyInteractionFloat` entirely. See INCONGRUENCIES.
- `main` and `main2` are demos (`main2` prints degree-3 products of the first four primes). `gcd(long,long)` is a standard Euclid helper used by the simplex-number math.
- `simplexNumber` and `simplexNumberSum` rely on `assert denom==1` / `long` arithmetic; correct only while results fit in `int` (the API returns `int`). Very large `n`/order can overflow silently.
- The `order==1` `varIndex` branch indexes `in[i]` rather than `in[inOffset+i]`; harmless only when `inOffset==0`.

## Cross-references

- [`PolyInteractionFloat`](PolyInteractionFloat.md) — the `float`-only twin (same logic, no `simplexNumber`/`gcd` driver beyond the included `gcd`).
- [`PolyInteractionString`](PolyInteractionString.md) — a `String` "oracle" copy that builds the symbolic monomials/derivatives as text, used to eyeball-verify the numeric versions.
- [`PolyInteractionCursor`](PolyInteractionCursor.md) — wraps a `TrainingBatchCursor` and applies `apply(...)` on the fly to expand each sample's inputs.
- [`Whitener`](Whitener.md) — commonly composed after expansion (expand features, then whiten) in a feature pipeline.
