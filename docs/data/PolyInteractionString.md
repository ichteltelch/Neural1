# `PolyInteractionString`

A `String`-typed copy of [`PolyInteraction`](PolyInteraction.md) that builds the *symbolic* monomials and derivatives as text — an executable oracle for visually verifying that the numeric `apply`/`diffApply` produce the right terms in the right order.

Source folder: `Neural1` (package `org.siquod.ml.data`).

Up: [index](../_index.md).

## What it is for

The class Javadoc states it plainly: *"a copy of `PolyInteraction` where doubles are replaced by Strings so as to verify that everything is correct."* Multiplication becomes string concatenation, so running `apply` on the symbol array `{"a","b","c","d"}` prints the actual list of monomials (e.g. `a`, `b`, …, `a a`, `a b`, …) and running `diffApply` prints the symbolic partial derivatives. Comparing those strings to expectation confirms the recursion in the numeric versions enumerates and differentiates the correct terms in the correct emission order.

## Symbolic forward expansion — `apply(…)`

Same recursion as the numeric version, with the "multiplier" being an accumulated string prefix:
- `order≤0` → write the prefix `multiplier` (the empty product → empty string).
- `order==1` → write `multiplier + " " + in[i]` for each `i`.
- otherwise → recurse with `n` capped at `i+1` and prefix `multiplier+" "+in[i]`, so each emitted string is the space-separated list of factors of one monomial.

`simplexNumber(n, order)` is the plain (non-`gcd`-reduced) multiset-coefficient product, included so offsets line up.

## Symbolic back-propagation — `diffApply(…)`

Mirrors the numeric `diffApply`, but accumulates into `din[…]` by string concatenation (`+=`) so the result is a human-readable sum of derivative terms:
- order≥2 has the same two-block split (one threading `varIndex` for the differentiated factor, one continuing the product);
- `order==1` (no `varIndex`) appends `" + " + multiplier + " " + dout[i]`;
- `order==1` (with `varIndex`) appends `" + " + multiplier + " " + in[i] + " " + dout[doutOffset+i]`;
- `order≤0` throws.

`main` runs the full round trip: expand `{a,b,c,d}` at orders 2–3, wrap each term, print them, then accumulate the symbolic gradients into `dsums` and print each partial — letting a human read off `∂/∂a`, `∂/∂b`, etc.

## Gotchas / dead code

- This is a test/verification artifact, not production code; it is not meant to run inside a production pipeline.
- `din[dinOffset+i] += …` and `din[varIndex] += …` rely on the `din`/`dsums` slots being pre-seeded with non-null strings (the demo seeds them with `"0"`); a null slot would render as the literal `"null + …"`.
- The decorative glyph `"(ð"+out[i]+")"` in `main` is just labeling.

## Cross-references

- [`PolyInteraction`](PolyInteraction.md) / [`PolyInteractionFloat`](PolyInteractionFloat.md) — the numeric implementations this oracle validates.
