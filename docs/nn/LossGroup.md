# `LossGroup`

Describes how to partition an output vector into independent probability distributions for softmax/NLL loss.

Source folder: `Neural1` (package `org.siquod.ml.neural1.modules.loss`).

Up: [index](../_index.md).

## What it is for

A net's output vector may encode several distributions side by side (e.g. several independent classifications). A `LossGroup` marks one contiguous slice `[start, end)` of the outputs as one distribution, so the loss head can softmax and compute NLL per group rather than over the whole vector. A singleton group (`length==1`) is interpreted as the `p` of a Bernoulli (the other class is implicitly `1-p`). An array of `LossGroup`s configures a [`SoftMaxNllLoss`](SoftMaxNllLoss.md).

## Fields (immutable)

- `start`, `end`, `length` — the output slice (`end = start + length`).
- `gate` — index into the target distribution whose value multiplies this group's loss (skip when `-1`; `isGated()` tests this). Lets one output gate whether another group's loss counts.
- `gateInverted` — if true, the gate uses `1-p` instead of `p`.
- `weight` — multiplier on this group's loss contributions.
- `isSingleton()` — `length==1` (Bernoulli case).

## Construction — constructors, `makeDefault`, builder

- `LossGroup(int start, int length)` — ungated, weight 1.
- `LossGroup(int start, int length, int gate, boolean gateInverted, float weight)` — full form.
- `LossGroup.makeDefault(int count)` — one ungated group spanning all `count` outputs (the common single-softmax case).
- `LossGroup.b()` → `LossGroupsBuilder`.

## Builder — `LossGroup.LossGroupsBuilder`

Appends groups left-to-right tracking a running `position`: `group(int length)` adds a length-`n` group; `singleton()` adds a Bernoulli group. `weight(float)`, `gate(int)`, `gate(int, boolean)`, `gateInverted(boolean)`, `ungated()` set sticky attributes for subsequently-added groups. `assertTotalLength(int)` sanity-checks the total; `toArray()` returns the `LossGroup[]`.

## Gotchas / dead code

- Javadoc on `gate` references `{@link #gateInverted}` where it means the gate *index* — the gating value is read at index `gate`, not `gateInverted`.

## Cross-references

- [SoftMaxNllLoss](SoftMaxNllLoss.md) — consumes `LossGroup[]` to drive per-group softmax + NLL.
- `LogSoftmax`, `NllLoss` — the underlying per-group operations.
