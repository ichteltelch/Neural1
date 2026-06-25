# `Neural1` — library index (Tier 1)

Source: the **Neural1** library (package `org.siquod.ml.*`).

## What this is

Neural1 is the in-house ML library. Three parts matter:

- **`org.siquod.ml.data` — data wrangling / iteration / transformation (kept, broadly useful).** A composable cursor framework for feeding training examples (input/output tensors + weights) to model-fitting code, plus feature-engineering transforms (whitening, polynomial interactions). These are useful well beyond the NN classifiers.
- **`org.siquod.ml.neural1` — the legacy feed-forward neural network (being phased out).** Consumers run pre-trained classifiers through `FeedForward.Eval` and (re)train them via `FeedForward.NaiveTrainer`. **This NN is legacy** — the plan is to replace it with **Neurotype**-based nets. These docs cover the NN at **usage/interface level only**, not the internal forward/backward math.
- **`org.siquod.ml.metrics`** — a small push-based observer graph for tracking training metrics and best-checkpoint selection.

(The `src2/neural/*` alternate source root and `examples/` are not documented.)

## `org.siquod.ml.data` — data layer

**Cursors (iteration)** — see [data/](data/):
- [`TrainingDataGiver`](data/TrainingDataGiver.md) — read-side interface for one example (inputs/outputs/weight/dims).
- [`TrainingBatchCursor`](data/TrainingBatchCursor.md) — the central iterator (next/reset/clone) + `RandomAccess` (size/seek) + a big factory library (concat/split/pad/remap/transform/polyInteraction/RAM-buffer/whiten).
- [`ShuffledCursor`](data/ShuffledCursor.md) — permuted-order `RandomAccess` decorator (per-epoch reshuffle).
- [`GenericFileCursor`](data/GenericFileCursor.md) — file-backed source (fixed-width big-endian float records; one-hot/scalar/fixed/in-file outputs).
- [`TransformedCursor`](data/TransformedCursor.md) / [`CursorTransformer`](data/CursorTransformer.md) — apply an in-place vector transform to inputs/outputs/weights.

**Transforms (feature engineering)**:
- [`Whitener`](data/Whitener.md) — affine standardize/whiten (diagonal z-score or full decorrelating); streaming weighted fit. · [`RepWhitener`](data/RepWhitener.md) — shared whitener over repeated channel-blocks.
- [`PolyInteraction`](data/PolyInteraction.md) — expand `n` features into degree-`minOrder…maxOrder` monomials (+ reverse-mode gradient). · [`PolyInteractionFloat`](data/PolyInteractionFloat.md) / [`PolyInteractionString`](data/PolyInteractionString.md) — float twin / symbolic oracle. · [`PolyInteractionCursor`](data/PolyInteractionCursor.md) — apply it per-sample in a cursor.
- [`DataManagement`](data/DataManagement.md) — ratio dataset splitting / fold concatenation / seeded `Random` cloning.

## `org.siquod.ml.neural1` — the legacy net (usage level) — see [nn/](nn/)

**Net & core**: [`FeedForward`](nn/FeedForward.md) (build/train/eval entry; inner `Eval` + `NaiveTrainer`) · [`Module`](nn/Module.md) (layer interface) · [`StackModule`](nn/StackModule.md) (sequential container = the net backbone) · [`TensorFormat`](nn/TensorFormat.md) (tensor shape/layout) · [`ParamSet`](nn/ParamSet.md) (flat weight vector) · [`InOutModule`](nn/InOutModule.md).

**Layers**: [`Dense`](nn/Dense.md) · [`Dropout`](nn/Dropout.md) · [`BatchNorm`](nn/BatchNorm.md) · [`BatchReNorm`](nn/BatchReNorm.md) · [`Nonlin`](nn/Nonlin.md) · [`ParameterizedNonlin`](nn/ParameterizedNonlin.md) · [`QuadraticInteraction`](nn/QuadraticInteraction.md) · [`QuadraticInteractionBuilder`](nn/QuadraticInteractionBuilder.md) · loss [`LossGroup`](nn/LossGroup.md) / [`SoftMaxNllLoss`](nn/SoftMaxNllLoss.md).

**Activations**: [`Neuron`](nn/Neuron.md) · [`ParameterizedNeuron`](nn/ParameterizedNeuron.md) · [`Isrlu`](nn/Isrlu.md) · [`FadeInNeuron`](nn/FadeInNeuron.md).

**Optimizers / regularizers**: [`Updater`](nn/Updater.md) · [`Adam`](nn/Adam.md) · [`AmsGrad`](nn/AmsGrad.md) · [`Regularizer`](nn/Regularizer.md) · [`L2Reg`](nn/L2Reg.md).

## `org.siquod.ml.metrics` — see [metrics/](metrics/)
[`Metric`](metrics/Metric.md) · [`MetricObserver`](metrics/MetricObserver.md) / [`DerivedMetricObserver`](metrics/DerivedMetricObserver.md) · [`MetricTracker`](metrics/MetricTracker.md) / [`DerivableMetricTracker`](metrics/DerivableMetricTracker.md) · [`TrackBest`](metrics/TrackBest.md) (best-checkpoint / early-stopping).

## Typical usage in one paragraph
A consumer builds feature vectors (often via `Whitener` + `PolyInteraction`), streams training examples through a `TrainingBatchCursor` pipeline (file/RAM source → shuffle → transform → split/pad), and for the NN trains a `FeedForward` with `NaiveTrainer` + an `Updater` (Adam/AmsGrad), tracking with `org.siquod.ml.metrics` and `TrackBest`. At inference it loads a `ParamSet` into a `FeedForward.Eval` and maps input tensors to outputs — this is the legacy path slated for Neurotype replacement.
