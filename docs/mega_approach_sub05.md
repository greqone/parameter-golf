# Mega Approach for Sub-0.5

This note is the current theory synthesis for a world-class Parameter Golf run, grounded in the local publication dump and the live competition frontier.

## Brutal honesty first

Sub-`0.5` is not a reasonable target for "stack a few known tuning tricks on PR114 and pray."

If we want to get anywhere near `0.5`, we need a qualitatively different regime:

1. A **challenge-safe modeling regime** that gets dramatically more modeling power per byte and per train-second than the current PR114/PR135 family.
2. A **compression-first regime** that spends artifact bytes on explicit evaluation-time memory or side information instead of only parametric weights.

The second regime is the only one that currently looks plausibly capable of smashing through `1.0` and pushing toward `0.5`, but it may not be accepted for the official record track.

Evidence:

- Closed PR `#168` reports a claimed `val_bpb=1.0238` with an `8.75MB` paid-prefix blob:
  - <https://github.com/openai/parameter-golf/pull/168>
- Closed PR `#275` explicitly says a paid-prefix line was "ruled out-of-scope by organizers" and estimates that larger prefix coverage could reach about `0.75 BPB`:
  - <https://github.com/openai/parameter-golf/pull/275>

So the honest conclusion is:

- **Official-safe path**: plausibly push well below the current public leaderboard and maybe toward the high `0.x` range.
- **Sub-0.5 path**: probably requires a compression-memory artifact that starts to look more like a learned coder with side information than a pure small LM.

## What the papers say actually matters

### 1. Gating is a real win, not cosmetic

The Gated Attention paper is one of the clearest "cheap gain" results in the whole dump:

- head/query-dependent gating after SDPA
- more stable optimization
- larger learning-rate tolerance
- less attention-sink pathology
- better long-context behavior

Practical implication:

- Keep an attention gate in the stack.
- Also apply the same philosophy to FFN/channel routing, but in a byte-cheap grouped form.

Sources:

- local abstract:
  - `../all_llm_training_sources_complete/global_deduped/gated-attention-for-large-language-models-non-linearity-sparsity-and-att-2505-06708/markdownfile_abstract.md`
- primary page:
  - <https://openreview.net/forum?id=1b7whO4SfY>

### 2. Pure weight tying is too weak; token-interface tying should be smarter

Pseudo-Inverse Tying (PIT) is interesting because it upgrades the usual "shared embedding/head matrix" trick into a structured token interface:

- shared latent token memory
- stable pseudo-inverse-like encode/decode interface
- stronger optimization stability

Practical implication:

- If we continue using tied embeddings, we should stop treating them as a dumb equality constraint.
- A compact PIT-like latent token memory is one of the highest-leverage byte-saving ideas in the dump.

Source:

- `../all_llm_training_sources_complete/global_deduped/rethinking-weight-tying-pseudo-inverse-tying-for-stable-lm-training-and-2602-04556/markdownfile_abstract.md`
- primary page:
  - <https://arxiv.org/abs/2602.04556>

### 3. Low-rank only is not enough; low-rank plus structured sparse is stronger

LOST's core lesson is exactly what matters for this challenge:

- keep the dominant low-rank subspace
- recover lost expressivity with channel-structured sparse components
- place a nonlinearity inside the low-rank factorization

Practical implication:

- The right FFN is not dense-vs-pruned.
- The right FFN is probably:
  - low-rank factorized,
  - gated,
  - plus a tiny channel-sparse residual path.

Source:

- `../all_llm_training_sources_complete/global_deduped/lost-low-rank-and-sparse-pre-training-for-large-language-models-2508-02668/markdownfile_section_003_3-methodology.md`
- primary page:
  - <https://arxiv.org/abs/2508.02668>

### 4. Sequential low-rank accumulation matters more than static LoRA-style deltas

AccLoRT argues for:

- training successive low-rank subspaces
- accumulating them into a richer effective matrix
- using early-stage memory savings to raise rank when it matters most

Practical implication:

- If we introduce low-rank factors, they should not remain fixed-rank and static for the whole run.
- A tiny-rank, stagewise-accumulated update path is more plausible than permanent full-rank matrices everywhere.

Source:

- `../all_llm_training_sources_complete/global_deduped/acclort-efficient-large-language-models-pretraining-through-low-rank-acc-qkmmrgglbu/markdownfile_section_001_acclort-efficient-large-language-models-pr.md`

### 5. Sub-quadratic recurrence is mandatory if we want a true step-change

Mamba-3, Kimi Linear, RWKV-7, and xLSTM all point in the same direction:

- the next efficiency frontier is hybrid recurrent/state-space structure
- not pure transformer everywhere
- not pure linear model everywhere either

The shared idea:

- use recurrent/state-space blocks for cheap token ingestion and state tracking
- keep a smaller number of exact/sparse attention anchors for copy/retrieval/high-fidelity mixing

Sources:

- `../all_llm_training_sources_complete/global_deduped/mamba-3-improved-sequence-modeling-using-state-space-principles-2603-15569/markdownfile_abstract.md`
- `../all_llm_training_sources_complete/global_deduped/kimi-linear-an-expressive-efficient-attention-architecture-2510-26692/markdownfile_abstract.md`
- `../all_llm_training_sources_complete/global_deduped/rwkv-7-goose-with-expressive-dynamic-state-evolution-2503-14456`
- `../all_llm_training_sources_complete/global_deduped/xlstm-7b-a-recurrent-llm-for-fast-and-efficient-inference-2503-13427`
- primary pages:
  - <https://openreview.net/forum?id=HwCvaJOiCj>
  - <https://arxiv.org/abs/2510.26692>

### 6. Sparse attention only helps if it is hardware-friendly

The sparse-attention papers matter for one reason:

- they remind us that "algorithmically cheaper" is worthless if it is not GPU-efficient

Practical implication:

- If we add sparse attention, it must be:
  - coarse and structured,
  - few-pattern,
  - anchor-style,
  - not a custom irregular mask circus.

Sources:

- `../all_llm_training_sources_complete/global_deduped/native-sparse-attention-hardware-aligned-and-natively-trainable-sparse-a-2502-11089/markdownfile_abstract.md`
- `../all_llm_training_sources_complete/global_deduped/sparse-query-attention-sqa-a-computationally-efficient-attention-mechani-2510-01817`
- primary page:
  - <https://arxiv.org/abs/2502.11089>

### 7. Native low-bit training is a destination, not just an export trick

BitNet's lesson is not "ternary inference is neat."
It is:

- native low-bit structure can keep far more useful capacity per byte than naive post-hoc compression

Practical implication:

- We should stop thinking of quantization as only the final exporter.
- The best version of this project likely trains with an explicit low-bit-aware parameterization and only keeps a tiny set of sensitive tensors in higher precision.

Sources:

- `../all_llm_training_sources_complete/global_deduped/bitnet-b1-58-2b4t-technical-report-2504-12285/markdownfile_abstract.md`
- primary page:
  - <https://arxiv.org/abs/2504.12285>

## The actual Mega stack

The strongest plausible stack is:

### Tier A: official-safe "monster LM"

This is the best serious path for an accepted record attempt.

#### Architecture

- **Hybrid recurrent-attention backbone**
  - repeat a 4-block logical macro-layer
  - 3 recurrent/state-space-style blocks
  - 1 exact or structured-sparse attention anchor block
- **Looped/shared depth**
  - small number of unique blocks
  - many effective depth steps via recurrence / unrolling
  - explicit loop/timestep embedding
- **PIT-style token interface**
  - shared latent token memory
  - structured input/output coupling instead of plain tied weights
- **Grouped gated FFN**
  - grouped channel gates
  - low-rank factorized FFN core
  - small channel-sparse residual branch
- **Selective precision**
  - tiny control tensors high precision
  - heavy matrices native low-bit-aware

#### Training

- Muon on matrix-shaped params
- Adam only on scalars / norms / gates / token-interface transforms
- aggressive LR enabled by attention gating
- stagewise low-rank accumulation instead of fixed low-rank forever
- shard ordering / curriculum chosen for fastest compression progress, not generic loss aesthetics

#### Evaluation

- sliding-window eval
- recurrent carry state across windows where rules allow
- no direct validation-token storage

This is the path most likely to produce a **real accepted SOTA**.

### Tier B: compression-first "artifact as side information"

This is the only path I currently see with a serious theoretical route toward `0.5` or below.

#### Core idea

Spend most artifact bytes on:

- compressed prefix memory
- learned dictionary/state cache
- rank-coded or entropy-coded token side information

and keep only a much smaller model to interpolate the uncovered regions.

The evidence from PR `#275` is damning and illuminating:

- each MB of prefix can buy more BPB reduction than each MB of model in that regime
- they explicitly estimated that bigger coverage could reach about `0.75 BPB`

Extrapolation:

- a stronger coder + better coverage + a lightweight interpolating hybrid LM could push much lower
- but the organizers appear to consider direct val-token storage out-of-scope

So this is the **moonshot** path, but probably not the **official record** path.

## My current best theory of what can actually win

If the goal is "best accepted record," the best path is:

1. **Hybrid recurrent-attention model**
2. **PIT-style smarter tying**
3. **Grouped gated low-rank+sparse FFN**
4. **Native low-bit-aware training**
5. **Selective attention anchors and long-context eval**

If the goal is "numerically absurd BPB, maybe sub-0.5," the best path is:

1. **Artifact-paid external memory / prefix / dictionary**
2. **Tiny hybrid recurrent interpolator**
3. **Compression-aware coding of the side information**

The second path is the real Ultra Instinct route.
It is also the one most likely to be ruled out.

## What this means for our next experiments

We should not keep mutating PR114 randomly.

The next execution ladder should be:

1. **Finish the cheap grouped-gate PR114 probes**
   - already in progress
2. **Add PIT-lite token-interface tying**
   - smallest high-leverage new mechanism
3. **Replace dense FFN with low-rank + grouped-gate + tiny channel-sparse branch**
   - LOST-inspired but challenge-sized
4. **Build a looped hybrid block**
   - recurrent/state-space block for 3 out of every 4 logical layers
   - sparse/full attention anchor every 4th logical step
5. **Only then consider a prefix-memory research branch**
   - clearly separated from official-safe work

## Working thesis

The winning accepted model is probably not:

- a plain transformer
- a plain Mamba
- a plain quantized baseline
- a pile of post-hoc compression tricks

It is probably:

- a **hybrid recurrent-attention compressor**
- with **smart tying**
- **gated routing**
- **low-rank + sparse FFNs**
- and **native low-bit structure from early training**

That is the most defensible path to a world-class result without crossing the "out-of-scope compression artifact" line.
