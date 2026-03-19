# Initial Ablation Plan

Current local priors from this checkout:
- `records/track_10min_16mb/2026-03-18_FP16Embed_WD3600` shows that preserving `tok_emb.weight` in fp16 plus a longer warmdown is already a clean win.
- `records/track_10min_16mb/2026-03-18_LongContextSeq2048` shows that longer context survives the 16 MB budget and improves exact roundtrip BPB.
- `records/track_10min_16mb/2026-03-19_SlidingWindowEval` shows evaluation-only changes can move the metric dramatically, so every candidate needs exact metric verification.
- Live public PRs as of 2026-03-19 point at a frontier built around wider MLPs, mixed or lower-bit quantization, sliding-window eval, larger tokenizers, and selective precision.

Proxy batch order:
1. Baseline smoke to validate data, export, exact metric parsing, checkpoints, and the Modal plumbing.
2. FP16 embedding + warmdown because it is a high-confidence training-side gain with low complexity.
3. Seq2048 because longer context appears robust and changes the optimization regime enough to deserve its own lane.
4. Sliding-window eval because it meaningfully changes the exact scored metric and must be measured separately from training quality.

Promotion gate:
- Exact metric available and parsed cleanly.
- Total bytes plausibly remain under 16,000,000.
- No trainer/runtime instability from compile, SDP backend, or checkpoint resume.
- Clear proxy signal relative to the smoke baseline or a strong prior from an already reproduced local record script.
