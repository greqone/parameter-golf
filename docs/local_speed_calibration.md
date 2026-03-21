# Local Speed Calibration

This note records the current local-screening heuristic for deciding whether a candidate is worth promoting from the RTX 4090 to a real `8xH100` verification run.

## Why this exists

The frontier PR114-family trainers are benchmark-faithful on `8xH100`, but their default Flash-SDPA / `torch.compile` path is not portable to this local Windows + RTX 4090 setup. For local iteration we therefore use a **relative** throughput screen rather than pretending the 4090 exactly matches cloud hardware.

## Current anchors

### Local proxy anchor

- Hardware: `RTX 4090`
- Script: `experiments/custom_trainers/train_gpt_pr114_localsafe.py`
- Settings:
  - `TRAIN_BATCH_TOKENS=65536`
  - `TRAIN_SEQ_LEN=1024`
  - `ENABLE_TORCH_COMPILE=0`
  - `SDP_BACKEND=math`
  - `VAL_LOSS_EVERY=0`
- Observed steady-state training speed:
  - about `937.4 ms/step`
  - about `69.9k train tokens/sec`
- Log:
  - `experiments/benchmarks/20260321-pr114-localsafe-4090-65k-math/train_partial.stdout.log`

### Cloud verification anchors

- RunPod `8xH100 SXM` PR114 verify:
  - about `80.76 ms/step`
  - about `6.49M train tokens/sec`
  - log: `experiments/runpod_artifacts/3m0b4blkdgg27s_seed1338/runpod_pr114_verify_seed1338.log`
- Modal `8xH100!` PR114 verify:
  - about `95.52 ms/step`
  - about `5.49M train tokens/sec`
  - log: `experiments/runs/20260319-215447-candidate-pr114-int6-mlp3x-slide-full/train.log`

### Approximate throughput ratio

- RunPod `8xH100 SXM` vs local proxy: about `92.9x` tokens/sec
- Modal `8xH100!` vs local proxy: about `78.5x` tokens/sec

These ratios are not meant as exact hardware emulation. They are a **promotion heuristic**.

## Promotion heuristic

For local-first screening, compare new candidates against the same local PR114 proxy setup:

1. If local steady-state step time is materially slower than the PR114 proxy without a clearly better loss trend, do not promote.
2. Treat roughly `<= 1.05x` PR114 local step time as safe.
3. Treat roughly `1.05x` to `1.15x` slower as borderline and require a clear proxy-loss win.
4. Treat `> 1.15x` slower as unlikely to justify `8xH100` spend unless the architecture change is unusually promising on bytes or optimization.

## Important caveat

Trying to run the full-size `TRAIN_BATCH_TOKENS=524288` local fallback path on the 4090 with `SDP_BACKEND=math` did not even reach the first logged training step inside three minutes, so that configuration is not useful as a local benchmark mode.
