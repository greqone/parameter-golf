# Modal Workflow

This workspace adds a Modal-first experiment stack around the existing `train_gpt.py` and standalone record scripts.

## Files

- `infra/modal_utils.py`: config loading, FineWeb download helper, subprocess runner, log parsing, and TSV/JSON ledgers
- `infra/modal_app.py`: Modal app, image, volumes, and remote entrypoints for proxy, promotion, Hopper fallback, and H100 verification
- `scripts/modal_smoke.py`: one-command smoke path, with local fallback when Modal auth is absent
- `scripts/modal_proxy_sweep.py`: batch proxy runner
- `scripts/modal_promote.py`: 8-GPU promotion launcher
- `scripts/modal_verify_h100.py`: 8-GPU H100 verification launcher

## Volumes

- `pg-data`: cached datasets and tokenizers
- `pg-checkpoints`: checkpoint payloads for resume-safe retries
- `pg-results`: remote run directories with logs, models, and parsed result JSON

## Typical Flow

1. Smoke the pipeline:
   ```powershell
   .\.venv\Scripts\python.exe scripts\modal_smoke.py --mode auto
   ```
2. Launch the first proxy batch:
   ```powershell
   .\.venv\Scripts\python.exe scripts\modal_proxy_sweep.py --mode auto
   ```
3. Promote a promising config:
   ```powershell
   .\.venv\Scripts\python.exe scripts\modal_promote.py --config experiments\configs\proxy_fp16_embed_wd3600.json
   ```
4. Verify on benchmark-faithful H100:
   ```powershell
   .\.venv\Scripts\python.exe scripts\modal_verify_h100.py --config experiments\configs\proxy_fp16_embed_wd3600.json
   ```

## Notes

- Local execution defaults to `ENABLE_TORCH_COMPILE=0` and `SDP_BACKEND=math` because the current Windows + 4090 path trips an invalid-backend error under the stock compile + flash setup.
- Remote execution keeps the benchmark-oriented defaults: `ENABLE_TORCH_COMPILE=1` and `SDP_BACKEND=flash`.
- `run_promote_b200` targets `B200+:8`, but `scripts/modal_promote.py --hopper-fallback` is available when Hopper is the safer path.
- Every run writes `spec.json`, `train.log`, and `result.json` in its run directory, and appends a summarized row to `experiments/results.tsv`.
