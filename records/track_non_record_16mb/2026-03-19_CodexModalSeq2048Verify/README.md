This package captures the best candidate run executed in this workspace through the new Modal experiment stack.

Important honesty note:
- This is **not** a new SOTA submission.
- It is a clean **Modal reproduction / re-verification** of the existing `Long Context Seq2048` recipe already present in the public repo.
- It is packaged under `track_non_record_16mb` because the `H100!:8` verify result from this session (`1.21223295`) does not beat the stronger public seq2048 record already in the repo.

## Candidate

- Recipe source: `records/track_10min_16mb/2026-03-18_LongContextSeq2048/train_gpt.py`
- Model family: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4`
- Sequence length: `TRAIN_SEQ_LEN=2048`
- Tokenizer/data family: `fineweb10B_sp1024`
- Training budget: `MAX_WALLCLOCK_SECONDS=600`
- Train shard count used here: `80`

## Executed Runs In This Package

### Promotion run

- Runtime: Modal `B200+:8`
- Command:
  ```bash
  torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```
- Exact metric from this session:
  - `final_int8_zlib_roundtrip_exact val_loss:2.03933200`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.20780728`
- Size accounting:
  - compressed model: `15,809,089` bytes
  - code: `47,716` bytes
  - total: `15,856,805` bytes

### Verify run

- Runtime: Modal `H100!:8`
- Command:
  ```bash
  torchrun --standalone --nproc_per_node=8 train_gpt.py
  ```
- Exact metric from this session:
  - `final_int8_zlib_roundtrip_exact val_loss:2.04680455`
  - `final_int8_zlib_roundtrip_exact val_bpb:1.21223295`
- Size accounting:
  - compressed model: `15,806,306` bytes
  - code: `47,716` bytes
  - total: `15,854,022` bytes

## Included Files

- `train_gpt.py`: standalone seq2048 trainer used for both runs
- `train.log`: exact `H100!:8` verification log from this session
- `train_b200.log`: exact `B200+:8` promotion log from this session
- `verify_result.json`: parsed result metadata for the H100 verification run
- `promote_result.json`: parsed result metadata for the B200 promotion run
- `submission.json`: summary metadata for this packaged reproduction
