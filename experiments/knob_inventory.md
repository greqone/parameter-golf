# Root `train_gpt.py` Knob Inventory

Model shape:
- `VOCAB_SIZE`
- `NUM_LAYERS`
- `MODEL_DIM`
- `NUM_HEADS`
- `NUM_KV_HEADS`
- `MLP_MULT`
- `TIE_EMBEDDINGS`

Context and evaluation:
- `TRAIN_SEQ_LEN`
- `VAL_BATCH_SIZE`
- `VAL_LOSS_EVERY`
- `ROPE_BASE`
- `LOGIT_SOFTCAP`

Optimization and routing:
- `EMBED_LR`
- `HEAD_LR`
- `TIED_EMBED_LR`
- `TIED_EMBED_INIT_STD`
- `MATRIX_LR`
- `SCALAR_LR`
- `MUON_MOMENTUM`
- `MUON_BACKEND_STEPS`
- `MUON_MOMENTUM_WARMUP_START`
- `MUON_MOMENTUM_WARMUP_STEPS`
- `BETA1`
- `BETA2`
- `ADAM_EPS`
- `GRAD_CLIP_NORM`
- `QK_GAIN_INIT`
- `WARMDOWN_ITERS`
- `WARMUP_STEPS`
- `TRAIN_BATCH_TOKENS`
- `ITERATIONS`
- `MAX_WALLCLOCK_SECONDS`

Compression and export:
- `CONTROL_TENSOR_NAME_PATTERNS`
- `INT8_KEEP_FLOAT_FP32_NAME_PATTERNS`
- Built-in export path is raw `final_model.pt` plus exact int8+zlib roundtrip export to `final_model.int8.ptz`
- Quantization scheme is per-row int8 for 2D float tensors, per-tensor int8 for lower-rank float tensors, fp16 passthrough for small tensors, and exact passthrough for non-floats

Tokenizer and data:
- `DATA_PATH`
- `TOKENIZER_PATH`
- Train shard selection is path-driven through `fineweb_train_*.bin`
- Validation always scans the full `fineweb_val_*.bin` pattern in the chosen dataset directory

Runtime and portability controls added in this workspace:
- `ENABLE_TORCH_COMPILE`
- `SDP_BACKEND`
- `CHECKPOINT_PATH`
- `CHECKPOINT_EVERY`
- `RESUME_FROM_CHECKPOINT`

Record-only knobs already present in local record scripts:
- FP16 embed record adds `MLP_HIDDEN`
- Sliding-window record adds `EVAL_STRIDE`, `EVAL_BATCH_SEQS`, `NUM_LOOPS`, `LORA_RANK`, and `QAT`
