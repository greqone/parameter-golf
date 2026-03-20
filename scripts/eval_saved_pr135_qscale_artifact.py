from __future__ import annotations

import argparse
import importlib.util
import io
import os
import sys
import time
import zlib
from pathlib import Path

import sentencepiece as spm
import torch
import torch.distributed as dist


ROOT = Path(__file__).resolve().parents[1]


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved PR135 qscale int6 artifact with sliding-window BPB.")
    parser.add_argument("--artifact", required=True, help="Path to final_model.int6.ptz")
    return parser.parse_args()


def maybe_decompress(blob: bytes) -> tuple[bytes, str]:
    try:
        import zstandard

        try:
            return zstandard.ZstdDecompressor().decompress(blob), "zstd"
        except zstandard.ZstdError:
            pass
    except ImportError:
        pass
    return zlib.decompress(blob), "zlib"


def main() -> None:
    args_cli = parse_args()
    qscale = load_module(
        ROOT / "experiments" / "custom_trainers" / "train_gpt_pr135_qscale_int6.py",
        "pr135_qscale_saved_eval",
    )
    base = qscale.base
    args = base.Hyperparameters()

    distributed = int(os.environ.get("WORLD_SIZE", "1")) > 1
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master_process = rank == 0

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp

    enable_cudnn_sdp(False)
    enable_flash_sdp(True)
    enable_mem_efficient_sdp(False)
    enable_math_sdp(False)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(
            f"VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}"
        )
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model = base.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
        mlp_hidden=args.mlp_hidden,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, base.CastedLinear):
            module.float()
            if args.qat_enabled:
                module.qat = True
    base.restore_low_dim_params_to_fp32(model)

    artifact_path = Path(args_cli.artifact).resolve()
    blob = artifact_path.read_bytes()
    decompressed, codec = maybe_decompress(blob)
    quant_state = torch.load(io.BytesIO(decompressed), map_location="cpu")
    template_sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    deq_state = qscale.dequantize_mixed_int6(quant_state["w"], quant_state["m"], template_sd)
    model.load_state_dict(deq_state, strict=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    sw_val_loss, sw_val_bpb = base.eval_val_sliding(
        args,
        model,
        rank,
        world_size,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
    )
    torch.cuda.synchronize()
    elapsed_ms = 1000.0 * (time.perf_counter() - t0)

    if master_process:
        print(f"artifact_codec:{codec}")
        print(
            f"eval_saved_int6_sliding_window stride:{args.eval_stride} "
            f"val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} eval_time:{elapsed_ms:.0f}ms"
        )
        print(f"eval_saved_int6_sliding_window_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
