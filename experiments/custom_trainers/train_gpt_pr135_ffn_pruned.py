"""
PR135 variant with pruning-aware FFN channel gates.

This keeps the strong PR135 recipe intact while adding:
- learned per-channel FFN gates during training
- an L1-style gate penalty
- export-time zeroing of the least-salient FFN channels

The goal is to retain more of the over-size PR135 model's raw score while
recovering bytes through structured zeros that the existing int6+zstd path can
compress well.
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
BASE_PATH = ROOT / "experiments" / "external_prs" / "135" / "train_gpt.py"
SPEC = importlib.util.spec_from_file_location("pr135_base", BASE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"Unable to load base trainer from {BASE_PATH}")
base = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = base
SPEC.loader.exec_module(base)


PRUNE_LAMBDA = float(os.environ.get("PRUNE_LAMBDA", 0.0))
PRUNE_FRACTION = float(os.environ.get("PRUNE_FRACTION", 0.0))
PRUNE_GATE_INIT = float(os.environ.get("PRUNE_GATE_INIT", 2.5))
PRUNE_FORWARD_GATING = bool(int(os.environ.get("PRUNE_FORWARD_GATING", "1")))
PRUNE_USE_GATE_SALIENCY = bool(int(os.environ.get("PRUNE_USE_GATE_SALIENCY", "1")))

EXTRA_CONTROL_PATTERNS = ("channel_gate",)
base.CONTROL_TENSOR_NAME_PATTERNS = tuple(
    dict.fromkeys(base.CONTROL_TENSOR_NAME_PATTERNS + EXTRA_CONTROL_PATTERNS)
)


class MLP(nn.Module):
    def __init__(self, dim: int, mlp_mult: int, mlp_hidden: int = 0):
        super().__init__()
        hidden = mlp_hidden if mlp_hidden > 0 else int(mlp_mult * dim)
        self.fc = base.CastedLinear(dim, hidden, bias=False)
        self.proj = base.CastedLinear(hidden, dim, bias=False)
        self.proj._zero_init = True
        self.channel_gate = nn.Parameter(torch.full((hidden,), PRUNE_GATE_INIT, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        h = torch.relu(self.fc(x)).square()
        if PRUNE_FORWARD_GATING:
            gate = torch.sigmoid(self.channel_gate.to(dtype=h.dtype))[None, None, :]
            h = h * gate
        return self.proj(h)

    def prune_penalty(self) -> Tensor:
        return torch.sigmoid(self.channel_gate).mean()


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        mlp_hidden: int = 0,
    ):
        super().__init__()
        self.attn_norm = base.RMSNorm()
        self.mlp_norm = base.RMSNorm()
        self.attn = base.CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = MLP(dim, mlp_mult, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())

    def forward(self, x: Tensor, x0: Tensor) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        attn_out = self.attn(self.attn_norm(x))
        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out
        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))
        return x


class GPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_layers: int,
        model_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        mlp_hidden: int = 0,
        bigram_vocab_size: int = 0,
        bigram_dim: int = 128,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.bigram = base.BigramHashEmbedding(bigram_vocab_size, bigram_dim, model_dim) if bigram_vocab_size > 0 else None
        self.num_encoder_layers = num_layers // 2
        self.num_decoder_layers = num_layers - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.smear = base.SmearGate(model_dim)
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    mlp_hidden=mlp_hidden,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = base.RMSNorm()
        self.lm_head = None if tie_embeddings else base.CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        num_layers = len(self.blocks)
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                if getattr(module, "_zero_init", False):
                    nn.init.zeros_(module.weight)
                elif module.weight.ndim == 2 and module.weight.shape[0] >= 64 and module.weight.shape[1] >= 64:
                    nn.init.orthogonal_(module.weight, gain=1.0)
                    if ".proj." in name or name.endswith(".proj"):
                        with torch.no_grad():
                            module.weight.mul_(1.0 / math.sqrt(2 * num_layers))

    def _run_body(self, input_ids: Tensor) -> Tensor:
        x = self.tok_emb(input_ids)
        if self.bigram is not None:
            x = x + self.bigram(input_ids)
        x = F.rms_norm(x, (x.size(-1),))
        x = self.smear(x)
        x0 = x
        skips: list[Tensor] = []
        for i in range(self.num_encoder_layers):
            x = self.blocks[i](x, x0)
            skips.append(x)
        for i in range(self.num_decoder_layers):
            if skips:
                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
            x = self.blocks[self.num_encoder_layers + i](x, x0)
        return self.final_norm(x)

    def ffn_prune_penalty(self) -> Tensor:
        penalties = [block.mlp.prune_penalty() for block in self.blocks]
        return torch.stack(penalties).mean() if penalties else torch.zeros((), dtype=torch.float32, device=self.tok_emb.weight.device)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._run_body(input_ids).reshape(-1, self.tok_emb.embedding_dim)
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        loss = F.cross_entropy(logits.float(), targets, reduction="mean")
        if PRUNE_LAMBDA > 0.0:
            loss = loss + PRUNE_LAMBDA * self.ffn_prune_penalty().to(dtype=loss.dtype)
        return loss

    def forward_logits(self, input_ids: Tensor) -> Tensor:
        x = self._run_body(input_ids)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


def _apply_ffn_export_pruning(state_dict: dict[str, Tensor]) -> dict[str, Tensor]:
    if PRUNE_FRACTION <= 0.0:
        return {name: tensor.detach().cpu().contiguous() for name, tensor in state_dict.items()}

    pruned = {name: tensor.detach().cpu().clone().contiguous() for name, tensor in state_dict.items()}
    num_layers = max((int(k.split(".")[1]) for k in pruned if k.startswith("blocks.")), default=-1) + 1
    for layer_idx in range(num_layers):
        gate_name = f"blocks.{layer_idx}.mlp.channel_gate"
        fc_name = f"blocks.{layer_idx}.mlp.fc.weight"
        proj_name = f"blocks.{layer_idx}.mlp.proj.weight"
        if gate_name not in pruned or fc_name not in pruned or proj_name not in pruned:
            continue
        if PRUNE_USE_GATE_SALIENCY:
            saliency = torch.sigmoid(pruned[gate_name].float())
        else:
            fc = pruned[fc_name].float()
            proj = pruned[proj_name].float()
            saliency = fc.square().mean(dim=1) + proj.square().mean(dim=0)
        hidden = saliency.numel()
        prune_count = min(max(int(round(PRUNE_FRACTION * hidden)), 0), max(hidden - 1, 0))
        if prune_count <= 0:
            continue
        prune_idx = torch.argsort(saliency)[:prune_count]
        pruned[fc_name][prune_idx, :] = 0
        pruned[proj_name][:, prune_idx] = 0
        pruned[gate_name][prune_idx] = -20.0
    return pruned


_orig_quantize_state_dict_int8 = base.quantize_state_dict_int8
_orig_mixed_quantize_int6 = base.mixed_quantize_int6


def quantize_state_dict_int8(state_dict: dict[str, Tensor]):
    return _orig_quantize_state_dict_int8(_apply_ffn_export_pruning(state_dict))


def mixed_quantize_int6(state_dict: dict[str, Tensor], int6_cats: set[str]):
    return _orig_mixed_quantize_int6(_apply_ffn_export_pruning(state_dict), int6_cats)


base.GPT = GPT
base.quantize_state_dict_int8 = quantize_state_dict_int8
base.mixed_quantize_int6 = mixed_quantize_int6


if __name__ == "__main__":
    base.main()
