"""
Timestep-conditioned looped PR114 variant.

This preserves the PR114 training/export stack, but upgrades the tied-depth
branch with explicit per-loop timestep vectors and per-loop residual scaling,
matching the core recommendation from the looped-transformer paper.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.custom_trainers import train_gpt_pr114_looped_int6 as base


EXTRA_CONTROL_PATTERNS = (
    "loop_timestep",
    "loop_input_delta",
    "loop_attn_delta",
    "loop_mlp_delta",
)
base.CONTROL_TENSOR_NAME_PATTERNS = tuple(
    dict.fromkeys(base.CONTROL_TENSOR_NAME_PATTERNS + EXTRA_CONTROL_PATTERNS)
)
base.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(
    dict.fromkeys(base.INT8_KEEP_FLOAT_FP32_NAME_PATTERNS + EXTRA_CONTROL_PATTERNS)
)

LOOP_TIMESTEP_INIT_STD = float(os.environ.get("LOOP_TIMESTEP_INIT_STD", 0.02))
LOOP_SCALE_RAMP = float(os.environ.get("LOOP_SCALE_RAMP", 0.12))
LOOP_INPUT_RAMP = float(os.environ.get("LOOP_INPUT_RAMP", 0.04))


class Block(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_mult: int,
        rope_base: float,
        qk_gain_init: float,
        num_loops: int,
        mlp_hidden: int = 0,
    ):
        super().__init__()
        self.attn_norm = base.RMSNorm()
        self.mlp_norm = base.RMSNorm()
        self.attn = base.CausalSelfAttention(dim, num_heads, num_kv_heads, rope_base, qk_gain_init)
        self.mlp = base.MLP(dim, mlp_mult, mlp_hidden)
        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))
        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())
        self.loop_timestep = nn.Parameter(torch.empty(num_loops, dim, dtype=torch.float32))
        self.loop_input_delta = nn.Parameter(torch.zeros(num_loops, dim, dtype=torch.float32))
        self.loop_attn_delta = nn.Parameter(torch.zeros(num_loops, dim, dtype=torch.float32))
        self.loop_mlp_delta = nn.Parameter(torch.zeros(num_loops, dim, dtype=torch.float32))
        nn.init.normal_(self.loop_timestep, mean=0.0, std=LOOP_TIMESTEP_INIT_STD)
        if num_loops > 1:
            ramp = torch.linspace(0.0, 1.0, steps=num_loops, dtype=torch.float32).unsqueeze(1)
            self.loop_input_delta.data.add_(LOOP_INPUT_RAMP * ramp)
            self.loop_attn_delta.data.add_(LOOP_SCALE_RAMP * ramp)
            self.loop_mlp_delta.data.add_(LOOP_SCALE_RAMP * ramp)

    def forward(self, x: Tensor, x0: Tensor, loop_idx: int, lora: base.AttentionLoRA | None = None) -> Tensor:
        mix = self.resid_mix.to(dtype=x.dtype)
        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0
        loop_timestep = self.loop_timestep[loop_idx].to(dtype=x.dtype)
        loop_input_scale = (1.0 + self.loop_input_delta[loop_idx]).to(dtype=x.dtype)
        loop_attn_scale = (1.0 + self.loop_attn_delta[loop_idx]).to(dtype=x.dtype)
        loop_mlp_scale = (1.0 + self.loop_mlp_delta[loop_idx]).to(dtype=x.dtype)
        x = loop_input_scale[None, None, :] * x + loop_timestep[None, None, :]
        attn_out = self.attn(self.attn_norm(x), lora=lora)
        x = x + (self.attn_scale.to(dtype=x.dtype) * loop_attn_scale)[None, None, :] * attn_out
        x = x + (self.mlp_scale.to(dtype=x.dtype) * loop_mlp_scale)[None, None, :] * self.mlp(self.mlp_norm(x))
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
        mlp_hidden: int,
        tie_embeddings: bool,
        tied_embed_init_std: float,
        logit_softcap: float,
        rope_base: float,
        qk_gain_init: float,
        num_loops: int = 1,
        lora_rank: int = 0,
    ):
        super().__init__()
        if logit_softcap <= 0.0:
            raise ValueError(f"logit_softcap must be positive, got {logit_softcap}")
        self.tie_embeddings = tie_embeddings
        self.tied_embed_init_std = tied_embed_init_std
        self.logit_softcap = logit_softcap
        self.num_unique_layers = num_layers
        self.num_loops = num_loops
        effective_depth = num_layers * num_loops
        self.tok_emb = nn.Embedding(vocab_size, model_dim)
        self.num_encoder_layers = effective_depth // 2
        self.num_decoder_layers = effective_depth - self.num_encoder_layers
        self.num_skip_weights = min(self.num_encoder_layers, self.num_decoder_layers)
        self.skip_weights = nn.Parameter(torch.ones(self.num_skip_weights, model_dim, dtype=torch.float32))
        self.blocks = nn.ModuleList(
            [
                Block(
                    model_dim,
                    num_heads,
                    num_kv_heads,
                    mlp_mult,
                    rope_base,
                    qk_gain_init,
                    num_loops=num_loops,
                    mlp_hidden=mlp_hidden,
                )
                for _ in range(num_layers)
            ]
        )
        kv_dim = num_kv_heads * (model_dim // num_heads)
        if lora_rank > 0 and num_loops > 1:
            self.lora_adapters = nn.ModuleList(
                [
                    nn.ModuleList(
                        [base.AttentionLoRA(model_dim, kv_dim, lora_rank) for _ in range(num_layers)]
                    )
                    for _ in range(num_loops)
                ]
            )
        else:
            self.lora_adapters = None
        self.final_norm = base.RMSNorm()
        self.lm_head = None if tie_embeddings else base.CastedLinear(model_dim, vocab_size, bias=False)
        if self.lm_head is not None:
            self.lm_head._zero_init = True
        self._init_weights()

    def _init_weights(self) -> None:
        if self.tie_embeddings:
            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)
        for module in self.modules():
            if isinstance(module, nn.Linear) and getattr(module, "_zero_init", False):
                nn.init.zeros_(module.weight)

    def _run_body(self, x: Tensor) -> Tensor:
        x = F.rms_norm(x, (x.size(-1),))
        x0 = x
        skips: list[Tensor] = []
        eff_idx = 0
        for loop_idx in range(self.num_loops):
            for block_idx in range(self.num_unique_layers):
                block = self.blocks[block_idx]
                lora = self.lora_adapters[loop_idx][block_idx] if self.lora_adapters is not None else None
                if eff_idx < self.num_encoder_layers:
                    x = block(x, x0, loop_idx, lora=lora)
                    skips.append(x)
                else:
                    dec_idx = eff_idx - self.num_encoder_layers
                    if dec_idx < self.num_skip_weights and skips:
                        x = x + self.skip_weights[dec_idx].to(dtype=x.dtype)[None, None, :] * skips.pop()
                    x = block(x, x0, loop_idx, lora=lora)
                eff_idx += 1
        return self.final_norm(x)

    def forward(self, input_ids: Tensor, target_ids: Tensor) -> Tensor:
        x = self._run_body(self.tok_emb(input_ids)).reshape(-1, self.tok_emb.embedding_dim)
        targets = target_ids.reshape(-1)
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)
        return F.cross_entropy(logits.float(), targets, reduction="mean")

    @torch.no_grad()
    def get_logits(self, input_ids: Tensor) -> Tensor:
        x = self._run_body(self.tok_emb(input_ids))
        if self.tie_embeddings:
            logits_proj = F.linear(x, self.tok_emb.weight)
        else:
            if self.lm_head is None:
                raise RuntimeError("lm_head is required when tie_embeddings=False")
            logits_proj = self.lm_head(x)
        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)


base.GPT = GPT


if __name__ == "__main__":
    base.main()
