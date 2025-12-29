# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Prophet world model building blocks (ProphRL, Section 3.2).

This module is intentionally *self-contained* and *minimally invasive*:
- It reuses Cosmos-Predict2's DiT stack (`MiniTrainDIT` via `MinimalV1LVGDiT`).
- It adds two additions that are required by Prophet:
  1) Dual Action Conditioning:
     - Scalar stream: action chunk -> global embedding, injected into timestep embedding.
     - Action-frame stream: 7D action -> rendered 2D action frames -> 3D conv -> embedding, injected too.
  2) History-aware mechanism:
     - Optional rolling latent history buffer (T_h frames).
     - History latents -> 3D avg-pool -> projection -> tokens concatenated to cross-attn context for all blocks.

Wiring into datasets/pipelines/configs is done separately; this file focuses on the core architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange, repeat

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.text2image_dit import Attention, GPT2FeedForward, VideoSize
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from imaginaire.utils.graph import create_cuda_graph

# Prophet uses Wan2.1 Video Autoencoder geometry (Section 4.1.1).
WAN21_LATENT_CHANNELS = 16
WAN21_SPATIAL_DOWNSAMPLE = 8
WAN21_TEMPORAL_DOWNSAMPLE = 4


class ScalarActionMLP(nn.Module):
    """Scalar stream: action chunk (B,T,7) -> (B,1,Dm)."""

    def __init__(self, *, action_dim: int, action_horizon: int, d_model: int = 1024) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.action_horizon = action_horizon
        self.d_model = d_model
        in_dim = action_dim * action_horizon
        self.net = nn.Sequential(
            nn.Linear(in_dim, d_model * 4, bias=True),
            nn.GELU(approximate="tanh"),
            nn.Linear(d_model * 4, d_model, bias=True),
        )

    def forward(self, action_B_T_7: torch.Tensor) -> torch.Tensor:
        if action_B_T_7.ndim != 3 or action_B_T_7.shape[-1] != self.action_dim:
            raise ValueError(f"Expected action (B,T,{self.action_dim}), got {action_B_T_7.shape}")
        if action_B_T_7.shape[1] != self.action_horizon:
            raise ValueError(
                f"Expected action horizon {self.action_horizon}, got {action_B_T_7.shape[1]} (shape={action_B_T_7.shape})"
            )
        x = rearrange(action_B_T_7, "b t d -> b (t d)")
        return self.net(x).unsqueeze(1)  # (B,1,Dm)


class ActionFrameLatentEncoder(nn.Module):
    """
    Action Frame Stream (Section 3.2.3): consumes Wan2.1 VAE latents of rendered action frames.

    Expected input: (B, C_l=16, T_l, H_l, W_l)
    Conv group: 1x1x1 -> 1x3x3 -> 1x1x1, then global avg pool to (B,Dm).
    """

    def __init__(self, *, in_channels: int = WAN21_LATENT_CHANNELS, d_model: int = 1024) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.d_model = d_model
        mid = d_model // 2
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, mid, kernel_size=(1, 1, 1), padding=0, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Conv3d(mid, mid, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.GELU(approximate="tanh"),
            nn.Conv3d(mid, d_model, kernel_size=(1, 1, 1), padding=0, bias=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

    def forward(self, action_latents_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        if action_latents_B_C_T_H_W.ndim != 5:
            raise ValueError(f"Expected action latents (B,C,T,H,W), got {action_latents_B_C_T_H_W.shape}")
        if action_latents_B_C_T_H_W.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected action latents channels {self.in_channels}, got {action_latents_B_C_T_H_W.shape[1]}"
            )
        x = self.net(action_latents_B_C_T_H_W)
        return x.flatten(1).unsqueeze(1)  # (B,1,Dm)


class FramePack(nn.Module):
    """
    FramePack (Section 3.2.4): history latents -> pooled memory matrix M.

    Input: history_latents (B, C_l, T_h, H_l, W_l)
    Output: M (B, S_mem, D_ctx) where D_ctx == cross-attn context dim.
    """

    def __init__(
        self,
        *,
        latent_channels: int = WAN21_LATENT_CHANNELS,
        context_dim: int,
        pool_kernel: tuple[int, int, int] = (3, 2, 2),
        pool_stride: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()

        def _as_int_tuple3(x: Any, *, name: str) -> tuple[int, int, int]:
            # Hydra / OmegaConf may pass ListConfig here; PyTorch needs plain tuple[int,...].
            try:
                from omegaconf import ListConfig  # type: ignore
            except Exception:  # pragma: no cover
                ListConfig = ()  # type: ignore

            if isinstance(x, int):
                return (int(x), int(x), int(x))
            if isinstance(x, (list, tuple)) or (ListConfig != () and isinstance(x, ListConfig)):
                xs = tuple(int(v) for v in list(x))
                if len(xs) != 3:
                    raise ValueError(f"{name} must have length 3, got {len(xs)}: {x}")
                return xs  # type: ignore[return-value]
            raise TypeError(f"{name} must be int or 3-tuple/list, got {type(x)}: {x}")

        pool_kernel = _as_int_tuple3(pool_kernel, name="pool_kernel")
        stride = _as_int_tuple3(pool_stride if pool_stride is not None else pool_kernel, name="pool_stride")
        self.pool = nn.AvgPool3d(kernel_size=pool_kernel, stride=stride, ceil_mode=False)
        self.proj = nn.Linear(latent_channels, context_dim, bias=False)

    def forward(self, history_latents_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        if history_latents_B_C_T_H_W.ndim != 5:
            raise ValueError(f"Expected history latents (B,C,T,H,W), got {history_latents_B_C_T_H_W.shape}")
        # Extra safety: if someone constructed AvgPool3d with a non-tuple kernel/stride (e.g. ListConfig),
        # rebuild it once here to avoid `TypeError: avg_pool3d()` at runtime.
        if not isinstance(self.pool.kernel_size, tuple) or not isinstance(self.pool.stride, tuple):
            k = tuple(int(v) for v in self.pool.kernel_size)  # type: ignore[arg-type]
            s = tuple(int(v) for v in self.pool.stride)  # type: ignore[arg-type]
            self.pool = nn.AvgPool3d(kernel_size=k, stride=s, ceil_mode=False)
        pooled = self.pool(history_latents_B_C_T_H_W)
        tokens = rearrange(pooled, "b c t h w -> b (t h w) c").contiguous()
        return self.proj(tokens)


class ProphetCrossAttention(nn.Module):
    """
    Cross-attention with explicit memory KV concatenation:
      new_K = cat([K_mem, K], dim=seq_len)
      new_V = cat([V_mem, V], dim=seq_len)
    """

    def __init__(self, base: Attention) -> None:
        super().__init__()
        self.base = base

    def forward(self, x: torch.Tensor, context: torch.Tensor, memory: torch.Tensor | None = None) -> torch.Tensor:
        # Build Q from x
        q = self.base.q_proj(x)
        q = rearrange(q, "b s (h d) -> b s h d", h=self.base.n_heads, d=self.base.head_dim)
        q = self.base.q_norm(q)

        # Build K/V from context
        k = self.base.k_proj(context)
        v = self.base.v_proj(context)
        k = rearrange(k, "b s (h d) -> b s h d", h=self.base.n_heads, d=self.base.head_dim)
        v = rearrange(v, "b s (h d) -> b s h d", h=self.base.n_heads, d=self.base.head_dim)
        k = self.base.k_norm(k)
        v = self.base.v_norm(v)

        if memory is not None:
            km = self.base.k_proj(memory)
            vm = self.base.v_proj(memory)
            km = rearrange(km, "b s (h d) -> b s h d", h=self.base.n_heads, d=self.base.head_dim)
            vm = rearrange(vm, "b s (h d) -> b s h d", h=self.base.n_heads, d=self.base.head_dim)
            km = self.base.k_norm(km)
            vm = self.base.v_norm(vm)
            k = torch.cat([km, k], dim=1)
            v = torch.cat([vm, v], dim=1)

        # Compute attention (no RoPE for cross-attn)
        out = self.base.attn_op(q, k, v)
        return self.base.output_dropout(self.base.output_proj(out))


class ProphetBlock(nn.Module):
    """
    DiT block with explicit KV concat for memory in cross-attention.
    This is a Prophet-specific variant of `cosmos_predict2.models.text2image_dit.Block`.
    """

    def __init__(
        self,
        *,
        x_dim: int,
        context_dim: int,
        num_heads: int,
        mlp_ratio: float,
        self_attention_backend: str,
        cross_attention_backend: str,
        natten_params: Any = None,
        use_adaln_lora: bool = False,
        adaln_lora_dim: int = 256,
    ) -> None:
        super().__init__()
        self.x_dim = x_dim
        self.layer_norm_self_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.self_attn = Attention(
            x_dim,
            None,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=self_attention_backend,
            natten_params=natten_params,
        )
        self.layer_norm_cross_attn = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        base_cross = Attention(
            x_dim,
            context_dim,
            num_heads,
            x_dim // num_heads,
            qkv_format="bshd",
            backend=cross_attention_backend,
        )
        self.cross_attn = ProphetCrossAttention(base_cross)
        self.layer_norm_mlp = nn.LayerNorm(x_dim, elementwise_affine=False, eps=1e-6)
        self.mlp = GPT2FeedForward(x_dim, int(x_dim * mlp_ratio))

        # Keep AdaLN-LoRA flags for compatibility with existing configs.
        self.use_adaln_lora = use_adaln_lora
        if self.use_adaln_lora:
            self.adaln_modulation_self_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_cross_attn = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
            self.adaln_modulation_mlp = nn.Sequential(
                nn.SiLU(),
                nn.Linear(x_dim, adaln_lora_dim, bias=False),
                nn.Linear(adaln_lora_dim, 3 * x_dim, bias=False),
            )
        else:
            self.adaln_modulation_self_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_cross_attn = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))
            self.adaln_modulation_mlp = nn.Sequential(nn.SiLU(), nn.Linear(x_dim, 3 * x_dim, bias=False))

        self.cp_size = None

    def set_context_parallel_group(self, process_group, ranks, stream) -> None:  # parity with base Block API
        self.cp_size = None if ranks is None else len(ranks)
        self.self_attn.set_context_parallel_group(process_group=process_group, ranks=ranks, stream=stream)

    def forward(
        self,
        x_B_T_H_W_D: torch.Tensor,
        emb_B_T_D: torch.Tensor,
        crossattn_emb: torch.Tensor,
        memory_M_B_S_D: torch.Tensor | None = None,
        rope_emb_L_1_1_D: torch.Tensor | None = None,
        adaln_lora_B_T_3D: torch.Tensor | None = None,
        extra_per_block_pos_emb: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if extra_per_block_pos_emb is not None:
            x_B_T_H_W_D = x_B_T_H_W_D + extra_per_block_pos_emb

        if self.use_adaln_lora:
            assert adaln_lora_B_T_3D is not None
            shift_sa, scale_sa, gate_sa = (self.adaln_modulation_self_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_ca, scale_ca, gate_ca = (self.adaln_modulation_cross_attn(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
            shift_mlp, scale_mlp, gate_mlp = (self.adaln_modulation_mlp(emb_B_T_D) + adaln_lora_B_T_3D).chunk(
                3, dim=-1
            )
        else:
            shift_sa, scale_sa, gate_sa = self.adaln_modulation_self_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_ca, scale_ca, gate_ca = self.adaln_modulation_cross_attn(emb_B_T_D).chunk(3, dim=-1)
            shift_mlp, scale_mlp, gate_mlp = self.adaln_modulation_mlp(emb_B_T_D).chunk(3, dim=-1)

        # (B,T,D) -> (B,T,1,1,D)
        def bt1(ten: torch.Tensor) -> torch.Tensor:
            return rearrange(ten, "b t d -> b t 1 1 d")

        shift_sa, scale_sa, gate_sa = bt1(shift_sa), bt1(scale_sa), bt1(gate_sa)
        shift_ca, scale_ca, gate_ca = bt1(shift_ca), bt1(scale_ca), bt1(gate_ca)
        shift_mlp, scale_mlp, gate_mlp = bt1(shift_mlp), bt1(scale_mlp), bt1(gate_mlp)

        B, T, H, W, D = x_B_T_H_W_D.shape

        def modulate(x: torch.Tensor, ln: nn.Module, scale: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
            return ln(x) * (1 + scale) + shift

        # self-attn
        norm_x = modulate(x_B_T_H_W_D, self.layer_norm_self_attn, scale_sa, shift_sa)
        video_size = VideoSize(T=T, H=H, W=W)
        if self.cp_size is not None and self.cp_size > 1:
            video_size = VideoSize(T=T * self.cp_size, H=H, W=W)
        sa_out = rearrange(
            self.self_attn(
                rearrange(norm_x, "b t h w d -> b (t h w) d"),
                None,
                rope_emb=rope_emb_L_1_1_D,
                video_size=video_size,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_sa * sa_out

        # cross-attn with explicit memory KV concat
        norm_x = modulate(x_B_T_H_W_D, self.layer_norm_cross_attn, scale_ca, shift_ca)
        ca_out = rearrange(
            self.cross_attn(
                rearrange(norm_x, "b t h w d -> b (t h w) d"),
                crossattn_emb,
                memory=memory_M_B_S_D,
            ),
            "b (t h w) d -> b t h w d",
            t=T,
            h=H,
            w=W,
        )
        x_B_T_H_W_D = x_B_T_H_W_D + gate_ca * ca_out

        # mlp
        norm_x = modulate(x_B_T_H_W_D, self.layer_norm_mlp, scale_mlp, shift_mlp)
        mlp_out = self.mlp(norm_x)
        x_B_T_H_W_D = x_B_T_H_W_D + gate_mlp * mlp_out
        return x_B_T_H_W_D


@dataclass
class ProphetHistoryBufferConfig:
    enabled: bool = True
    history_size: int = 60  # T_h


class ProphetMinimalV1LVGDiT(MinimalV1LVGDiT):
    """
    Prophet-augmented DiT for Video2World (ProphRL Section 3.2.3 & 3.2.4).

    Hard changes vs Cosmos:
    - Dual Action Conditioning: scalar MLP + action-frame latent encoder, both injected into timestep embeddings.
    - History-aware memory: FramePack -> memory M, and each block cross-attn explicitly concatenates K/V with memory K/V.
    """

    def __init__(
        self,
        *args: Any,
        action_horizon: int = 12,
        d_model: int = 1024,
        history: ProphetHistoryBufferConfig | None = None,
        history_pool_kernel: tuple[int, int, int] = (3, 2, 2),
        history_pool_stride: tuple[int, int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Enforce Prophet hidden size expectation (Dm=1024) unless caller deliberately changes it.
        if self.model_channels != d_model:
            raise ValueError(f"Prophet requires model_channels (Dm) == {d_model}, got {self.model_channels}")

        self.d_model = d_model
        self.action_horizon = action_horizon
        self.scalar_action = ScalarActionMLP(action_dim=7, action_horizon=action_horizon, d_model=d_model)
        self.action_frame_latent = ActionFrameLatentEncoder(in_channels=WAN21_LATENT_CHANNELS, d_model=d_model)

        self.history_cfg = history if history is not None else ProphetHistoryBufferConfig(enabled=True)
        self._history_latents: torch.Tensor | None = None

        # Determine context dim for cross-attn. If there is a projection, it sets to crossattn_emb_channels.
        context_dim = self.blocks[0].cross_attn.context_dim  # type: ignore[attr-defined]
        self.framepack = FramePack(
            latent_channels=WAN21_LATENT_CHANNELS,
            context_dim=context_dim,
            pool_kernel=history_pool_kernel,
            pool_stride=history_pool_stride,
        )

        # Replace blocks with ProphetBlocks (to implement explicit KV concat).
        # NOTE: this is a structural change and is not checkpoint-compatible with Cosmos weights.
        new_blocks = nn.ModuleList()
        for i, old_block in enumerate(self.blocks):
            # Best-effort: preserve backend choices by looking at underlying Attention backend strings.
            # Self-attn backend is stored as `backend` on Attention.
            sa_backend = getattr(old_block.self_attn, "backend", "transformer_engine")
            ca_backend = getattr(old_block.cross_attn, "backend", "transformer_engine")
            natten_params = getattr(old_block.self_attn, "natten_params", None) if sa_backend == "natten" else None

            new_blocks.append(
                ProphetBlock(
                    x_dim=self.model_channels,
                    context_dim=context_dim,
                    num_heads=self.num_heads,
                    mlp_ratio=getattr(old_block, "mlp_ratio", 4.0),
                    self_attention_backend=sa_backend,
                    cross_attention_backend=ca_backend,
                    natten_params=natten_params,
                    use_adaln_lora=self.use_adaln_lora,
                    adaln_lora_dim=self.adaln_lora_dim,
                )
            )
        self.blocks = new_blocks

    # ---------------------------------------------------------------------
    # History buffer helpers (for closed-loop training; optional in forward)
    # ---------------------------------------------------------------------
    def reset_history(self) -> None:
        self._history_latents = None

    @torch.no_grad()
    def update_history(self, new_latents_B_C_T_H_W: torch.Tensor) -> None:
        if not self.history_cfg.enabled:
            return
        if new_latents_B_C_T_H_W.ndim != 5:
            raise ValueError(f"Expected new latents (B,C,T,H,W), got {new_latents_B_C_T_H_W.shape}")
        if self._history_latents is None:
            self._history_latents = new_latents_B_C_T_H_W.detach()
        else:
            self._history_latents = torch.cat([self._history_latents, new_latents_B_C_T_H_W.detach()], dim=2)
        if self._history_latents.shape[2] > self.history_cfg.history_size:
            self._history_latents = self._history_latents[:, :, -self.history_cfg.history_size :, :, :].contiguous()

    def forward(
        self,
        x_B_C_T_H_W: torch.Tensor,
        timesteps_B_T: torch.Tensor,
        crossattn_emb: torch.Tensor,
        condition_video_input_mask_B_C_T_H_W: torch.Tensor | None = None,
        fps: torch.Tensor | None = None,
        padding_mask: torch.Tensor | None = None,
        data_type: DataType | None = DataType.VIDEO,
        use_cuda_graphs: bool = False,
        *,
        # Dual Action Conditioning inputs:
        # - scalar stream still uses the raw action chunk (B,Ta,7)
        action: torch.Tensor | None = None,
        # - action frame stream MUST receive Wan2.1 VAE-encoded action-frame latents
        action_frame_latents: torch.Tensor | None = None,
        # History:
        history_latents: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        del kwargs

        # Keep original condition-mask behavior.
        if data_type == DataType.VIDEO:
            x_B_C_T_H_W = torch.cat(
                [x_B_C_T_H_W, condition_video_input_mask_B_C_T_H_W.type_as(x_B_C_T_H_W)], dim=1
            )
        else:
            B, _, T, H, W = x_B_C_T_H_W.shape
            x_B_C_T_H_W = torch.cat(
                [
                    x_B_C_T_H_W,
                    torch.zeros((B, 1, T, H, W), dtype=x_B_C_T_H_W.dtype, device=x_B_C_T_H_W.device),
                ],
                dim=1,
            )

        assert isinstance(data_type, DataType), f"Expected DataType, got {type(data_type)}."
        assert not (self.training and use_cuda_graphs), "CUDA Graphs are supported only for inference"

        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb = self.prepare_embedded_sequence(
            x_B_C_T_H_W, fps=fps, padding_mask=padding_mask
        )

        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        # timestep embedding
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        # -------------------- Dual Action Conditioning --------------------
        if action is not None:
            # scalar stream
            a_scalar = self.scalar_action(action.to(dtype=t_embedding_B_T_D.dtype, device=t_embedding_B_T_D.device))
            # broadcast from (B,1,D) to (B,T,D)
            a_scalar = repeat(a_scalar, "b 1 d -> b t d", t=t_embedding_B_T_D.shape[1])
            t_embedding_B_T_D = t_embedding_B_T_D + a_scalar

            # action-frame stream (expects Wan2.1 VAE latents, not pixels)
            if action_frame_latents is not None:
                a_frame = self.action_frame_latent(
                    action_frame_latents.to(dtype=t_embedding_B_T_D.dtype, device=t_embedding_B_T_D.device)
                )  # (B,1,D)
                a_frame = repeat(a_frame, "b 1 d -> b t d", t=t_embedding_B_T_D.shape[1])
                t_embedding_B_T_D = t_embedding_B_T_D + a_frame

        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # -------------------- History-aware Memory --------------------
        hist = history_latents
        if hist is None and self.history_cfg.enabled:
            hist = self._history_latents
        memory_M = None
        if hist is not None:
            memory_M = self.framepack(hist.to(dtype=crossattn_emb.dtype, device=crossattn_emb.device))

        # blocks
        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        for block in blocks:
            x_B_T_H_W_D = block(
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                memory_M_B_S_D=memory_M,
                rope_emb_L_1_1_D=rope_emb_L_1_1_D,
                adaln_lora_B_T_3D=adaln_lora_B_T_3D,
                extra_per_block_pos_emb=extra_pos_emb,
            )

        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        return self.unpatchify(x_B_T_H_W_O)

