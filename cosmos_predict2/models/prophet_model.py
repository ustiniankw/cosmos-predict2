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
from einops import rearrange

from cosmos_predict2.conditioner import DataType
from cosmos_predict2.models.video2world_dit import MinimalV1LVGDiT
from imaginaire.utils.graph import create_cuda_graph


class _MLP(nn.Module):
    """Small MLP used for scalar action embedding."""

    def __init__(self, in_features: int, out_features: int, hidden_mult: int = 4) -> None:
        super().__init__()
        hidden = out_features * hidden_mult
        self.net = nn.Sequential(
            nn.Linear(in_features, hidden),
            nn.GELU(approximate="tanh"),
            nn.Linear(hidden, out_features),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionFrameRenderer(nn.Module):
    """
    Differentiable "renderer" from 7D actions to small 2D action frames.

    Prophet describes projecting and rendering actions into 2D frames. The paper does not mandate a
    specific renderer; here we implement a lightweight linear renderer:

        action[t] (7,) -> frame[t] (C, H, W)
        frame[t] = sum_i action_i[t] * basis_i

    where `basis_i` is learnable.
    """

    def __init__(self, *, action_dim: int = 7, channels: int = 8, height: int = 16, width: int = 16) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.channels = channels
        self.height = height
        self.width = width
        self.basis = nn.Parameter(torch.zeros(action_dim, channels, height, width))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.basis, std=0.02)

    def forward(self, action_B_T_D: torch.Tensor) -> torch.Tensor:
        """
        Args:
            action_B_T_D: (B, T, 7) or (B, T, D) where D==action_dim
        Returns:
            frames_B_C_T_H_W: (B, C, T, H, W)
        """
        if action_B_T_D.ndim != 3:
            raise ValueError(f"Expected action of shape (B,T,D), got {action_B_T_D.shape}")
        if action_B_T_D.shape[-1] != self.action_dim:
            raise ValueError(
                f"Expected action last dim={self.action_dim}, got {action_B_T_D.shape[-1]} (shape={action_B_T_D.shape})"
            )
        # (B, T, D) x (D, C, H, W) -> (B, T, C, H, W)
        frames_B_T_C_H_W = torch.einsum("btd,dchw->btchw", action_B_T_D, self.basis)
        return frames_B_T_C_H_W.permute(0, 2, 1, 3, 4).contiguous()


class ActionFrameEncoder(nn.Module):
    """3D conv + pooling to turn rendered action frames into a global embedding."""

    def __init__(
        self,
        *,
        in_channels: int,
        out_dim: int,
        hidden_channels: int = 128,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GELU(approximate="tanh"),
            nn.Conv3d(hidden_channels, out_dim, kernel_size=1, bias=False),
            nn.AdaptiveAvgPool3d((1, 1, 1)),
        )

    def forward(self, frames_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        x = self.net(frames_B_C_T_H_W)
        return x.flatten(1)  # (B, out_dim)


class HistoryEncoder(nn.Module):
    """
    History-aware external memory encoder.

    Turns latent history (B, C, T_h, H, W) into cross-attention tokens (B, M, D_ctx) using:
      3D avg pool -> flatten spatiotemporal positions -> linear projection.
    """

    def __init__(
        self,
        *,
        latent_channels: int,
        context_dim: int,
        pool_kernel: tuple[int, int, int] = (3, 4, 4),
        pool_stride: tuple[int, int, int] | None = None,
    ) -> None:
        super().__init__()
        self.pool_kernel = pool_kernel
        self.pool_stride = pool_stride if pool_stride is not None else pool_kernel
        self.pool = nn.AvgPool3d(kernel_size=self.pool_kernel, stride=self.pool_stride, ceil_mode=False)
        self.proj = nn.Linear(latent_channels, context_dim, bias=False)

    def forward(self, history_latents_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        if history_latents_B_C_T_H_W.ndim != 5:
            raise ValueError(f"Expected history latents (B,C,T,H,W), got {history_latents_B_C_T_H_W.shape}")
        pooled = self.pool(history_latents_B_C_T_H_W)  # (B, C, T', H', W')
        tokens = rearrange(pooled, "b c t h w -> b (t h w) c").contiguous()
        return self.proj(tokens)  # (B, M, D_ctx)


@dataclass
class ProphetHistoryBufferConfig:
    enabled: bool = True
    history_size: int = 60  # T_h


class ProphetMinimalV1LVGDiT(MinimalV1LVGDiT):
    """
    Prophet-augmented DiT for Video2World.

    This is designed as a drop-in replacement for `MinimalV1LVGDiT`, with extra optional kwargs:
    - `action`: (B, T_a, 7) action chunk.
    - `history_latents`: (B, C, T_h, H, W) latent history frames for external memory.
    """

    def __init__(
        self,
        *args: Any,
        action_chunk_dim: int | None = None,
        action_dim: int = 7,
        action_frame_channels: int = 8,
        action_frame_size: tuple[int, int] = (16, 16),
        use_action_frame_stream: bool = True,
        use_scalar_action_stream: bool = True,
        inject_action_into_adaln_lora: bool = False,
        history: ProphetHistoryBufferConfig | None = None,
        history_pool_kernel: tuple[int, int, int] = (3, 4, 4),
        history_pool_stride: tuple[int, int, int] | None = None,
        **kwargs: Any,
    ) -> None:
        # `MinimalV1LVGDiT` already increments in_channels for condition mask.
        super().__init__(*args, **kwargs)

        self.use_scalar_action_stream = use_scalar_action_stream
        self.use_action_frame_stream = use_action_frame_stream
        self.inject_action_into_adaln_lora = inject_action_into_adaln_lora

        self._action_dim = action_dim
        self._action_chunk_dim = action_chunk_dim  # if None, infer at runtime from input action shape

        # Cross-attn context dimension (after optional `crossattn_proj`).
        self.crossattn_context_dim = self.blocks[0].cross_attn.context_dim  # type: ignore[attr-defined]

        # -------------------------
        # Dual Action Conditioning
        # -------------------------
        # Scalar stream: action chunk -> global embedding.
        if self.use_scalar_action_stream:
            # The action chunk is typically (B, T_a, 7). We flatten to (B, T_a*7).
            # If action_chunk_dim is not provided, we lazily instantiate the MLP on first forward.
            self.scalar_action_mlp: nn.Module | None = None
        else:
            self.scalar_action_mlp = None

        # Action-frame stream: 7D -> frames -> 3D conv -> embedding.
        if self.use_action_frame_stream:
            h, w = action_frame_size
            self.action_frame_renderer = ActionFrameRenderer(
                action_dim=action_dim, channels=action_frame_channels, height=h, width=w
            )
            self.action_frame_encoder = ActionFrameEncoder(
                in_channels=action_frame_channels, out_dim=self.model_channels
            )
        else:
            self.action_frame_renderer = None
            self.action_frame_encoder = None

        # -------------------------
        # History-aware Mechanism
        # -------------------------
        self.history_cfg = history if history is not None else ProphetHistoryBufferConfig(enabled=False)
        self._history_latents: torch.Tensor | None = None
        self.history_encoder = HistoryEncoder(
            latent_channels=self.in_channels,  # NOTE: expects latent channels BEFORE adding mask channel
            context_dim=self.crossattn_context_dim,
            pool_kernel=history_pool_kernel,
            pool_stride=history_pool_stride,
        )

    # ---------------------------------------------------------------------
    # History buffer helpers (used by closed-loop rollouts; optional in fwd)
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
            # If batch/shape mismatch, reset history (closed-loop runner should keep shapes consistent).
            if (
                self._history_latents.shape[0] != new_latents_B_C_T_H_W.shape[0]
                or self._history_latents.shape[1] != new_latents_B_C_T_H_W.shape[1]
                or self._history_latents.shape[3:] != new_latents_B_C_T_H_W.shape[3:]
            ):
                self._history_latents = new_latents_B_C_T_H_W.detach()
            else:
                self._history_latents = torch.cat([self._history_latents, new_latents_B_C_T_H_W.detach()], dim=2)

        # Keep last T_h frames.
        if self._history_latents.shape[2] > self.history_cfg.history_size:
            self._history_latents = self._history_latents[:, :, -self.history_cfg.history_size :, :, :].contiguous()

    def _build_scalar_action_mlp_if_needed(self, action_B_T_D: torch.Tensor) -> None:
        if self.scalar_action_mlp is not None:
            return
        if self._action_chunk_dim is not None:
            in_dim = self._action_chunk_dim
        else:
            if action_B_T_D.ndim != 3:
                raise ValueError(f"Expected action shape (B,T,D), got {action_B_T_D.shape}")
            in_dim = action_B_T_D.shape[1] * action_B_T_D.shape[2]
        self.scalar_action_mlp = _MLP(in_dim, self.model_channels)

    def _compute_action_embedding(self, action_B_T_D: torch.Tensor) -> torch.Tensor:
        """
        Returns a global action embedding shaped (B, 1, D_model) to be broadcast across video timesteps.
        """
        if action_B_T_D.ndim != 3:
            raise ValueError(f"Expected action shape (B,T,D), got {action_B_T_D.shape}")
        B = action_B_T_D.shape[0]
        flat = rearrange(action_B_T_D, "b t d -> b (t d)")

        emb_parts: list[torch.Tensor] = []

        if self.use_scalar_action_stream:
            self._build_scalar_action_mlp_if_needed(action_B_T_D)
            assert self.scalar_action_mlp is not None
            emb_parts.append(self.scalar_action_mlp(flat))  # (B, D_model)

        if self.use_action_frame_stream:
            assert self.action_frame_renderer is not None
            assert self.action_frame_encoder is not None
            frames = self.action_frame_renderer(action_B_T_D.to(dtype=self.action_frame_renderer.basis.dtype))
            emb_parts.append(self.action_frame_encoder(frames))  # (B, D_model)

        if not emb_parts:
            return torch.zeros((B, 1, self.model_channels), device=action_B_T_D.device, dtype=action_B_T_D.dtype)

        emb = torch.stack(emb_parts, dim=0).sum(dim=0)  # (B, D_model)
        return emb.unsqueeze(1)  # (B, 1, D_model)

    def _compute_history_tokens(self, history_latents_B_C_T_H_W: torch.Tensor) -> torch.Tensor:
        return self.history_encoder(history_latents_B_C_T_H_W.to(dtype=self.history_encoder.proj.weight.dtype))

    # ----------------------------- forward -----------------------------
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
        action: torch.Tensor | None = None,
        history_latents: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> torch.Tensor | list[torch.Tensor] | tuple[torch.Tensor, list[torch.Tensor]]:
        # Keep compatibility with other callers that may pass extra keys.
        del kwargs

        # --- same mask-channel behavior as MinimalV1LVGDiT ---
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

        # --- patch+pos embed ---
        x_B_T_H_W_D, rope_emb_L_1_1_D, extra_pos_emb_B_T_H_W_D = self.prepare_embedded_sequence(
            x_B_C_T_H_W,
            fps=fps,
            padding_mask=padding_mask,
        )

        # --- cross-attn context projection (text) ---
        if self.crossattn_proj is not None:
            crossattn_emb = self.crossattn_proj(crossattn_emb)

        # --- history external memory: concat to cross-attn context ---
        hist_src = history_latents
        if hist_src is None and self.history_cfg.enabled:
            hist_src = self._history_latents
        if hist_src is not None:
            history_tokens = self._compute_history_tokens(hist_src).to(device=crossattn_emb.device, dtype=crossattn_emb.dtype)
            crossattn_emb = torch.cat([crossattn_emb, history_tokens], dim=1)

        # --- timestep embedding ---
        if timesteps_B_T.ndim == 1:
            timesteps_B_T = timesteps_B_T.unsqueeze(1)
        t_embedding_B_T_D, adaln_lora_B_T_3D = self.t_embedder(timesteps_B_T)

        # --- dual action conditioning: inject into timestep embedding ---
        if action is not None:
            if action.ndim != 3:
                raise ValueError(f"Expected action (B,T,7), got {action.shape}")
            if action.shape[-1] != self._action_dim:
                raise ValueError(f"Expected action last dim={self._action_dim}, got {action.shape[-1]} (shape={action.shape})")
            action_emb_B_1_D = self._compute_action_embedding(action).to(device=t_embedding_B_T_D.device, dtype=t_embedding_B_T_D.dtype)
            t_embedding_B_T_D = t_embedding_B_T_D + action_emb_B_1_D
            if self.inject_action_into_adaln_lora and adaln_lora_B_T_3D is not None:
                # Project the same action embedding to 3*D for AdaLN-LoRA modulation.
                # This is optional; Prophet only requires timestep injection.
                proj_3d = torch.cat([action_emb_B_1_D, action_emb_B_1_D, action_emb_B_1_D], dim=-1)
                adaln_lora_B_T_3D = adaln_lora_B_T_3D + proj_3d

        t_embedding_B_T_D = self.t_embedding_norm(t_embedding_B_T_D)

        # --- blocks ---
        if use_cuda_graphs:
            shapes_key = create_cuda_graph(
                self.cuda_graphs,
                self.blocks,
                x_B_T_H_W_D,
                t_embedding_B_T_D,
                crossattn_emb,
                rope_emb_L_1_1_D,
                adaln_lora_B_T_3D,
                extra_pos_emb_B_T_H_W_D,
            )
            blocks = self.cuda_graphs[shapes_key]
        else:
            blocks = self.blocks

        block_kwargs = {
            "rope_emb_L_1_1_D": rope_emb_L_1_1_D,
            "adaln_lora_B_T_3D": adaln_lora_B_T_3D,
            "extra_per_block_pos_emb": extra_pos_emb_B_T_H_W_D,
        }
        for block in blocks:
            x_B_T_H_W_D = block(x_B_T_H_W_D, t_embedding_B_T_D, crossattn_emb, **block_kwargs)

        # --- final ---
        x_B_T_H_W_O = self.final_layer(x_B_T_H_W_D, t_embedding_B_T_D, adaln_lora_B_T_3D=adaln_lora_B_T_3D)
        return self.unpatchify(x_B_T_H_W_O)

