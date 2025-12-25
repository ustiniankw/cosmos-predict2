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
Prophet closed-loop rollout utilities.

This file implements the FIFO history-latent buffer logic requested for ProphRL Algorithm 1.

It is intentionally lightweight and does not yet implement the policy/RM/FA-GRPO update.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from cosmos_predict2.tokenizers.wan21_tokenizer import Wan21VideoTokenizer


@dataclass
class ProphetClosedLoopConfig:
    vae_checkpoint_path: str
    history_length: int = 60  # T_h
    rollout_chunk: int = 20  # C


class ProphetClosedLoopSimulator:
    """
    Maintains a FIFO history buffer of Wan2.1 latents (B,16,T_h,H/8,W/8).

    - init(): encode initial_frame and fill buffer by repeating it along time.
    - update_buffer(): encode new frames (C=20) and slide:
        new_buffer = cat([old[:, :, C:], new_latents], dim=2)
    - prepare_next_initial_frame(): last generated frame becomes next initial frame.
    """

    def __init__(self, cfg: ProphetClosedLoopConfig, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.tokenizer = Wan21VideoTokenizer(
            vae_pth=cfg.vae_checkpoint_path,
            device=device,
            dtype=dtype,
            is_amp=False,
        )

        self.history_latent_buffer: torch.Tensor | None = None

    @torch.no_grad()
    def init(self, initial_frame_B_3_H_W: torch.Tensor) -> torch.Tensor:
        """
        Args:
            initial_frame_B_3_H_W: (B,3,H,W) in [-1,1]
        Returns:
            history_latent_buffer: (B,16,T_h,H/8,W/8)
        """
        if initial_frame_B_3_H_W.ndim != 4 or initial_frame_B_3_H_W.shape[1] != 3:
            raise ValueError(f"Expected initial_frame (B,3,H,W), got {initial_frame_B_3_H_W.shape}")

        x0 = initial_frame_B_3_H_W.to(device=self.device, dtype=self.dtype)
        x0_video = x0.unsqueeze(2)  # (B,3,1,H,W)
        x0_lat = self.tokenizer.encode(x0_video)  # (B,16,1,H/8,W/8)

        Th = self.cfg.history_length
        self.history_latent_buffer = x0_lat.repeat(1, 1, Th, 1, 1).contiguous()
        return self.history_latent_buffer

    @torch.no_grad()
    def update_buffer(self, generated_video_B_3_T_H_W: torch.Tensor) -> torch.Tensor:
        """
        Slide FIFO buffer with newly generated frames.

        Args:
            generated_video_B_3_T_H_W: (B,3,C,H,W) in [-1,1], where C == rollout_chunk (default 20)
        Returns:
            history_latent_buffer: (B,16,T_h,H/8,W/8)
        """
        if self.history_latent_buffer is None:
            raise RuntimeError("history_latent_buffer is not initialized. Call init() first.")

        if generated_video_B_3_T_H_W.ndim != 5 or generated_video_B_3_T_H_W.shape[1] != 3:
            raise ValueError(f"Expected generated_video (B,3,T,H,W), got {generated_video_B_3_T_H_W.shape}")

        C = generated_video_B_3_T_H_W.shape[2]
        if C != self.cfg.rollout_chunk:
            raise ValueError(f"Expected rollout_chunk={self.cfg.rollout_chunk}, got {C}")

        new_latents = self.tokenizer.encode(generated_video_B_3_T_H_W.to(device=self.device, dtype=self.dtype))

        old = self.history_latent_buffer
        Th = self.cfg.history_length
        # Slide by the *latent* temporal length returned by the tokenizer.
        # This keeps the FIFO logic correct even when temporal compression is 4x.
        c_lat = new_latents.shape[2]
        new_buffer = torch.cat([old[:, :, c_lat:], new_latents], dim=2)
        if new_buffer.shape[2] != Th:
            raise AssertionError(f"history buffer length mismatch: {new_buffer.shape[2]} != {Th}")
        self.history_latent_buffer = new_buffer.contiguous()
        return self.history_latent_buffer

    @torch.no_grad()
    def prepare_next_initial_frame(self, generated_video_B_3_T_H_W: torch.Tensor) -> torch.Tensor:
        """
        Take the last generated frame as the next initial frame.
        """
        if generated_video_B_3_T_H_W.ndim != 5 or generated_video_B_3_T_H_W.shape[1] != 3:
            raise ValueError(f"Expected generated_video (B,3,T,H,W), got {generated_video_B_3_T_H_W.shape}")
        last_frame = generated_video_B_3_T_H_W[:, :, -1, :, :]  # (B,3,H,W)
        return last_frame

