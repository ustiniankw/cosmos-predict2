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
Wan2.1 Video Autoencoder tokenizer wrapper.

This provides a `VideoTokenizerInterface` implementation that matches Prophet's requirement:
  - latent channels C_l = 16
  - spatial compression 8x8
  - temporal compression 4x

Upstream reference:
  https://github.com/Wan-Video/Wan2.1 (wan/modules/vae.py)
"""

from __future__ import annotations

from contextlib import nullcontext

import torch

from cosmos_predict2.third_party.wan21.vae import WanVAE_
from cosmos_predict2.tokenizers.interface import VideoTokenizerInterface
from imaginaire.utils import log


class Wan21VideoTokenizer(torch.nn.Module, VideoTokenizerInterface):
    """
    A thin wrapper around Wan2.1's `WanVAE_` that exposes the Cosmos tokenizer interface.

    Expected input/output:
      - encode: state (B,3,T,H,W) in [-1,1] -> latents (B,16,T',H/8,W/8)
      - decode: latents -> recon (B,3,T,H,W) in [-1,1]
    """

    def __init__(
        self,
        *,
        vae_pth: str | None = None,
        vae_checkpoint_path: str | None = None,
        name: str = "wan21_tokenizer",
        dtype: torch.dtype = torch.bfloat16,
        device: str = "cuda",
        is_amp: bool = False,
        chunk_duration: int = 81,
        apply_mean_std: bool = True,
        z_dim: int = 16,
        dim: int = 96,
        dim_mult: list[int] | None = None,
        num_res_blocks: int = 2,
        attn_scales: list[float] | None = None,
        temperal_downsample: list[bool] | None = None,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        # Allow passing either `vae_pth` (legacy) or `vae_checkpoint_path` (config-driven).
        if vae_checkpoint_path is not None:
            if vae_pth is not None and vae_pth != vae_checkpoint_path:
                raise ValueError("Pass only one of `vae_pth` or `vae_checkpoint_path` (or make them identical).")
            vae_pth = vae_checkpoint_path
        if vae_pth is None or vae_pth == "":
            raise ValueError("Wan21VideoTokenizer requires `vae_checkpoint_path` (or `vae_pth`) to load weights.")
        self._name = name
        self._dtype = dtype
        self._device = device
        self._is_amp = is_amp
        self._chunk_duration = chunk_duration
        self._apply_mean_std = apply_mean_std

        # Prophet / Wan2.1 fixed geometry.
        self._latent_ch = 16
        self._spatial_compression_factor = 8
        self._temporal_compression_factor = 4

        if z_dim != self._latent_ch:
            raise ValueError(f"Wan2.1 requires z_dim={self._latent_ch}, got {z_dim}")

        dim_mult = dim_mult if dim_mult is not None else [1, 2, 4, 4]
        attn_scales = attn_scales if attn_scales is not None else []
        temperal_downsample = temperal_downsample if temperal_downsample is not None else [False, True, True]

        # Construct model on CPU then load.
        model = WanVAE_(
            dim=dim,
            z_dim=z_dim,
            dim_mult=dim_mult,
            num_res_blocks=num_res_blocks,
            attn_scales=attn_scales,
            temperal_downsample=temperal_downsample,
            dropout=dropout,
        )
        ckpt = torch.load(vae_pth, map_location="cpu", weights_only=False)
        if isinstance(ckpt, dict) and "state_dict" in ckpt and isinstance(ckpt["state_dict"], dict):
            ckpt = ckpt["state_dict"]
        missing, unexpected = model.load_state_dict(ckpt, strict=False)
        if missing or unexpected:
            log.warning(f"Wan2.1 VAE load_state_dict: missing={len(missing)} unexpected={len(unexpected)}")

        self.model = model.eval().requires_grad_(False).to(device=self._device)

        # Wan2.1 uses a fixed per-channel scale (mean/std) for latents.
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.register_buffer("_latent_mean", torch.tensor(mean, dtype=torch.float32).view(1, 16, 1, 1, 1), persistent=False)
        self.register_buffer("_latent_inv_std", (1.0 / torch.tensor(std, dtype=torch.float32)).view(1, 16, 1, 1, 1), persistent=False)

        if self._is_amp:
            self._context = torch.amp.autocast("cuda", dtype=self._dtype)
        else:
            self._context = nullcontext()
            self.model = self.model.to(dtype=self._dtype)

        log.info(f"Built Wan2.1 tokenizer '{self._name}' from: {vae_pth}")

    # -------- interface --------
    @property
    def latent_ch(self) -> int:
        return self._latent_ch

    @property
    def spatial_compression_factor(self) -> int:
        return self._spatial_compression_factor

    @property
    def temporal_compression_factor(self) -> int:
        return self._temporal_compression_factor

    @property
    def pixel_chunk_duration(self) -> int:
        return self._chunk_duration

    @property
    def latent_chunk_duration(self) -> int:
        return self.get_latent_num_frames(self._chunk_duration)

    @property
    def spatial_resolution(self) -> int:
        # Not used as a hard constraint in Cosmos; keep a placeholder.
        return 512

    @property
    def name(self) -> str:
        return self._name

    def reset_dtype(self):
        # No-op: weights are already in correct dtype.
        return

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        # Matches existing Cosmos tokenizer: T' = 1 + floor((T-1)/4)
        return 1 + (num_pixel_frames - 1) // 4

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return (num_latent_frames - 1) * 4 + 1

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """
        state: (B,3,T,H,W) in [-1,1]
        """
        if state.ndim != 5 or state.shape[1] != 3:
            raise ValueError(f"Expected state (B,3,T,H,W), got {state.shape}")
        in_dtype = state.dtype
        scale = [self._latent_mean.to(device=state.device), self._latent_inv_std.to(device=state.device)]
        with self._context:
            x = state.to(self._dtype)
            lat = self.model.encode(x, scale)
        return lat.to(in_dtype)

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """
        latent: (B,16,T',H',W')
        """
        if latent.ndim != 5 or latent.shape[1] != 16:
            raise ValueError(f"Expected latent (B,16,T,H,W), got {latent.shape}")
        in_dtype = latent.dtype
        scale = [self._latent_mean.to(device=latent.device), self._latent_inv_std.to(device=latent.device)]
        with self._context:
            z = latent.to(self._dtype)
            vid = self.model.decode(z, scale)
        return vid.clamp(-1.0, 1.0).to(in_dtype)

