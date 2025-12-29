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
Debug Wan2.1 VAE reconstruction on a LIBERO frame.

Pipeline:
  Original RGB (uint8) -> normalize to [-1,1] -> VAE encode -> VAE decode -> Reconstructed RGB (uint8)

If reconstruction is clear:
  - VAE loading + normalization are likely correct
  - remaining issues likely come from DiT / weight mapping / conditioning

If reconstruction is garbled:
  - VAE checkpoint loading or normalization / channel order may be wrong
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# Allow running as a standalone script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_predict2.tokenizers.wan21_tokenizer import Wan21VideoTokenizer  # noqa: E402
from cosmos_predict2.utils.libero_adapter import load_libero_hdf5_trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Debug Wan2.1 VAE reconstruction on LIBERO")
    p.add_argument("--vae_path", type=str, required=True, help="Path to Wan2.1_VAE.pth")
    p.add_argument(
        "--hdf5_path",
        type=str,
        default="",
        help="Optional explicit LIBERO HDF5 file. If empty, pick the first .hdf5 under --libero_dir.",
    )
    p.add_argument("--libero_dir", type=str, default="./datasets/libero_spatial", help="Directory containing LIBERO HDF5 files")
    p.add_argument("--demo_index", type=int, default=0)
    p.add_argument("--frame_index", type=int, default=0)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--out_dir", type=str, default="output/vae_reconstruct")
    return p.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[s]


def _to_tensor_b3hw_minus1_1(img_u8_hwc: np.ndarray, device: str) -> torch.Tensor:
    if img_u8_hwc.ndim != 3 or img_u8_hwc.shape[-1] != 3:
        raise ValueError(f"Expected HWC uint8 RGB, got {img_u8_hwc.shape} dtype={img_u8_hwc.dtype}")
    x = torch.from_numpy(img_u8_hwc).to(device=device)
    x = x.permute(2, 0, 1).contiguous().float()  # (3,H,W) float32
    x = x / 127.5 - 1.0
    return x


def _to_uint8_hwc(x_b3hw_minus1_1: torch.Tensor) -> np.ndarray:
    if x_b3hw_minus1_1.ndim != 4 or x_b3hw_minus1_1.shape[1] != 3:
        raise ValueError(f"Expected (B,3,H,W), got {tuple(x_b3hw_minus1_1.shape)}")
    x = x_b3hw_minus1_1.detach().float().clamp(-1, 1)
    x = (x[0].permute(1, 2, 0) * 0.5 + 0.5) * 255.0
    return x.round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def _psnr(mse: float) -> float:
    if mse <= 1e-12:
        return 99.0
    return float(10.0 * np.log10((255.0 * 255.0) / mse))


def main() -> None:
    args = parse_args()

    # Pick an HDF5 file.
    hdf5_path = args.hdf5_path
    if not hdf5_path:
        libero_dir = Path(args.libero_dir)
        cands = sorted([p for p in libero_dir.rglob("*.hdf5")])
        if not cands:
            raise FileNotFoundError(f"No .hdf5 found under: {libero_dir}")
        hdf5_path = str(cands[0])

    traj = load_libero_hdf5_trajectory(hdf5_path, demo_index=args.demo_index, camera=args.camera)
    if args.frame_index < 0 or args.frame_index >= traj.rgb.shape[0]:
        raise IndexError(f"frame_index out of range: {args.frame_index} (T={traj.rgb.shape[0]})")

    img_u8 = traj.rgb[args.frame_index]  # (H,W,3) uint8

    device = args.device
    dtype = _dtype_from_str(args.dtype)

    # Load Wan2.1 tokenizer/VAE.
    tok = Wan21VideoTokenizer(
        vae_checkpoint_path=args.vae_path,
        device=device,
        dtype=dtype,
        is_amp=False,
    )

    # Encode/decode a single-frame "video" (B,3,T,H,W) with T=1.
    x0 = _to_tensor_b3hw_minus1_1(img_u8, device=device).unsqueeze(0)  # (1,3,H,W)
    x0_vid = x0.unsqueeze(2)  # (1,3,1,H,W)

    with torch.no_grad():
        lat = tok.encode(x0_vid.to(dtype=dtype))
        rec = tok.decode(lat.to(dtype=dtype))  # (1,3,1,H,W) in [-1,1]

    rec_frame = rec[:, :, 0, :, :]  # (1,3,H,W)

    # Metrics (uint8 space)
    rec_u8 = _to_uint8_hwc(rec_frame)
    mse = float(np.mean((img_u8.astype(np.float32) - rec_u8.astype(np.float32)) ** 2))
    psnr = _psnr(mse)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    orig_path = out_dir / "orig.png"
    rec_path = out_dir / "recon.png"
    side_path = out_dir / "side_by_side.png"

    Image.fromarray(img_u8).save(orig_path)
    Image.fromarray(rec_u8).save(rec_path)
    Image.fromarray(np.concatenate([img_u8, rec_u8], axis=1)).save(side_path)

    print("=== Wan2.1 VAE reconstruction ===")
    print(f"- hdf5: {hdf5_path}")
    print(f"- demo_index: {args.demo_index} frame_index: {args.frame_index} camera: {args.camera}")
    print(f"- device: {device} dtype: {args.dtype}")
    print(f"- latents shape: {tuple(lat.shape)}")
    print(f"- MSE (uint8): {mse:.4f}")
    print(f"- PSNR (dB): {psnr:.2f}")
    print(f"- saved: {orig_path}")
    print(f"- saved: {rec_path}")
    print(f"- saved: {side_path}")


if __name__ == "__main__":
    main()

