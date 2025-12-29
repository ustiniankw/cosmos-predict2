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
Initialize Prophet weights from official Cosmos-Predict2-2B-Video2World checkpoint.

Why this script exists
----------------------
Prophet changes the backbone in three ways:
- uses Wan2.1 VAE geometry (latent C=16) while official Cosmos-Predict2 Video2World checkpoints may be C=8
- adds action conditioning modules (scalar + action-frame latent stream)
- replaces cross-attention with a Prophet variant that supports history latents (explicit memory KV concat)

This script maps compatible parameters by name, and applies hand-crafted rules for mismatched parts:
- PatchEmbed input projection: keep old channels, zero-init new channels
- FinalLayer output projection: keep old channels, zero-init new channels
- ActionFrameLatentEncoder: Xavier init (new)
- Other new layers (scalar action MLP + FramePack projection): Xavier init

It also prints a migration report and validates that the produced state_dict loads into
`ProphetMinimalV1LVGDiT` without size-mismatch errors.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Allow running as a standalone script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_predict2.models.prophet_model import ProphetHistoryBufferConfig, ProphetMinimalV1LVGDiT  # noqa: E402
from cosmos_predict2.models.utils import init_weights_on_device, load_state_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Initialize Prophet weights from Cosmos-Predict2-2B-Video2World weights")
    p.add_argument("--cosmos_path", type=str, required=True, help="Path to official Cosmos-Predict2 DiT checkpoint")
    p.add_argument("--output_path", type=str, required=True, help="Where to write the initialized Prophet checkpoint")
    # Optional knobs (safe defaults for 2B)
    p.add_argument("--d_model", type=int, default=2048, help="Prophet/Cosmos model_channels (2B uses 2048)")
    p.add_argument("--num_blocks", type=int, default=28)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--patch_spatial", type=int, default=2)
    p.add_argument("--patch_temporal", type=int, default=1)
    p.add_argument("--history_size", type=int, default=60)
    p.add_argument("--max_img_h", type=int, default=240)
    p.add_argument("--max_img_w", type=int, default=240)
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--verbose", action="store_true")
    return p.parse_args()


def _extract_tensor_state_dict(raw: dict) -> dict[str, torch.Tensor]:
    """
    Normalize various checkpoint formats to a flat {name: tensor} dict.
    """
    # Common wrappers
    for k in ("state_dict", "model", "module", "net", "net_ema"):
        if k in raw and isinstance(raw[k], dict) and all(isinstance(v, torch.Tensor) for v in raw[k].values()):
            raw = raw[k]
            break

    out: dict[str, torch.Tensor] = {}
    for k, v in raw.items():
        if isinstance(k, str) and isinstance(v, torch.Tensor):
            out[k] = v
    return out


def _strip_prefix_if_present(sd: dict[str, torch.Tensor], prefix: str) -> dict[str, torch.Tensor]:
    if not any(k.startswith(prefix) for k in sd):
        return sd
    return {k[len(prefix) :]: v for k, v in sd.items() if k.startswith(prefix)}


def load_cosmos_backbone_state_dict(path: str) -> dict[str, torch.Tensor]:
    """
    Load a Cosmos checkpoint and extract DiT weights (stripping net./net_ema. if present).
    """
    raw = load_state_dict(path)
    sd = _extract_tensor_state_dict(raw)

    # Prefer net. over net_ema. if both exist (mirrors common training save formats).
    if any(k.startswith("net.") for k in sd):
        sd = _strip_prefix_if_present(sd, "net.")
    elif any(k.startswith("net_ema.") for k in sd):
        sd = _strip_prefix_if_present(sd, "net_ema.")

    return sd


def _map_source_key_for_target(target_key: str) -> list[str]:
    """
    Return candidate source keys for a given Prophet target key.
    """
    cands = [target_key]
    # ProphetCrossAttention wraps the original Attention as `cross_attn.base`.
    if ".cross_attn.base." in target_key:
        cands.append(target_key.replace(".cross_attn.base.", ".cross_attn."))
    return cands


def _init_xavier_(t: torch.Tensor) -> torch.Tensor:
    if t.ndim < 2:
        # fallback for 1D (e.g., bias) – keep zeros
        return torch.zeros_like(t)
    out = t.clone()
    nn.init.xavier_uniform_(out)
    return out


def _expand_patch_embed_in_channels(
    *,
    old_w: torch.Tensor,
    new_in_ch_total: int,
    patch_mult: int,
) -> torch.Tensor:
    """
    Expand PatchEmbed linear weight for increased input channels.

    PatchEmbed uses Rearrange -> Linear with input layout "(c r m n)" (channel-major blocks),
    so we can copy contiguous per-channel blocks.
    """
    if old_w.ndim != 2:
        raise ValueError(f"Expected 2D PatchEmbed weight, got {old_w.shape}")

    out_dim, old_in_dim = old_w.shape
    if old_in_dim % patch_mult != 0:
        raise ValueError(f"old_in_dim {old_in_dim} not divisible by patch_mult {patch_mult}")
    old_in_ch_total = old_in_dim // patch_mult

    new_in_dim = new_in_ch_total * patch_mult
    new_w = torch.zeros((out_dim, new_in_dim), dtype=old_w.dtype)

    # Assume the last channel is the condition mask channel (concat_padding_mask=True).
    old_latent_ch = max(old_in_ch_total - 1, 0)
    new_latent_ch = max(new_in_ch_total - 1, 0)

    copy_latent = min(old_latent_ch, new_latent_ch)
    # Copy first `copy_latent` latent channels.
    for c in range(copy_latent):
        src0, src1 = c * patch_mult, (c + 1) * patch_mult
        dst0, dst1 = c * patch_mult, (c + 1) * patch_mult
        new_w[:, dst0:dst1] = old_w[:, src0:src1]

    # Copy mask channel (last).
    if old_in_ch_total >= 1 and new_in_ch_total >= 1:
        src0, src1 = (old_in_ch_total - 1) * patch_mult, old_in_ch_total * patch_mult
        dst0, dst1 = (new_in_ch_total - 1) * patch_mult, new_in_ch_total * patch_mult
        new_w[:, dst0:dst1] = old_w[:, src0:src1]

    return new_w


def _expand_final_layer_out_channels(
    *,
    old_w: torch.Tensor,
    new_out_ch: int,
    patch_mult: int,
) -> torch.Tensor:
    """
    Expand FinalLayer linear weight for increased output channels.

    FinalLayer outputs "(p1 p2 t C)" where channel C is the fastest-changing dimension.
    So we need to copy per-patch-position channel blocks.
    """
    if old_w.ndim != 2:
        raise ValueError(f"Expected 2D FinalLayer weight, got {old_w.shape}")

    old_out_dim, hidden = old_w.shape
    if old_out_dim % patch_mult != 0:
        raise ValueError(f"old_out_dim {old_out_dim} not divisible by patch_mult {patch_mult}")
    old_out_ch = old_out_dim // patch_mult

    new_out_dim = new_out_ch * patch_mult
    new_w = torch.zeros((new_out_dim, hidden), dtype=old_w.dtype)

    copy_ch = min(old_out_ch, new_out_ch)
    for p in range(patch_mult):
        src0, src1 = p * old_out_ch, p * old_out_ch + copy_ch
        dst0, dst1 = p * new_out_ch, p * new_out_ch + copy_ch
        new_w[dst0:dst1, :] = old_w[src0:src1, :]

    return new_w


def build_prophet_model_on_meta(args: argparse.Namespace) -> ProphetMinimalV1LVGDiT:
    # Prophet uses in/out channels = 16 for Wan2.1 latents.
    with init_weights_on_device(torch.device("meta")):
        model = ProphetMinimalV1LVGDiT(
            max_img_h=args.max_img_h,
            max_img_w=args.max_img_w,
            max_frames=args.max_frames,
            in_channels=16,
            out_channels=16,
            patch_spatial=args.patch_spatial,
            patch_temporal=args.patch_temporal,
            concat_padding_mask=True,
            model_channels=args.d_model,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            atten_backend="minimal_a2a",
            pos_emb_cls="rope3d",
            pos_emb_learnable=True,
            pos_emb_interpolation="crop",
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=1.0,
            extra_per_block_abs_pos_emb=False,
            rope_enable_fps_modulation=False,
            action_horizon=20,
            d_model=args.d_model,
            history=ProphetHistoryBufferConfig(enabled=True, history_size=args.history_size),
        )
    return model


def main() -> None:
    args = parse_args()

    cosmos_sd = load_cosmos_backbone_state_dict(args.cosmos_path)
    prophet = build_prophet_model_on_meta(args)
    target_sd = prophet.state_dict()

    # Stats
    copied_elems = 0
    inited_elems = 0
    copied_keys: list[str] = []
    inited_keys: list[str] = []
    skipped_missing_keys: list[str] = []
    skipped_shape_mismatch: list[tuple[str, torch.Size, torch.Size]] = []

    out_sd: dict[str, torch.Tensor] = {}

    patch_mult = args.patch_spatial * args.patch_spatial * args.patch_temporal

    def _keep_target_default(k: str, t: torch.Tensor) -> None:
        nonlocal inited_elems
        out_sd[k] = t.detach().to(torch.float32).cpu()
        inited_keys.append(k)
        inited_elems += t.numel()

    for k_tgt, t_tgt in target_sd.items():
        # Custom rule 1: PatchEmbed input projection (channel expansion).
        if k_tgt == "x_embedder.proj.1.weight":
            # Find matching source key.
            src_key = "x_embedder.proj.1.weight"
            if src_key not in cosmos_sd:
                skipped_missing_keys.append(k_tgt)
                out_sd[k_tgt] = torch.zeros(t_tgt.shape, dtype=torch.float32)
                inited_keys.append(k_tgt)
                inited_elems += t_tgt.numel()
                continue
            old_w = cosmos_sd[src_key].to(torch.float32)
            # New in_channels includes mask channel due to concat_padding_mask=True.
            new_in_dim = t_tgt.shape[1]
            if new_in_dim % patch_mult != 0:
                raise RuntimeError(f"Unexpected new PatchEmbed in_dim: {new_in_dim} (patch_mult={patch_mult})")
            new_in_ch_total = new_in_dim // patch_mult
            new_w = _expand_patch_embed_in_channels(old_w=old_w, new_in_ch_total=new_in_ch_total, patch_mult=patch_mult)
            if new_w.shape != t_tgt.shape:
                raise RuntimeError(f"PatchEmbed expanded shape mismatch: {new_w.shape} vs {t_tgt.shape}")
            out_sd[k_tgt] = new_w
            copied_keys.append(k_tgt)
            copied_elems += (old_w.numel())  # report copied source elems (not counting zeros)
            inited_elems += (new_w.numel() - old_w.numel())
            continue

        # Custom rule 2: FinalLayer output projection (channel expansion).
        if k_tgt == "final_layer.linear.weight":
            src_key = "final_layer.linear.weight"
            if src_key not in cosmos_sd:
                skipped_missing_keys.append(k_tgt)
                out_sd[k_tgt] = torch.zeros(t_tgt.shape, dtype=torch.float32)
                inited_keys.append(k_tgt)
                inited_elems += t_tgt.numel()
                continue
            old_w = cosmos_sd[src_key].to(torch.float32)
            new_out_dim = t_tgt.shape[0]
            if new_out_dim % patch_mult != 0:
                raise RuntimeError(f"Unexpected new FinalLayer out_dim: {new_out_dim} (patch_mult={patch_mult})")
            new_out_ch = new_out_dim // patch_mult
            new_w = _expand_final_layer_out_channels(old_w=old_w, new_out_ch=new_out_ch, patch_mult=patch_mult)
            if new_w.shape != t_tgt.shape:
                raise RuntimeError(f"FinalLayer expanded shape mismatch: {new_w.shape} vs {t_tgt.shape}")
            out_sd[k_tgt] = new_w
            copied_keys.append(k_tgt)
            copied_elems += old_w.numel()
            inited_elems += (new_w.numel() - old_w.numel())
            continue

        # Custom rule 3: ActionFrameLatentEncoder (new) -> Xavier init.
        if k_tgt.startswith("action_frame_latent."):
            # Conv3d weights: Xavier; biases (if any) -> zeros
            if k_tgt.endswith(".weight"):
                out_sd[k_tgt] = _init_xavier_(torch.empty(t_tgt.shape, dtype=torch.float32))
            else:
                out_sd[k_tgt] = torch.zeros(t_tgt.shape, dtype=torch.float32)
            inited_keys.append(k_tgt)
            inited_elems += t_tgt.numel()
            continue

        # Additional new layers (not in Cosmos): scalar_action + framepack.
        if k_tgt.startswith("scalar_action.") or k_tgt.startswith("framepack."):
            if k_tgt.endswith(".weight"):
                out_sd[k_tgt] = _init_xavier_(torch.empty(t_tgt.shape, dtype=torch.float32))
            else:
                out_sd[k_tgt] = torch.zeros(t_tgt.shape, dtype=torch.float32)
            inited_keys.append(k_tgt)
            inited_elems += t_tgt.numel()
            continue

        # Default: try name mapping and copy if shape matches.
        src_tensor = None
        src_key_used = None
        for k_src in _map_source_key_for_target(k_tgt):
            if k_src in cosmos_sd:
                src_tensor = cosmos_sd[k_src]
                src_key_used = k_src
                break

        if src_tensor is None:
            skipped_missing_keys.append(k_tgt)
            # If this is a non-meta tensor (typically a buffer), keep the model default.
            if not getattr(t_tgt, "is_meta", False):
                _keep_target_default(k_tgt, t_tgt)
            else:
                # Initialize leftover parameters safely: zeros for 1D, Xavier for 2D+.
                out_sd[k_tgt] = (
                    _init_xavier_(torch.empty(t_tgt.shape, dtype=torch.float32))
                    if t_tgt.ndim >= 2
                    else torch.zeros(t_tgt.shape, dtype=torch.float32)
                )
                inited_keys.append(k_tgt)
                inited_elems += t_tgt.numel()
            continue

        if tuple(src_tensor.shape) != tuple(t_tgt.shape):
            skipped_shape_mismatch.append((k_tgt, src_tensor.shape, t_tgt.shape))
            if not getattr(t_tgt, "is_meta", False):
                _keep_target_default(k_tgt, t_tgt)
            else:
                out_sd[k_tgt] = (
                    _init_xavier_(torch.empty(t_tgt.shape, dtype=torch.float32))
                    if t_tgt.ndim >= 2
                    else torch.zeros(t_tgt.shape, dtype=torch.float32)
                )
                inited_keys.append(k_tgt)
                inited_elems += t_tgt.numel()
            continue

        out_sd[k_tgt] = src_tensor.detach().to(torch.float32).cpu()
        copied_keys.append(k_tgt if src_key_used == k_tgt else f"{k_tgt} <= {src_key_used}")
        copied_elems += src_tensor.numel()

    # Save
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(out_sd, str(output_path))

    # Validation: instantiate on meta and load with assign=True (will error on size mismatch).
    with init_weights_on_device(torch.device("meta")):
        check = ProphetMinimalV1LVGDiT(
            max_img_h=args.max_img_h,
            max_img_w=args.max_img_w,
            max_frames=args.max_frames,
            in_channels=16,
            out_channels=16,
            patch_spatial=args.patch_spatial,
            patch_temporal=args.patch_temporal,
            concat_padding_mask=True,
            model_channels=args.d_model,
            num_blocks=args.num_blocks,
            num_heads=args.num_heads,
            atten_backend="minimal_a2a",
            pos_emb_cls="rope3d",
            pos_emb_learnable=True,
            pos_emb_interpolation="crop",
            use_adaln_lora=True,
            adaln_lora_dim=256,
            rope_h_extrapolation_ratio=3.0,
            rope_w_extrapolation_ratio=3.0,
            rope_t_extrapolation_ratio=1.0,
            extra_per_block_abs_pos_emb=False,
            rope_enable_fps_modulation=False,
            action_horizon=20,
            d_model=args.d_model,
            history=ProphetHistoryBufferConfig(enabled=True, history_size=args.history_size),
        )
    # If any tensor has wrong shape, this will throw a RuntimeError.
    missing, unexpected = check.load_state_dict(out_sd, strict=False, assign=True)

    # Report
    total_elems = sum(v.numel() for v in target_sd.values())
    print("=== Prophet init report ===")
    print(f"- cosmos checkpoint: {args.cosmos_path}")
    print(f"- output checkpoint: {args.output_path}")
    print(f"- copied params: {copied_elems:,} / {total_elems:,} elements")
    print(f"- newly initialized params: {inited_elems:,} / {total_elems:,} elements")
    print(f"- missing keys on load_state_dict (strict=False): {len(missing)}")
    print(f"- unexpected keys on load_state_dict (strict=False): {len(unexpected)}")
    if args.verbose:
        if skipped_shape_mismatch:
            print("\n[shape mismatch -> initialized]")
            for k, s0, s1 in skipped_shape_mismatch[:100]:
                print(f"  {k}: source {tuple(s0)} != target {tuple(s1)}")
            if len(skipped_shape_mismatch) > 100:
                print(f"  ... ({len(skipped_shape_mismatch) - 100} more)")
        if skipped_missing_keys:
            print("\n[missing in source -> initialized]")
            for k in skipped_missing_keys[:100]:
                print(f"  {k}")
            if len(skipped_missing_keys) > 100:
                print(f"  ... ({len(skipped_missing_keys) - 100} more)")
    print("✅ Load verification passed (no size mismatch).")


if __name__ == "__main__":
    main()

