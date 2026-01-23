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
Prophet closed-loop validation on LIBERO (ProphRL Algorithm 1).

Core loop:
  - take a LIBERO demo's first frame x0
  - initialize history buffer (T_h=60)
  - repeat 5 times:
      - feed next 20 GT actions (servo commands)
      - generate next 20 frames (chunk)
      - slide history buffer with generated frames
  - save side-by-side video: [GT | Prophet] for visual drift / consistency checks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

# Allow running as a standalone script without installing the package:
#   python3 scripts/test_prophet_libero.py ...
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_predict2.auxiliary.guardrail.common.io_utils import save_video
from cosmos_predict2.pipelines.prophet_simulator import ProphetClosedLoopConfig, ProphetClosedLoopSimulator
from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline
from cosmos_predict2.utils.libero_adapter import load_libero_hdf5_trajectory
from imaginaire.lazy_config import LazyCall as L
from imaginaire.utils import log, misc


def _to_u8_video(frames_B_3_T_H_W: torch.Tensor) -> np.ndarray:
    """
    Convert (B,3,T,H,W) in [-1,1] -> (T,H,W,3) uint8.
    """
    x = frames_B_3_T_H_W.detach().float().clamp(-1, 1)
    x = (x[0].permute(1, 2, 3, 0) * 0.5 + 0.5) * 255.0  # (T,H,W,3)
    return x.round().clamp(0, 255).to(torch.uint8).cpu().numpy()


def _resize_hw_uint8(img: np.ndarray, *, height: int, width: int) -> np.ndarray:
    import cv2

    if img.shape[0] == height and img.shape[1] == width:
        return img
    return cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)


def _build_prophet_pipeline(
    *,
    dit_path: str,
    vae_checkpoint_path: str,
    rollout_chunk: int,
    history_length: int,
    d_model: int,
    device: str,
    torch_dtype: torch.dtype,
) -> Video2WorldActionConditionedPipeline:
    """
    Build a Prophet-capable Video2World pipeline config in-code.

    Notes:
    - We use Wan2.1 tokenizer (temporal compression 4x), and set state_t so that
      pixel length == rollout_chunk + 1 (first frame conditioned, remaining are generated).
    """
    from cosmos_predict2.conditioner import ActionConditioner, BooleanFlag, ReMapkey, TextAttr
    from cosmos_predict2.configs.base.config_text2image import CosmosGuardrailConfig
    from cosmos_predict2.configs.base.config_video2world import ConditioningStrategy, CosmosReason1Config, Video2WorldPipelineConfig
    from cosmos_predict2.configs.base.defaults.ema import EMAConfig
    from cosmos_predict2.models.prophet_model import ProphetHistoryBufferConfig, ProphetMinimalV1LVGDiT
    from cosmos_predict2.tokenizers.wan21_tokenizer import Wan21VideoTokenizer
    from imaginaire.constants import CHECKPOINTS_DIR, COSMOS_REASON1_MODEL_DIR

    # Generate (rollout_chunk + 1) pixel frames.
    # Wan2.1: pixel = (latent - 1) * 4 + 1 => latent = (pixel - 1)/4 + 1
    pixel_T = rollout_chunk + 1
    state_t = 1 + (pixel_T - 1 + 3) // 4  # ceil((pixel_T-1)/4) + 1

    cfg = Video2WorldPipelineConfig(
        adjust_video_noise=True,
        conditioner=L(ActionConditioner)(
            fps=L(ReMapkey)(dropout_rate=0.0, dtype=None, input_key="fps", output_key="fps"),
            padding_mask=L(ReMapkey)(dropout_rate=0.0, dtype=None, input_key="padding_mask", output_key="padding_mask"),
            text=L(TextAttr)(dropout_rate=0.0, input_key=["t5_text_embeddings"]),
            use_video_condition=L(BooleanFlag)(dropout_rate=0.0, input_key="fps", output_key="use_video_condition"),
            action=L(ReMapkey)(input_key="action", output_key="action", dropout_rate=0.0, dtype=None),
        ),
        conditioning_strategy=str(ConditioningStrategy.FRAME_REPLACE),
        min_num_conditional_frames=1,
        max_num_conditional_frames=1,
        sigma_conditional=0.0001,
        net=L(ProphetMinimalV1LVGDiT)(
            max_img_h=240,
            max_img_w=240,
            max_frames=128,
            in_channels=16,
            out_channels=16,
            patch_spatial=2,
            patch_temporal=1,
            concat_padding_mask=True,
            model_channels=d_model,
            num_blocks=28,
            num_heads=16,
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
            action_horizon=rollout_chunk,
            d_model=d_model,
            history=ProphetHistoryBufferConfig(enabled=True, history_size=history_length),
        ),
        tokenizer=L(Wan21VideoTokenizer)(
            vae_checkpoint_path=vae_checkpoint_path,
            chunk_duration=81,
            device=device,
            dtype=torch_dtype,
            is_amp=False,
        ),
        prompt_refiner_config=CosmosReason1Config(
            checkpoint_dir=COSMOS_REASON1_MODEL_DIR,
            offload_model_to_cpu=True,
            enabled=False,
        ),
        guardrail_config=CosmosGuardrailConfig(
            checkpoint_dir=CHECKPOINTS_DIR,
            offload_model_to_cpu=True,
            enabled=False,
        ),
        precision="bfloat16",
        rectified_flow_t_scaling_factor=1.0,
        rectified_flow_loss_weight_uniform=True,
        resize_online=False,
        resolution="480",
        ema=L(EMAConfig)(enabled=False),
        sigma_data=1.0,
        state_ch=16,
        state_t=state_t,
    )

    pipe = Video2WorldActionConditionedPipeline.from_config(
        config=cfg,
        dit_path=dit_path,
        use_text_encoder=False,
        device=device,
        torch_dtype=torch_dtype,
        load_ema_to_reg=False,
        load_prompt_refiner=False,
    )
    return pipe


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Closed-loop Prophet validation on LIBERO")
    p.add_argument("--libero_hdf5", type=str, required=True, help="Path to a LIBERO HDF5 demo file")
    p.add_argument("--demo_index", type=int, default=0)
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--dit_path", type=str, required=True, help="Path to Prophet DiT checkpoint")
    p.add_argument("--vae_path", type=str, default="", help="Path to Wan2.1 VAE checkpoint (defaults to prophet_config.yaml)")
    p.add_argument("--out_mp4", type=str, default="output/prophet_libero_side_by_side.mp4")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--guidance", type=float, default=7.0)
    p.add_argument("--steps", type=int, default=35)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--history_length", type=int, default=60)
    p.add_argument("--rollout_chunk", type=int, default=20)
    p.add_argument("--num_loops", type=int, default=5)
    p.add_argument("--action_mode", type=str, default="servo", choices=["servo", "delta"])
    p.add_argument("--d_model", type=int, default=2048, help="Prophet model_channels (Dm). Must match checkpoint.")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument(
        "--print_pixel_diff",
        action="store_true",
        help="Print per-frame mean |Δpixel| between consecutive predicted frames (debug motion/degeneracy).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve VAE path (default from prophet_config.yaml)
    vae_path = args.vae_path
    if not vae_path:
        import yaml

        with open("prophet_config.yaml") as f:
            vae_path = yaml.safe_load(f).get("vae_checkpoint_path", "")
    if not vae_path:
        raise ValueError("Wan2.1 VAE path is required (pass --vae_path or set prophet_config.yaml:vae_checkpoint_path).")

    # Load one LIBERO trajectory
    traj = load_libero_hdf5_trajectory(args.libero_hdf5, demo_index=args.demo_index, camera=args.camera)
    total_future = args.rollout_chunk * args.num_loops
    if traj.rgb.shape[0] < total_future + 1 or traj.action.shape[0] < total_future:
        raise ValueError(
            f"Trajectory too short: need >= {total_future + 1} RGB frames and >= {total_future} actions, "
            f"got rgb={traj.rgb.shape[0]} action={traj.action.shape[0]}"
        )

    # Normalize resolution to something the model is trained with (commonly 256x256 in LIBERO).
    H0, W0 = traj.rgb.shape[1], traj.rgb.shape[2]
    target_hw = (256, 256)
    if (H0, W0) != target_hw:
        rgb = np.stack([_resize_hw_uint8(f, height=target_hw[0], width=target_hw[1]) for f in traj.rgb], axis=0)
        traj = type(traj)(rgb=rgb, action=traj.action, K=traj.K, E=traj.E)

    device = args.device
    torch_dtype = torch.bfloat16

    misc.set_random_seed(seed=args.seed, by_rank=True)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Build Prophet pipeline (loads Wan2.1 tokenizer + Prophet DiT weights).
    pipe = _build_prophet_pipeline(
        dit_path=args.dit_path,
        vae_checkpoint_path=vae_path,
        rollout_chunk=args.rollout_chunk,
        history_length=args.history_length,
        d_model=args.d_model,
        device=device,
        torch_dtype=torch_dtype,
    )

    # Initialize closed-loop history buffer from x0.
    x0_u8 = traj.rgb[0]
    x0 = torch.from_numpy(x0_u8).permute(2, 0, 1).float() / 127.5 - 1.0  # (3,H,W)
    closed_loop = ProphetClosedLoopSimulator(
        ProphetClosedLoopConfig(vae_checkpoint_path=vae_path, history_length=args.history_length, rollout_chunk=args.rollout_chunk),
        device=device,
        dtype=torch_dtype,
    )
    history_latents = closed_loop.init(x0.unsqueeze(0).to(device=device))  # (B,16,T_h,*,*)

    # Closed-loop rollout.
    # NOTE: LIBERO data has identity K/E matrices, so we DISABLE action_frame_latents
    # by NOT passing K/E. This matches the training configuration (ProphRL paper).
    # K = torch.from_numpy(traj.K).to(device=device, dtype=torch.float32)
    # E = torch.from_numpy(traj.E).to(device=device, dtype=torch.float32)

    first_frame = x0_u8
    generated_chunks: list[torch.Tensor] = []
    for k in range(args.num_loops):
        a_chunk = traj.action[k * args.rollout_chunk : (k + 1) * args.rollout_chunk]
        video = pipe(
            first_frame=first_frame,
            actions=a_chunk,
            prompt="",
            negative_prompt="",
            num_conditional_frames=1,
            guidance=args.guidance,
            num_sampling_step=args.steps,
            seed=args.seed + k,
            K=None,  # DISABLED: LIBERO has invalid camera params
            E=None,  # DISABLED: LIBERO has invalid camera params
            history_latents=history_latents,
            action_mode=args.action_mode,
        )
        if video is None:
            raise RuntimeError("Generation returned None (guardrail or model failure).")

        # Expect (B,3,rollout_chunk+1,H,W) where frame 0 is conditioned.
        if video.shape[2] < args.rollout_chunk + 1:
            raise RuntimeError(f"Unexpected generated length: got T={video.shape[2]}, expected >= {args.rollout_chunk + 1}")
        pred = video[:, :, 1 : 1 + args.rollout_chunk, :, :].contiguous()
        generated_chunks.append(pred)

        # Slide history buffer with the new generated chunk.
        history_latents = closed_loop.update_buffer(pred.to(device=device, dtype=torch_dtype))

        # Next initial frame is the last generated frame (uint8).
        last = pred[0, :, -1].detach().float().clamp(-1, 1)
        last_u8 = ((last.permute(1, 2, 0) * 0.5 + 0.5) * 255.0).round().clamp(0, 255).to(torch.uint8).cpu().numpy()
        first_frame = last_u8

    pred_all = torch.cat(generated_chunks, dim=2)  # (B,3,100,H,W)
    gt_all = torch.from_numpy(traj.rgb[1 : 1 + total_future]).permute(0, 3, 1, 2)  # (T,3,H,W)
    gt_all = gt_all.permute(1, 0, 2, 3).unsqueeze(0).float() / 127.5 - 1.0  # (B,3,T,H,W)

    gt_u8 = _to_u8_video(gt_all)
    pred_u8 = _to_u8_video(pred_all)

    if args.print_pixel_diff:
        # mean absolute pixel difference between consecutive predicted frames
        diffs = np.mean(np.abs(pred_u8[1:].astype(np.int16) - pred_u8[:-1].astype(np.int16)), axis=(1, 2, 3))
        head = diffs[:20]
        print("=== predicted frame-to-frame pixel_difference (mean |Δ|) ===")
        print("first 20:", ", ".join(f"{v:.3f}" for v in head))
        print(f"min={diffs.min():.3f} max={diffs.max():.3f} mean={diffs.mean():.3f}")

    # Side-by-side concat: (T,H,W,3) where W doubles.
    side = np.concatenate([gt_u8, pred_u8], axis=2)
    os.makedirs(os.path.dirname(args.out_mp4) or ".", exist_ok=True)
    save_video(args.out_mp4, side, fps=args.fps)
    log.success(f"Saved side-by-side video to: {args.out_mp4}")


if __name__ == "__main__":
    main()

