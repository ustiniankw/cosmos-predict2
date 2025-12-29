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
Quick adaptation for Prophet on LIBERO (minimal fine-tuning).

Goal:
  Adjust the DiT latent distribution to better align with Wan2.1 VAE latents and reduce "rainbow noise".

Training objective (teacher-forced, diffusion-style x0 prediction):
  - sample a short chunk from a LIBERO demonstration: T = 1 + action_horizon pixel frames
  - encode GT video to Wan2.1 latents x0
  - add noise: xt = x0 + sigma * eps
  - run Prophet DiT denoiser to get x0_pred
  - compute MSE on FUTURE latents only (exclude the first conditional latent frame):
        loss = MSE(x0_pred[:, :, 1:], x0[:, :, 1:])

Frozen:
  - Wan2.1 VAE/tokenizer is frozen
  - DiT is frozen except x_embedder + final_layer (fast, stable)

Saves:
  - every `--save_every` steps, write a checkpoint compatible with pipeline loading:
      keys prefixed with "net."
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Allow running as a standalone script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline  # noqa: E402
from cosmos_predict2.utils.action_renderer import render_action_rgb  # noqa: E402
from cosmos_predict2.utils.libero_adapter import load_libero_hdf5_trajectory  # noqa: E402
from imaginaire.lazy_config import LazyCall as L  # noqa: E402
from imaginaire.utils import log, misc  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Quick adaptation (minimal fine-tune) for Prophet on LIBERO")
    p.add_argument("--libero_dir", type=str, default="./datasets/libero_spatial", help="Directory containing LIBERO *.hdf5")
    p.add_argument("--dit_path", type=str, required=True, help="Path to prophet_init.ckpt (or another init ckpt)")
    p.add_argument("--vae_path", type=str, required=True, help="Path to Wan2.1_VAE.pth")
    p.add_argument("--out_dir", type=str, default="output/prophet_adapt", help="Where to save adapted checkpoints")

    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--demo_index", type=int, default=0, help="Default demo index (used if file has a single demo)")

    p.add_argument("--action_horizon", type=int, default=20, help="Number of actions / future pixel frames")
    p.add_argument("--history_length", type=int, default=60, help="Model history buffer size (not used in this script)")
    p.add_argument("--action_mode", type=str, default="servo", choices=["servo", "delta"])

    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--save_every", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])

    # Model shape knobs (2B defaults)
    p.add_argument("--d_model", type=int, default=2048)
    p.add_argument("--num_blocks", type=int, default=28)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--max_img_h", type=int, default=240)
    p.add_argument("--max_img_w", type=int, default=240)
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--patch_spatial", type=int, default=2)
    p.add_argument("--patch_temporal", type=int, default=1)
    return p.parse_args()


def _dtype_from_str(s: str) -> torch.dtype:
    return {"float16": torch.float16, "bfloat16": torch.bfloat16}[s]


def _build_prophet_pipe(
    *,
    dit_path: str,
    vae_path: str,
    action_horizon: int,
    history_length: int,
    d_model: int,
    num_blocks: int,
    num_heads: int,
    max_img_h: int,
    max_img_w: int,
    max_frames: int,
    patch_spatial: int,
    patch_temporal: int,
    device: str,
    torch_dtype: torch.dtype,
) -> Video2WorldActionConditionedPipeline:
    """
    Construct a Prophet-capable action-conditioned pipeline (no text encoder).
    """
    from cosmos_predict2.conditioner import ActionConditioner, BooleanFlag, ReMapkey, TextAttr
    from cosmos_predict2.configs.base.config_text2image import CosmosGuardrailConfig
    from cosmos_predict2.configs.base.config_video2world import ConditioningStrategy, CosmosReason1Config, Video2WorldPipelineConfig
    from cosmos_predict2.configs.base.defaults.ema import EMAConfig
    from cosmos_predict2.models.prophet_model import ProphetHistoryBufferConfig, ProphetMinimalV1LVGDiT
    from cosmos_predict2.tokenizers.wan21_tokenizer import Wan21VideoTokenizer
    from imaginaire.constants import CHECKPOINTS_DIR, COSMOS_REASON1_MODEL_DIR

    # Ensure tokenizer pixel length covers 1 + action_horizon frames.
    pixel_T = action_horizon + 1
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
            max_img_h=max_img_h,
            max_img_w=max_img_w,
            max_frames=max_frames,
            in_channels=16,
            out_channels=16,
            patch_spatial=patch_spatial,
            patch_temporal=patch_temporal,
            concat_padding_mask=True,
            model_channels=d_model,
            num_blocks=num_blocks,
            num_heads=num_heads,
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
            action_horizon=action_horizon,
            d_model=d_model,
            history=ProphetHistoryBufferConfig(enabled=True, history_size=history_length),
        ),
        tokenizer=L(Wan21VideoTokenizer)(
            vae_checkpoint_path=vae_path,
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
        precision="bfloat16" if torch_dtype == torch.bfloat16 else "float16",
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


def _sample_batch(
    *,
    hdf5_paths: list[Path],
    batch_size: int,
    demo_index: int,
    camera: str,
    action_horizon: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      video_u8_B_3_T_H_W
      action_B_Ta_7  (Ta=action_horizon)
      K_B_3_3
      E_B_4_4
      padding_mask_B_1_H_W
    """
    videos = []
    actions = []
    Ks = []
    Es = []

    for _ in range(batch_size):
        path = random.choice(hdf5_paths)
        traj = load_libero_hdf5_trajectory(str(path), demo_index=demo_index, camera=camera)

        T_req = action_horizon + 1
        if traj.rgb.shape[0] < T_req or traj.action.shape[0] < action_horizon:
            raise ValueError(f"Trajectory too short in {path}: rgb_T={traj.rgb.shape[0]} action_T={traj.action.shape[0]}")

        # pick a random contiguous chunk
        max_start = traj.rgb.shape[0] - T_req
        start = 0 if max_start <= 0 else random.randint(0, max_start)

        rgb = traj.rgb[start : start + T_req]  # (T,H,W,3) uint8
        act = traj.action[start : start + action_horizon]  # (Ta,7) float32

        # (T,H,W,3) -> (3,T,H,W)
        v = torch.from_numpy(rgb).to(torch.uint8).permute(3, 0, 1, 2).contiguous()
        videos.append(v)
        actions.append(torch.from_numpy(act).to(torch.float32))
        Ks.append(torch.from_numpy(traj.K).to(torch.float32))
        Es.append(torch.from_numpy(traj.E).to(torch.float32))

    video_u8 = torch.stack(videos, dim=0).to(device=device)  # (B,3,T,H,W) uint8
    action = torch.stack(actions, dim=0).to(device=device)  # (B,Ta,7) float32
    K = torch.stack(Ks, dim=0).to(device=device)  # (B,3,3)
    E = torch.stack(Es, dim=0).to(device=device)  # (B,4,4)
    _, _, _, H, W = video_u8.shape
    padding_mask = torch.zeros((batch_size, 1, H, W), device=device, dtype=torch.float32)
    return video_u8, action, K, E, padding_mask


def _save_checkpoint(pipe: Video2WorldActionConditionedPipeline, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sd = {f"net.{k}": v.detach().cpu() for k, v in pipe.dit.state_dict().items()}
    torch.save(sd, str(out_path))


def main() -> None:
    args = parse_args()
    misc.set_random_seed(seed=args.seed, by_rank=True)
    random.seed(args.seed)
    np.random.seed(args.seed)

    device = args.device
    torch_dtype = _dtype_from_str(args.dtype)

    libero_dir = Path(args.libero_dir)
    hdf5_paths = sorted(libero_dir.rglob("*.hdf5"))
    if not hdf5_paths:
        raise FileNotFoundError(f"No .hdf5 found under: {libero_dir}")
    log.info(f"Found {len(hdf5_paths)} LIBERO HDF5 files under: {libero_dir}")

    # Build pipeline (loads init weights).
    pipe = _build_prophet_pipe(
        dit_path=args.dit_path,
        vae_path=args.vae_path,
        action_horizon=args.action_horizon,
        history_length=args.history_length,
        d_model=args.d_model,
        num_blocks=args.num_blocks,
        num_heads=args.num_heads,
        max_img_h=args.max_img_h,
        max_img_w=args.max_img_w,
        max_frames=args.max_frames,
        patch_spatial=args.patch_spatial,
        patch_temporal=args.patch_temporal,
        device=device,
        torch_dtype=torch_dtype,
    )

    # Freeze VAE/tokenizer.
    pipe.tokenizer.requires_grad_(False)
    pipe.tokenizer.eval()

    # Freeze all DiT params, then unfreeze x_embedder + final_layer.
    pipe.dit.requires_grad_(False)
    pipe.dit.train()
    pipe.dit.x_embedder.requires_grad_(True)
    pipe.dit.final_layer.requires_grad_(True)

    trainable = [p for p in pipe.dit.parameters() if p.requires_grad]
    log.info(f"Trainable params: {sum(p.numel() for p in trainable):,}")

    optim = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.999), weight_decay=0.0)

    # Main training loop.
    pipe.device = torch.device(device)
    pipe.dit.to(device=device, dtype=torch_dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for step in range(1, args.steps + 1):
        video_u8, action_f32, K, E, padding_mask = _sample_batch(
            hdf5_paths=hdf5_paths,
            batch_size=args.batch_size,
            demo_index=args.demo_index,
            camera=args.camera,
            action_horizon=args.action_horizon,
            device=device,
        )
        B, _, T_pix, H, W = video_u8.shape
        assert T_pix == args.action_horizon + 1

        # Build action_frame_latents (Wan2.1 latents of rendered action frames).
        # render_action_rgb expects actions (B,Ta,7) float32 and K/E tensors.
        action_rgb = render_action_rgb(
            action_f32,
            K,
            E,
            height=H,
            width=W,
            action_mode=args.action_mode,
        )  # (B,Ta,H,W,3) in [0,1]
        action_frames = action_rgb.permute(0, 4, 1, 2, 3).contiguous() * 2.0 - 1.0  # (B,3,Ta,H,W) in [-1,1]
        with torch.no_grad():
            action_frame_latents = pipe.tokenizer.encode(action_frames.to(dtype=torch_dtype))

        # Build data_batch compatible with ActionConditioner + Video2WorldPipeline.get_data_and_condition().
        data_batch = {
            "dataset_name": "video_data",
            "video": video_u8,
            "t5_text_embeddings": torch.zeros(
                B,
                256,  # CosmosTextEncoderConfig.NUM_TOKENS
                4096,  # CosmosTextEncoderConfig.EMBED_DIM
                device=device,
                dtype=torch.bfloat16,
            ),
            "fps": torch.ones((B,), device=device, dtype=torch.int32) * 10,
            "padding_mask": padding_mask,
            "num_conditional_frames": 1,
            "action": action_f32.to(dtype=torch_dtype),
            "action_frame_latents": action_frame_latents,
            "K": K,
            "E": E,
        }

        # Get x0 latents and condition.
        _, x0_lat, condition = pipe.get_data_and_condition(data_batch)  # x0_lat: (B,16,T_lat,H/8,W/8)

        # Diffusion-style training step (x0 prediction).
        eps = torch.randn_like(x0_lat, device=device, dtype=torch.float32)
        sigma_B = pipe.scheduler.sample_sigma(B).to(device=device, dtype=torch.float32)
        sigma_B_T = sigma_B.view(B, 1)  # (B,1) all frames share sigma
        xt = x0_lat.float() + eps * sigma_B_T.view(B, 1, 1, 1, 1)

        optim.zero_grad(set_to_none=True)
        pred = pipe.denoise(xt, sigma_B_T, condition).x0  # (B,16,T_lat,H/8,W/8)

        # MSE on future latent frames only (exclude first latent frame).
        if pred.shape[2] < 2:
            raise RuntimeError(f"Need at least 2 latent frames to compute future loss, got T_lat={pred.shape[2]}")
        loss = F.mse_loss(pred[:, :, 1:, :, :], x0_lat[:, :, 1:, :, :])
        loss.backward()
        optim.step()

        if step % 10 == 0 or step == 1:
            log.info(f"[step {step:04d}/{args.steps}] loss={loss.item():.6f}  sigma={sigma_B.mean().item():.4f}")

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = out_dir / f"prophet_adapted_step_{step:04d}.ckpt"
            _save_checkpoint(pipe, ckpt_path)
            log.success(f"Saved adapted checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()

