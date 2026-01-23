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
Prophet fine-tuning on LIBERO - ProphRL Paper Aligned Implementation.

Key design decisions aligned with ProphRL paper:
1. History-aware mechanism: 60-frame history buffer -> FramePack -> Memory M in cross-attn
2. LoRA fine-tuning (rank 16) instead of full model unfreezing
3. Standard diffusion noise prediction loss (ε-prediction), NOT MSE reconstruction
4. Scalar action embedding f_sa ADDED to timestep embedding
5. Weight decay 0.1, gradient accumulation for effective batch size 24

Reference: "Reinforcing Action Policies by Prophesying" (ProphRL)
"""

from __future__ import annotations

import argparse
import math
import random
import sys
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Allow running as a standalone script without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_predict2.pipelines.video2world_action import Video2WorldActionConditionedPipeline  # noqa: E402
from cosmos_predict2.utils.libero_adapter import load_libero_hdf5_trajectory  # noqa: E402
from imaginaire.lazy_config import LazyCall as L  # noqa: E402
from imaginaire.auxiliary.text_encoder import CosmosTextEncoderConfig  # noqa: E402
from imaginaire.utils import log, misc  # noqa: E402


# =============================================================================
# LoRA Implementation (ProphRL Section 4.1.2)
# =============================================================================

class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer for efficient fine-tuning.
    
    Implements: W' = W + BA where B ∈ R^{d×r}, A ∈ R^{r×k}
    Only A and B are trainable, original W is frozen.
    """
    
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 16,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA matrices
        self.lora_A = nn.Linear(in_features, rank, bias=False)
        self.lora_B = nn.Linear(rank, out_features, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        # Initialize: A with Kaiming, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)
        
        # Freeze original weights
        self.original_linear.requires_grad_(False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original forward + LoRA delta
        original_out = self.original_linear(x)
        lora_out = self.lora_B(self.lora_A(self.dropout(x))) * self.scaling
        return original_out + lora_out


def apply_lora_to_model(
    model: nn.Module,
    rank: int = 16,
    alpha: float = 16.0,
    target_modules: list[str] | None = None,
) -> list[nn.Parameter]:
    """
    Apply LoRA to specified linear layers in the model.
    
    Args:
        model: The model to apply LoRA to
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to apply LoRA to
                       (e.g., ["q_proj", "k_proj", "v_proj", "output_proj"])
    
    Returns:
        List of trainable LoRA parameters
    """
    if target_modules is None:
        # Default: apply to attention projections
        target_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
    
    lora_params = []
    replaced_count = 0
    
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear):
            # Check if this module should have LoRA applied
            should_apply = any(target in name for target in target_modules)
            if should_apply:
                # Get parent module and attribute name
                parts = name.rsplit(".", 1)
                if len(parts) == 2:
                    parent_name, attr_name = parts
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                    attr_name = name
                
                # Replace with LoRA version
                lora_linear = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, attr_name, lora_linear)
                
                # Collect trainable params
                lora_params.extend([lora_linear.lora_A.weight, lora_linear.lora_B.weight])
                replaced_count += 1
    
    log.info(f"[LoRA] Applied LoRA (rank={rank}) to {replaced_count} linear layers")
    return lora_params


# =============================================================================
# History Buffer for Training (ProphRL Section 3.2.4)
# =============================================================================

class TrainingHistoryBuffer:
    """
    Maintains a rolling history buffer of latent frames during training.
    
    This simulates the closed-loop setting where we have T_h=60 frames of history.
    """
    
    def __init__(self, history_length: int = 60, device: str = "cuda", dtype: torch.dtype = torch.bfloat16):
        self.history_length = history_length
        self.device = device
        self.dtype = dtype
        self._buffer: torch.Tensor | None = None
    
    def reset(self) -> None:
        self._buffer = None
    
    @torch.no_grad()
    def update(self, new_latents: torch.Tensor) -> None:
        """
        Add new latents to the history buffer.
        
        Args:
            new_latents: (B, C, T, H, W) latent frames to add
        """
        if new_latents.ndim != 5:
            raise ValueError(f"Expected (B,C,T,H,W), got {new_latents.shape}")
        
        new_latents = new_latents.detach().to(device=self.device, dtype=self.dtype)
        
        if self._buffer is None:
            self._buffer = new_latents
        else:
            self._buffer = torch.cat([self._buffer, new_latents], dim=2)
        
        # Trim to history_length
        if self._buffer.shape[2] > self.history_length:
            self._buffer = self._buffer[:, :, -self.history_length:].contiguous()
    
    def get(self) -> torch.Tensor | None:
        """Get current history buffer, or None if empty."""
        return self._buffer
    
    def initialize_from_latents(self, init_latents: torch.Tensor) -> None:
        """
        Initialize buffer by repeating initial latents to fill history.
        
        Args:
            init_latents: (B, C, T_init, H, W) initial latent frames
        """
        if init_latents.ndim != 5:
            raise ValueError(f"Expected (B,C,T,H,W), got {init_latents.shape}")
        
        B, C, T_init, H, W = init_latents.shape
        
        # Repeat to fill history
        repeats = (self.history_length + T_init - 1) // T_init
        repeated = init_latents.repeat(1, 1, repeats, 1, 1)
        self._buffer = repeated[:, :, -self.history_length:].contiguous().to(
            device=self.device, dtype=self.dtype
        )


# =============================================================================
# Training Script
# =============================================================================

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prophet fine-tuning on LIBERO (ProphRL aligned)")
    p.add_argument("--libero_dir", type=str, default="./datasets/libero_spatial")
    p.add_argument("--dit_path", type=str, required=True, help="Path to prophet_init.ckpt")
    p.add_argument("--vae_path", type=str, required=True, help="Path to Wan2.1_VAE.pth")
    p.add_argument("--out_dir", type=str, default="output/prophet_libero_lora")

    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--demo_index", type=int, default=0)

    p.add_argument("--action_horizon", type=int, default=20)
    p.add_argument("--history_length", type=int, default=60, help="History buffer length T_h (paper: 60)")

    # Training hyperparameters (ProphRL aligned)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--grad_accum", type=int, default=24, help="Gradient accumulation steps (effective batch = batch_size * grad_accum)")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay (paper: 0.1)")
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    
    # LoRA configuration (ProphRL aligned)
    p.add_argument("--lora_rank", type=int, default=16, help="LoRA rank (paper: 16)")
    p.add_argument("--lora_alpha", type=float, default=16.0, help="LoRA alpha")
    p.add_argument("--use_lora", action="store_true", default=True, help="Use LoRA instead of full fine-tuning")
    p.add_argument("--no_lora", action="store_true", help="Disable LoRA (full fine-tuning)")

    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16"])

    # Model architecture (2B defaults)
    p.add_argument("--d_model", type=int, default=2048)
    p.add_argument("--num_blocks", type=int, default=28)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--max_img_h", type=int, default=240)
    p.add_argument("--max_img_w", type=int, default=240)
    p.add_argument("--max_frames", type=int, default=128)
    p.add_argument("--patch_spatial", type=int, default=2)
    p.add_argument("--patch_temporal", type=int, default=1)
    
    args = p.parse_args()
    if args.no_lora:
        args.use_lora = False
    return args


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
    """Build Prophet pipeline with history-aware mechanism enabled."""
    from cosmos_predict2.conditioner import ActionConditioner, BooleanFlag, ReMapkey, TextAttr
    from cosmos_predict2.configs.base.config_text2image import CosmosGuardrailConfig
    from cosmos_predict2.configs.base.config_video2world import ConditioningStrategy, CosmosReason1Config, Video2WorldPipelineConfig
    from cosmos_predict2.configs.base.defaults.ema import EMAConfig
    from cosmos_predict2.models.prophet_model import ProphetHistoryBufferConfig, ProphetMinimalV1LVGDiT
    from cosmos_predict2.tokenizers.wan21_tokenizer import Wan21VideoTokenizer
    from imaginaire.constants import CHECKPOINTS_DIR, COSMOS_REASON1_MODEL_DIR

    pixel_T = action_horizon + 1
    state_t = 1 + (pixel_T - 1 + 3) // 4

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
            # CRITICAL: Enable history buffer (ProphRL Section 3.2.4)
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


def _sample_trajectory_chunk(
    *,
    hdf5_paths: list[Path],
    demo_index: int,
    camera: str,
    history_length: int,
    action_horizon: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sample a trajectory chunk with history context.
    
    Returns:
        history_rgb: (B, 3, T_h, H, W) uint8 - history frames for context
        future_rgb: (B, 3, T_f+1, H, W) uint8 - first frame + future frames
        action: (B, T_f, 7) float32 - actions for future prediction
        padding_mask: (B, 1, H, W) float32
    """
    path = random.choice(hdf5_paths)
    traj = load_libero_hdf5_trajectory(str(path), demo_index=demo_index, camera=camera)
    
    # Need: history_length + 1 (conditional) + action_horizon (future) frames
    T_total_needed = history_length + 1 + action_horizon
    T_actions_needed = action_horizon
    
    if traj.rgb.shape[0] < T_total_needed:
        # If trajectory is too short, we'll pad by repeating early frames
        T_total_needed = min(T_total_needed, traj.rgb.shape[0])
    
    if traj.action.shape[0] < T_actions_needed:
        raise ValueError(f"Trajectory too short for actions: need {T_actions_needed}, got {traj.action.shape[0]}")
    
    # Pick random start point
    max_start = max(0, traj.rgb.shape[0] - T_total_needed)
    start = random.randint(0, max_start) if max_start > 0 else 0
    
    # Extract history, conditional frame, and future frames
    end = start + T_total_needed
    all_rgb = traj.rgb[start:end]  # (T_total, H, W, 3)
    
    # Split into history and future
    if all_rgb.shape[0] >= history_length + 1 + action_horizon:
        history_rgb = all_rgb[:history_length]  # (T_h, H, W, 3)
        future_rgb = all_rgb[history_length:history_length + 1 + action_horizon]  # (T_f+1, H, W, 3)
    else:
        # Handle short trajectories by using what we have
        split_point = max(1, all_rgb.shape[0] - action_horizon - 1)
        history_rgb = all_rgb[:split_point]
        future_rgb = all_rgb[split_point:]
        # Pad history if needed
        if history_rgb.shape[0] < history_length:
            pad_count = history_length - history_rgb.shape[0]
            history_rgb = np.concatenate([history_rgb[:1].repeat(pad_count, axis=0), history_rgb], axis=0)
    
    # Get corresponding actions (for the future frames)
    action_start = start + history_length if start + history_length < traj.action.shape[0] else 0
    action_end = min(action_start + action_horizon, traj.action.shape[0])
    action = traj.action[action_start:action_end]
    
    # Pad actions if needed
    if action.shape[0] < action_horizon:
        pad_count = action_horizon - action.shape[0]
        action = np.concatenate([action, action[-1:].repeat(pad_count, axis=0)], axis=0)
    
    # Convert to tensors (B=1)
    history_tensor = torch.from_numpy(history_rgb).permute(3, 0, 1, 2).unsqueeze(0).to(device=device)  # (1,3,T_h,H,W)
    future_tensor = torch.from_numpy(future_rgb).permute(3, 0, 1, 2).unsqueeze(0).to(device=device)  # (1,3,T_f+1,H,W)
    action_tensor = torch.from_numpy(action).unsqueeze(0).to(device=device, dtype=torch.float32)  # (1,T_f,7)
    
    _, _, _, H, W = future_tensor.shape
    padding_mask = torch.zeros((1, 1, H, W), device=device, dtype=torch.float32)
    
    return history_tensor, future_tensor, action_tensor, padding_mask


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

    # Build pipeline
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

    # =========================================================================
    # FREEZE/UNFREEZE STRATEGY (ProphRL aligned)
    # =========================================================================
    
    # Always freeze VAE/tokenizer
    pipe.tokenizer.requires_grad_(False)
    pipe.tokenizer.eval()

    # Freeze entire DiT first
    pipe.dit.requires_grad_(False)
    pipe.dit.train()
    
    trainable_params = []
    
    if args.use_lora:
        # Apply LoRA to attention projections (ProphRL Section 4.1.2)
        log.info(f"[ProphRL] Applying LoRA (rank={args.lora_rank}) to DiT attention layers")
        lora_params = apply_lora_to_model(
            pipe.dit,
            rank=args.lora_rank,
            alpha=args.lora_alpha,
            target_modules=["q_proj", "k_proj", "v_proj", "output_proj"],
        )
        trainable_params.extend(lora_params)
    else:
        # Full fine-tuning fallback
        log.info("[WARNING] Full fine-tuning mode (not recommended)")
        pipe.dit.requires_grad_(True)
        trainable_params.extend(list(pipe.dit.parameters()))
    
    # ALWAYS unfreeze Scalar Action MLP (ProphRL requirement)
    pipe.dit.scalar_action.requires_grad_(True)
    trainable_params.extend(list(pipe.dit.scalar_action.parameters()))
    
    # ALWAYS unfreeze FramePack (history encoder)
    pipe.dit.framepack.requires_grad_(True)
    trainable_params.extend(list(pipe.dit.framepack.parameters()))
    
    # Unfreeze x_embedder and final_layer for latent space adaptation
    pipe.dit.x_embedder.requires_grad_(True)
    pipe.dit.final_layer.requires_grad_(True)
    trainable_params.extend(list(pipe.dit.x_embedder.parameters()))
    trainable_params.extend(list(pipe.dit.final_layer.parameters()))

    # Count trainable params
    trainable_params = list(set(trainable_params))  # Remove duplicates
    total_params = sum(p.numel() for p in pipe.dit.parameters())
    trainable_count = sum(p.numel() for p in trainable_params if p.requires_grad)
    log.info(f"Trainable params: {trainable_count:,} / {total_params:,} ({100*trainable_count/total_params:.2f}%)")

    # Optimizer (ProphRL: weight_decay=0.1)
    optim = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        betas=(0.9, 0.95),
        weight_decay=args.weight_decay,
    )
    
    # Learning rate scheduler with warmup
    warmup_steps = min(1000, args.steps // 10)
    
    def get_lr(step: int) -> float:
        if step < warmup_steps:
            return args.lr * step / warmup_steps
        # Cosine decay after warmup
        progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return args.lr * 0.5 * (1 + math.cos(math.pi * progress))

    pipe.device = torch.device(device)
    pipe.dit.to(device=device, dtype=torch_dtype)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    effective_batch = args.batch_size * args.grad_accum
    
    log.info("=" * 70)
    log.info("ProphRL LIBERO Training Configuration (Paper Aligned):")
    log.info(f"  - LoRA fine-tuning: {args.use_lora} (rank={args.lora_rank})")
    log.info(f"  - History buffer: ENABLED (T_h={args.history_length})")
    log.info(f"  - Loss: Diffusion noise prediction (ε-prediction)")
    log.info(f"  - Action Frame: DISABLED (LIBERO lacks camera params)")
    log.info(f"  - Scalar Action: f_sa ADDED to timestep embedding")
    log.info(f"  - Weight decay: {args.weight_decay}")
    log.info(f"  - Effective batch size: {effective_batch} (bs={args.batch_size} × accum={args.grad_accum})")
    log.info(f"  - Learning rate: {args.lr}")
    log.info(f"  - Steps: {args.steps}")
    log.info("=" * 70)

    # Initialize history buffer
    history_buffer = TrainingHistoryBuffer(
        history_length=args.history_length,
        device=device,
        dtype=torch_dtype,
    )

    accum_loss = 0.0
    accum_count = 0

    for step in range(1, args.steps + 1):
        # Update learning rate
        current_lr = get_lr(step)
        for param_group in optim.param_groups:
            param_group['lr'] = current_lr
        
        # Sample trajectory chunk with history
        history_rgb, future_rgb, action_raw, padding_mask = _sample_trajectory_chunk(
            hdf5_paths=hdf5_paths,
            demo_index=args.demo_index,
            camera=args.camera,
            history_length=args.history_length,
            action_horizon=args.action_horizon,
            device=device,
        )
        
        B = future_rgb.shape[0]
        
        # Debug: print action statistics on first step
        if step == 1:
            log.info(f"[DEBUG] RAW action stats: min={action_raw.min():.4f}, max={action_raw.max():.4f}")
            log.info(f"[DEBUG] History shape: {history_rgb.shape}, Future shape: {future_rgb.shape}")

        # Encode history to latents for FramePack
        with torch.no_grad():
            # Normalize history RGB to [-1, 1]
            history_normalized = history_rgb.float() / 127.5 - 1.0
            history_latents = pipe.tokenizer.encode(history_normalized.to(dtype=torch_dtype))
        
        # Build data batch
        data_batch = {
            "dataset_name": "video_data",
            "video": future_rgb,  # (B, 3, T_f+1, H, W) uint8
            "t5_text_embeddings": torch.zeros(
                B, 1, CosmosTextEncoderConfig.EMBED_DIM,
                device=device, dtype=torch.bfloat16,
            ),
            "fps": torch.ones((B,), device=device, dtype=torch.int32) * 10,
            "padding_mask": padding_mask,
            "num_conditional_frames": 1,
            "action": action_raw.to(dtype=torch_dtype),
            "action_frame_latents": None,  # DISABLED for LIBERO
            "history_latents": history_latents,  # ENABLED: FramePack memory
        }

        # Get x0 latents and condition
        _, x0_lat, condition = pipe.get_data_and_condition(data_batch)

        # =================================================================
        # DIFFUSION NOISE PREDICTION LOSS (ProphRL aligned)
        # L_diff = ||ε - ε_θ(x_t, t, c)||²
        # =================================================================
        
        # Sample noise and timestep
        eps = torch.randn_like(x0_lat, device=device, dtype=torch.float32)
        
        # Sample sigma from scheduler (EDM-style)
        sigma_B = pipe.scheduler.sample_sigma(B).to(device=device, dtype=torch.float32)
        sigma_B_T = sigma_B.view(B, 1)
        
        # Add noise: x_t = x_0 + σ * ε
        xt = x0_lat.float() + eps * sigma_B_T.view(B, 1, 1, 1, 1)

        # Forward pass through denoiser
        # The denoiser returns x0 prediction, we need to convert to noise prediction
        denoise_result = pipe.denoise(xt, sigma_B_T, condition)
        x0_pred = denoise_result.x0
        
        # Convert x0 prediction to noise prediction: ε_pred = (x_t - x0_pred) / σ
        eps_pred = (xt - x0_pred.float()) / sigma_B_T.view(B, 1, 1, 1, 1).clamp(min=1e-6)
        
        # Noise prediction loss on FUTURE frames only (exclude conditional frame)
        if eps_pred.shape[2] < 2:
            raise RuntimeError(f"Need at least 2 latent frames, got {eps_pred.shape[2]}")
        
        # L_diff = ||ε - ε_θ||² (only on future frames)
        loss = F.mse_loss(eps_pred[:, :, 1:], eps[:, :, 1:])
        
        # Scale loss for gradient accumulation
        loss = loss / args.grad_accum
        loss.backward()
        
        accum_loss += loss.item() * args.grad_accum
        accum_count += 1

        # Gradient accumulation step
        if step % args.grad_accum == 0:
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
            optim.step()
            optim.zero_grad(set_to_none=True)
            
            if (step // args.grad_accum) % 10 == 0 or step == args.grad_accum:
                avg_loss = accum_loss / accum_count if accum_count > 0 else 0.0
                log.info(
                    f"[step {step:05d}/{args.steps}] "
                    f"loss={avg_loss:.6f} "
                    f"sigma={sigma_B.mean().item():.4f} "
                    f"lr={current_lr:.2e}"
                )
            accum_loss = 0.0
            accum_count = 0

        if step % args.save_every == 0 or step == args.steps:
            ckpt_path = out_dir / f"prophet_libero_step_{step:05d}.ckpt"
            _save_checkpoint(pipe, ckpt_path)
            log.success(f"Saved checkpoint: {ckpt_path}")


if __name__ == "__main__":
    main()
