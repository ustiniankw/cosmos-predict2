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
Action rendering infrastructure for ProphRL (Section 3.2.1 / 3.2.2).

This module is intentionally model-agnostic so it can be reused in:
- Offline preprocessing (pre-render action frames, pre-encode with Wan2.1 VAE)
- Online inference pipelines (render on the fly, encode, then feed latents to Prophet)

Key responsibilities:
- Action standardization to a fixed max number of end-effectors (N, e.g. 2)
- Optional pose integration (cumsum) when absolute pose is not provided
- Pure geometric rendering (no learnable basis):
  input 7D action + camera K/E -> black RGB canvas with disks + axis lines
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class ActionRenderConstants:
    axis_length: float = 0.15
    r_ref: float = 40.0
    z_ref: float = 1.0
    r_min: float = 8.0
    r_max: float = 140.0
    line_width: float = 2.0


def standardize_actions(
    actions: torch.Tensor,
    *,
    max_arms: int = 2,
) -> torch.Tensor:
    """
    Standardize action tensor to shape (B, T, N, 7), padding missing arms with zeros.

    Expected action semantics per arm:
      c_{t,n} = [Δp_{t,n}(3), Δe_{t,n}(3), g_{t,n}(1)] ∈ R^7

    Supported input shapes:
      - (T, 7)                  -> (1, T, 1, 7) then padded to N=max_arms
      - (B, T, 7)               -> (B, T, 1, 7) then padded
      - (B, T, N, 7)            -> padded/truncated to N=max_arms

    Returns:
      (B, T, max_arms, 7)
    """
    if actions.ndim == 2:
        if actions.shape[-1] != 7:
            raise ValueError(f"Expected (T,7), got {actions.shape}")
        actions = actions.unsqueeze(0).unsqueeze(2)  # (1,T,1,7)
    elif actions.ndim == 3:
        if actions.shape[-1] != 7:
            raise ValueError(f"Expected (B,T,7), got {actions.shape}")
        actions = actions.unsqueeze(2)  # (B,T,1,7)
    elif actions.ndim == 4:
        if actions.shape[-1] != 7:
            raise ValueError(f"Expected (B,T,N,7), got {actions.shape}")
    else:
        raise ValueError(f"Unsupported actions shape: {actions.shape}")

    B, T, N, D = actions.shape
    if N == max_arms:
        return actions
    if N > max_arms:
        return actions[:, :, :max_arms, :].contiguous()

    pad = torch.zeros((B, T, max_arms - N, 7), device=actions.device, dtype=actions.dtype)
    return torch.cat([actions, pad], dim=2).contiguous()


def integrate_delta_actions_to_pose(
    actions_BTN7: torch.Tensor,
    *,
    z_ref: float = 1.0,
    mode: Literal["cumsum"] = "cumsum",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Integrate delta actions into an approximate pose (p, euler) when absolute pose is unavailable.

    Args:
      actions_BTN7: (B,T,N,7)
    Returns:
      p_BTN3:     (B,T,N,3)
      euler_BTN3: (B,T,N,3)
    """
    if actions_BTN7.ndim != 4 or actions_BTN7.shape[-1] != 7:
        raise ValueError(f"Expected actions (B,T,N,7), got {actions_BTN7.shape}")
    if mode != "cumsum":
        raise ValueError(f"Unsupported integration mode: {mode}")

    dxyz = actions_BTN7[..., 0:3]
    deuler = actions_BTN7[..., 3:6]
    # initial pose: p0=(0,0,z_ref), euler0=(0,0,0)
    p0 = torch.tensor([0.0, 0.0, z_ref], device=actions_BTN7.device, dtype=actions_BTN7.dtype)[None, None, None, :]
    p = p0 + torch.cumsum(dxyz, dim=1)
    euler = torch.cumsum(deuler, dim=1)
    return p, euler


def euler_xyz_to_matrix(euler_xyz: torch.Tensor) -> torch.Tensor:
    """
    Convert XYZ Euler angles to rotation matrices.
    Args:
      euler_xyz: (..., 3) in radians
    Returns:
      (..., 3, 3)
    """
    if euler_xyz.shape[-1] != 3:
        raise ValueError(f"Expected (...,3) euler, got {euler_xyz.shape}")
    x, y, z = euler_xyz[..., 0], euler_xyz[..., 1], euler_xyz[..., 2]
    cx, cy, cz = torch.cos(x), torch.cos(y), torch.cos(z)
    sx, sy, sz = torch.sin(x), torch.sin(y), torch.sin(z)

    # R = Rz * Ry * Rx
    r00 = cz * cy
    r01 = cz * sy * sx - sz * cx
    r02 = cz * sy * cx + sz * sx
    r10 = sz * cy
    r11 = sz * sy * sx + cz * cx
    r12 = sz * sy * cx - cz * sx
    r20 = -sy
    r21 = cy * sx
    r22 = cy * cx
    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


def _point_line_distance_2d(
    grid_xy_HW_2: torch.Tensor, p0_BT_2: torch.Tensor, p1_BT_2: torch.Tensor
) -> torch.Tensor:
    # grid: (H,W,2), p0/p1: (B,T,2)
    g = grid_xy_HW_2[None, None, :, :, :]  # (1,1,H,W,2)
    p0 = p0_BT_2[:, :, None, None, :]  # (B,T,1,1,2)
    p1 = p1_BT_2[:, :, None, None, :]  # (B,T,1,1,2)
    v = p1 - p0
    w = g - p0
    vv = (v * v).sum(dim=-1).clamp_min(1e-6)  # (B,T,1,1)
    t = (w * v).sum(dim=-1) / vv  # (B,T,H,W)
    t = t.clamp(0.0, 1.0)[..., None]
    proj = p0 + t * v
    d = g - proj
    return torch.sqrt((d * d).sum(dim=-1) + 1e-6)


def gripper_colormap(gripper_BT_1: torch.Tensor) -> torch.Tensor:
    """
    Fixed colormap: map scalar in [0,1] -> RGB (B,T,1,3).
    """
    g = gripper_BT_1.clamp(0.0, 1.0)
    c0 = torch.tensor([0.0, 0.2, 1.0], device=g.device, dtype=g.dtype)
    c1 = torch.tensor([0.0, 0.9, 1.0], device=g.device, dtype=g.dtype)
    c2 = torch.tensor([1.0, 0.9, 0.0], device=g.device, dtype=g.dtype)
    c3 = torch.tensor([1.0, 0.1, 0.0], device=g.device, dtype=g.dtype)
    t0 = (g * 3.0).clamp(0.0, 3.0)
    w0 = (1.0 - (t0 - 0.0).clamp(0.0, 1.0)).unsqueeze(-1)
    w1 = ((t0 - 0.0).clamp(0.0, 1.0)).unsqueeze(-1)
    w2 = ((t0 - 1.0).clamp(0.0, 1.0)).unsqueeze(-1)
    w3 = ((t0 - 2.0).clamp(0.0, 1.0)).unsqueeze(-1)
    col01 = c0 * w0 + c1 * w1
    col12 = c1 * (1.0 - w2) + c2 * w2
    col23 = c2 * (1.0 - w3) + c3 * w3
    seg0 = (t0 < 1.0).to(dtype=g.dtype).unsqueeze(-1)
    seg1 = ((t0 >= 1.0) & (t0 < 2.0)).to(dtype=g.dtype).unsqueeze(-1)
    seg2 = (t0 >= 2.0).to(dtype=g.dtype).unsqueeze(-1)
    return col01 * seg0 + col12 * seg1 + col23 * seg2


def render_action_rgb(
    actions: torch.Tensor,
    K: torch.Tensor,
    E: torch.Tensor,
    *,
    height: int = 256,
    width: int = 256,
    max_arms: int = 2,
    constants: ActionRenderConstants = ActionRenderConstants(),
    action_mode: Literal["delta", "servo"] = "delta",
) -> torch.Tensor:
    """
    Render action frames as RGB images on black background.

    Args:
      actions: (T,7) or (B,T,7) or (B,T,N,7)
      K: (3,3) or (B,3,3)
      E: (4,4) or (B,4,4)

    Returns:
      rgb_B_T_H_W_3: float32 in [0,1]
    """
    device = actions.device
    dtype = torch.float32

    a_BTN7 = standardize_actions(actions, max_arms=max_arms).to(dtype=dtype)
    B, T, N, _ = a_BTN7.shape

    if K.ndim == 2:
        K = K[None, :, :].expand(B, -1, -1)
    if E.ndim == 2:
        E = E[None, :, :].expand(B, -1, -1)
    if K.shape != (B, 3, 3):
        raise ValueError(f"Expected K shape (B,3,3), got {K.shape}")
    if E.shape != (B, 4, 4):
        raise ValueError(f"Expected E shape (B,4,4), got {E.shape}")

    # Pose source:
    # - delta: integrate (cumsum) to get an approximate pose trajectory for visualization.
    # - servo: actions already encode an absolute servo command (pose target); visualize directly.
    if action_mode == "delta":
        p_BTN3, euler_BTN3 = integrate_delta_actions_to_pose(a_BTN7, z_ref=constants.z_ref)
    elif action_mode == "servo":
        p_BTN3 = a_BTN7[..., 0:3]
        euler_BTN3 = a_BTN7[..., 3:6]
    else:
        raise ValueError(f"Unsupported action_mode: {action_mode}")

    R_BTN33 = euler_xyz_to_matrix(euler_BTN3)  # (B,T,N,3,3)
    grip_BTN1 = a_BTN7[..., 6:7]

    # pixel grid (H,W,2)
    ys = torch.arange(height, dtype=dtype, device=device)
    xs = torch.arange(width, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (H,W,2)

    axis_colors = torch.tensor(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.2, 1.0]], device=device, dtype=dtype
    )

    # output canvas
    rgb = torch.zeros((B, T, height, width, 3), device=device, dtype=dtype)

    # Draw each arm (overlay with max)
    for arm in range(N):
        p_BT3 = p_BTN3[:, :, arm, :]  # (B,T,3)
        R_BT33 = R_BTN33[:, :, arm, :, :]

        # axis endpoints
        axis = torch.eye(3, device=device, dtype=dtype) * constants.axis_length  # (3,3)
        axis_vecs = torch.einsum("btij,kj->btki", R_BT33, axis)  # (B,T,3,3)
        pk_BT33 = p_BT3[:, :, None, :] + axis_vecs  # (B,T,3,3)

        def project(points_B_T_N_3: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            ones = torch.ones((*points_B_T_N_3.shape[:-1], 1), device=device, dtype=dtype)
            ph = torch.cat([points_B_T_N_3, ones], dim=-1)  # (B,T,N,4)
            x_cam = torch.einsum("bij,btnj->btni", E.to(dtype=dtype), ph)
            x = x_cam[..., 0:3]
            z = x[..., 2:3].clamp_min(1e-6)
            u_h = torch.einsum("bij,btnj->btni", K.to(dtype=dtype), x)
            u = u_h[..., 0:2] / z
            return u, z.squeeze(-1)

        u_center, z_center = project(p_BT3[:, :, None, :])
        u_axes, _ = project(pk_BT33)
        u0 = u_center[:, :, 0, :]  # (B,T,2)

        # radius from depth
        r = (constants.r_ref * (constants.z_ref / z_center)).clamp(constants.r_min, constants.r_max)  # (B,T)

        # disk mask
        dxy = grid[None, None, :, :, :] - u0[:, :, None, None, :]
        dist_center = torch.sqrt((dxy * dxy).sum(dim=-1) + 1e-6)  # (B,T,H,W)
        disk = (dist_center <= r[:, :, None, None]).to(dtype=dtype)

        grip_color = gripper_colormap(((grip_BTN1[:, :, arm, :] + 1.0) / 2.0).to(dtype=dtype))  # (B,T,1,3)
        disk_rgb = disk[:, :, :, :, None] * grip_color[:, :, 0:1, :].squeeze(2)[:, :, None, None, :]

        rgb = torch.maximum(rgb, disk_rgb)

        # axis lines
        for k in range(3):
            p1 = u_axes[:, :, k, :]  # (B,T,2)
            dist_line = _point_line_distance_2d(grid, u0, p1)
            line_mask = (dist_line <= constants.line_width).to(dtype=dtype)
            col = axis_colors[k][None, None, None, None, :]  # (1,1,1,1,3)
            rgb = torch.maximum(rgb, line_mask[:, :, :, :, None] * col)

    return rgb.clamp(0.0, 1.0)

