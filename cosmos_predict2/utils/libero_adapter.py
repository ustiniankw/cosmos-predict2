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
LIBERO adapter utilities (ProphRL Sec 3.2.1).

LIBERO provides *servo commands* for the end-effector control stream. In Prophet, we visualize control
signals by rendering action frames from 7D commands plus camera parameters (K/E).

This module intentionally keeps the HDF5 parsing *robust* because LIBERO HDF5 layouts differ across
versions/tasks (obs vs observations, agentview vs wrist, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _require_h5py():
    try:
        import h5py  # type: ignore
    except ModuleNotFoundError as e:  # pragma: no cover
        raise ModuleNotFoundError(
            "Missing optional dependency 'h5py' required to read LIBERO HDF5 trajectories. "
            "Install it via `pip install h5py`."
        ) from e
    return h5py


@dataclass(frozen=True)
class LiberoTrajectory:
    """
    A single demonstration trajectory.

    - rgb:    (T,H,W,3) uint8
    - action: (T,7) float32
    - K:      (3,3) float32 (or broadcastable)
    - E:      (4,4) float32 (or broadcastable)
    """

    rgb: np.ndarray
    action: np.ndarray
    K: np.ndarray
    E: np.ndarray


class LiberoHDF5TrajectoryDataset:
    """
    Minimal Dataset-like wrapper around a single LIBERO HDF5 file.

    This is intentionally simple (no sharding, no worker-safe file handles):
    it loads the requested demo into memory on __getitem__.
    """

    def __init__(self, hdf5_path: str, *, camera: str | None = None) -> None:
        self.hdf5_path = hdf5_path
        self.camera = camera

    def __len__(self) -> int:
        # Best-effort: count demo groups if present; otherwise treat as 1.
        h5py = _require_h5py()
        with h5py.File(self.hdf5_path, "r") as f:
            if "data" in f:
                data = f["data"]
                # Common naming: demo_0, demo_1, ...
                demo_keys = [k for k in data.keys() if str(k).startswith("demo_")]
                return len(demo_keys) if demo_keys else len(list(data.keys()))
        return 1

    def __getitem__(self, idx: int) -> LiberoTrajectory:
        return load_libero_hdf5_trajectory(self.hdf5_path, demo_index=int(idx), camera=self.camera)


def _all_dataset_paths(h5: Any) -> list[str]:
    paths: list[str] = []

    def _visitor(name: str, obj: Any) -> None:
        # h5py.Dataset has attribute "shape"
        if hasattr(obj, "shape"):
            paths.append(name)

    h5.visititems(_visitor)
    return paths


def _try_get(h5: Any, path: str) -> Any | None:
    try:
        return h5[path]
    except Exception:
        return None


def _first_existing(h5: Any, candidates: list[str]) -> Any | None:
    for p in candidates:
        obj = _try_get(h5, p)
        if obj is not None:
            return obj
    return None


def _normalize_rgb(arr: np.ndarray) -> np.ndarray:
    """
    Normalize an arbitrary image tensor to (T,H,W,3) uint8.
    """
    if arr.ndim == 4 and arr.shape[-1] in (3, 4):  # (T,H,W,C)
        out = arr[..., :3]
    elif arr.ndim == 4 and arr.shape[1] in (3, 4):  # (T,C,H,W)
        out = np.transpose(arr[:, :3, :, :], (0, 2, 3, 1))
    else:
        raise ValueError(f"Unsupported rgb array shape: {arr.shape}")

    if out.dtype != np.uint8:
        # Common variants: float [0,1] or [0,255]
        if np.issubdtype(out.dtype, np.floating):
            mx = float(np.max(out)) if out.size else 1.0
            if mx <= 1.0 + 1e-3:
                out = (out * 255.0).round()
        out = np.clip(out, 0, 255).astype(np.uint8)
    return out


def _normalize_action(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape[-1] == 7:
        return arr.astype(np.float32)
    if arr.ndim == 2 and arr.shape[0] == 7:
        return arr.T.astype(np.float32)
    raise ValueError(f"Unsupported action array shape (expected (T,7)): {arr.shape}")


def _normalize_K(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape == (3, 3):
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (3, 3):
        return arr[0].astype(np.float32)
    raise ValueError(f"Unsupported K shape: {arr.shape}")


def _normalize_E(arr: np.ndarray) -> np.ndarray:
    if arr.ndim == 2 and arr.shape == (4, 4):
        return arr.astype(np.float32)
    if arr.ndim == 3 and arr.shape[-2:] == (4, 4):
        return arr[0].astype(np.float32)
    raise ValueError(f"Unsupported E shape: {arr.shape}")


def load_libero_hdf5_trajectory(
    hdf5_path: str,
    *,
    demo_index: int = 0,
    camera: str | None = None,
    max_frames: int | None = None,
) -> LiberoTrajectory:
    """
    Load one trajectory from a LIBERO HDF5 file.

    This function uses best-effort key discovery:
    - RGB: prefers datasets containing "rgb" and (T,*,*,3/4)
    - action: prefers datasets named "action"/"actions" with (T,7)
    - K/E: looks for "K"/"intrinsics" and "E"/"extrinsics"; defaults to identity if missing.
    """
    h5py = _require_h5py()
    with h5py.File(hdf5_path, "r") as f:
        # Locate the demo group if present.
        demo_candidates = [
            f"data/demo_{demo_index}",
            f"data/{demo_index}",
            f"demo_{demo_index}",
            str(demo_index),
        ]
        demo_group = _first_existing(f, demo_candidates)
        root = demo_group if demo_group is not None else f

        # Candidate paths for rgb and action under the chosen root.
        cam = camera or "agentview"
        rgb_candidates = [
            f"observations/{cam}_rgb",
            f"observations/{cam}_image",
            f"observations/images/{cam}",
            f"observations/images/{cam}_rgb",
            f"obs/{cam}_rgb",
            f"obs/{cam}_image",
            f"obs/images/{cam}",
            f"obs/images/{cam}_rgb",
            "observations/rgb",
            "obs/rgb",
        ]
        action_candidates = [
            "actions",
            "action",
            "actions/ee",
            "action/ee",
        ]
        K_candidates = [
            "K",
            "camera_intrinsics",
            "camera/intrinsics",
            "observations/K",
            "obs/K",
        ]
        E_candidates = [
            "E",
            "camera_extrinsics",
            "camera/extrinsics",
            "observations/E",
            "obs/E",
        ]

        rgb_ds = _first_existing(root, rgb_candidates)
        action_ds = _first_existing(root, action_candidates)
        K_ds = _first_existing(root, K_candidates)
        E_ds = _first_existing(root, E_candidates)

        # Fallback: global scan
        if rgb_ds is None or action_ds is None:
            for p in _all_dataset_paths(root):
                ds = root[p]
                shape = getattr(ds, "shape", None)
                if shape is None:
                    continue
                name = p.lower()
                if rgb_ds is None and ("rgb" in name or "image" in name) and len(shape) == 4:
                    c_last = shape[-1]
                    c_mid = shape[1]
                    if c_last in (3, 4) or c_mid in (3, 4):
                        rgb_ds = ds
                if action_ds is None and ("action" in name) and len(shape) == 2 and shape[-1] == 7:
                    action_ds = ds
                if rgb_ds is not None and action_ds is not None:
                    break

        if rgb_ds is None:
            raise KeyError(
                f"Failed to locate RGB dataset in {hdf5_path}. "
                f"Try passing --camera (e.g. agentview/wrist) or inspect keys."
            )
        if action_ds is None:
            raise KeyError(f"Failed to locate action dataset (T,7) in {hdf5_path}.")

        rgb = _normalize_rgb(np.asarray(rgb_ds))
        action = _normalize_action(np.asarray(action_ds))

        if max_frames is not None:
            rgb = rgb[:max_frames]
            action = action[:max_frames]

        if K_ds is not None:
            K = _normalize_K(np.asarray(K_ds))
        else:
            K = np.eye(3, dtype=np.float32)
        if E_ds is not None:
            E = _normalize_E(np.asarray(E_ds))
        else:
            E = np.eye(4, dtype=np.float32)

    # Align lengths: LIBERO often has T actions aligned with T frames, but occasionally action is T-1.
    T_rgb = rgb.shape[0]
    T_act = action.shape[0]
    T = min(T_rgb, T_act)
    if T != T_rgb or T != T_act:
        rgb = rgb[:T]
        action = action[:T]

    return LiberoTrajectory(rgb=rgb, action=action, K=K, E=E)


def libero_servo_actions_to_prophet(actions_T_7: np.ndarray) -> np.ndarray:
    """
    LIBERO actions are servo commands. Prophet's scalar stream consumes 7D action chunks directly.

    This function exists mostly for explicitness / future normalization hooks.
    """
    return _normalize_action(actions_T_7)

