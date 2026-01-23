#!/usr/bin/env python3
"""
Diagnostic script to inspect LIBERO HDF5 data format.
This helps identify issues with action format, camera parameters, and data structure.
"""

import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from cosmos_predict2.utils.libero_adapter import load_libero_hdf5_trajectory


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--libero_dir", type=str, default="./datasets/libero_spatial")
    p.add_argument("--camera", type=str, default="agentview")
    p.add_argument("--demo_index", type=int, default=0)
    args = p.parse_args()

    libero_dir = Path(args.libero_dir)
    hdf5_files = sorted(libero_dir.rglob("*.hdf5"))
    
    if not hdf5_files:
        print(f"[ERROR] No HDF5 files found in {libero_dir}")
        return
    
    print(f"Found {len(hdf5_files)} HDF5 files")
    print("=" * 80)
    
    # Inspect first file in detail
    hdf5_path = hdf5_files[0]
    print(f"\n[Inspecting] {hdf5_path}")
    
    # Also print raw HDF5 structure
    try:
        import h5py
        with h5py.File(str(hdf5_path), "r") as f:
            print("\n--- HDF5 Structure ---")
            def print_structure(name, obj):
                indent = "  " * name.count("/")
                if hasattr(obj, "shape"):
                    print(f"{indent}{name}: shape={obj.shape}, dtype={obj.dtype}")
                else:
                    print(f"{indent}{name}/ (group)")
            f.visititems(print_structure)
    except Exception as e:
        print(f"[WARN] Could not inspect HDF5 structure: {e}")
    
    # Load trajectory
    print("\n--- Loaded Trajectory ---")
    traj = load_libero_hdf5_trajectory(str(hdf5_path), demo_index=args.demo_index, camera=args.camera)
    
    print(f"RGB shape: {traj.rgb.shape}, dtype: {traj.rgb.dtype}")
    print(f"Action shape: {traj.action.shape}, dtype: {traj.action.dtype}")
    print(f"K (intrinsics) shape: {traj.K.shape}")
    print(f"E (extrinsics) shape: {traj.E.shape}")
    
    # Check if K/E are identity (default fallback)
    K_is_identity = np.allclose(traj.K, np.eye(3))
    E_is_identity = np.allclose(traj.E, np.eye(4))
    
    print(f"\n--- Camera Parameters ---")
    print(f"K is identity matrix: {K_is_identity}")
    print(f"E is identity matrix: {E_is_identity}")
    print(f"\nK =\n{traj.K}")
    print(f"\nE =\n{traj.E}")
    
    if K_is_identity or E_is_identity:
        print("\n[WARNING] Camera parameters are identity matrices!")
        print("This means the HDF5 file does NOT contain camera intrinsics/extrinsics.")
        print("Action rendering will produce INCORRECT results!")
    
    # Action statistics
    print(f"\n--- Action Statistics ---")
    print(f"Action min: {traj.action.min(axis=0)}")
    print(f"Action max: {traj.action.max(axis=0)}")
    print(f"Action mean: {traj.action.mean(axis=0)}")
    print(f"Action std: {traj.action.std(axis=0)}")
    
    # Check action range to determine if it's delta or absolute
    action_range = traj.action.max(axis=0) - traj.action.min(axis=0)
    print(f"\nAction range per dim: {action_range}")
    
    # Heuristic: if position dims (0-2) have range > 1, likely absolute/servo
    # if range is small (< 0.1), likely delta
    pos_range = action_range[:3]
    if np.all(pos_range < 0.1):
        print("\n[HINT] Action position dims have small range (<0.1) - likely DELTA actions")
        print("       Consider using --action_mode delta")
    elif np.all(pos_range > 0.1):
        print("\n[HINT] Action position dims have larger range - likely SERVO/absolute actions")
        print("       Using --action_mode servo should be correct")
    
    # Show first few actions
    print(f"\n--- First 5 Actions ---")
    for i in range(min(5, len(traj.action))):
        print(f"  [{i}] {traj.action[i]}")
    
    # Test action rendering
    print(f"\n--- Testing Action Rendering ---")
    try:
        import torch
        from cosmos_predict2.utils.action_renderer import render_action_rgb
        
        actions = torch.from_numpy(traj.action[:20]).unsqueeze(0).float()  # (1, 20, 7)
        K = torch.from_numpy(traj.K).float()
        E = torch.from_numpy(traj.E).float()
        
        for mode in ["servo", "delta"]:
            rgb = render_action_rgb(actions, K, E, height=256, width=256, action_mode=mode)
            rgb_np = rgb[0].cpu().numpy()  # (T, H, W, 3)
            
            non_zero_pixels = (rgb_np > 0.01).sum()
            total_pixels = rgb_np.size
            non_zero_ratio = non_zero_pixels / total_pixels
            
            print(f"  mode={mode}: non_zero_ratio={non_zero_ratio:.4f}, mean={rgb_np.mean():.6f}, max={rgb_np.max():.4f}")
            
            if non_zero_ratio < 0.001:
                print(f"    [WARNING] Almost all pixels are zero! Rendering may be broken.")
    except Exception as e:
        print(f"  [ERROR] Rendering test failed: {e}")
    
    print("\n" + "=" * 80)
    print("Diagnosis complete.")


if __name__ == "__main__":
    main()
