"""
Module Latency Profiling Script for FastVGGT

This script measures the latency of each module in the FastVGGT model
across various frame counts and merging settings.

The results include only essential information: frame_count, merge_ratio,
and per-module inference times.
"""

import os
import sys
import gc
import torch
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT


# ============================================================================
# Module Timing with Forward Hooks
# ============================================================================

class LatencyProfiler:
    """Measures per-module inference latency using forward hooks."""

    def __init__(self, model: torch.nn.Module):
        self.model = model
        self.module_times = {}
        self.handles = []
        self.accumulated_times = {}  # Accumulate times across multiple runs
    
    def _get_module_name(self, module: torch.nn.Module) -> str:
        """Get a descriptive name for a module."""
        for name, mod in self.model.named_modules():
            if mod is module:
                return name
        return "unknown"
    
    def register_hooks(self):
        """Register forward hooks on key modules."""
        
        # Aggregator components
        if hasattr(self.model, 'aggregator'):
            agg = self.model.aggregator
            
            # Patch embed
            if hasattr(agg, 'patch_embed'):
                self._hook_module(agg.patch_embed, 'patch_embed')
            
            # Frame blocks
            if hasattr(agg, 'frame_blocks'):
                for i, block in enumerate(agg.frame_blocks):
                    self._hook_module(block, f'frame_block_{i}')
            
            # Global blocks  
            if hasattr(agg, 'global_blocks'):
                for i, block in enumerate(agg.global_blocks):
                    self._hook_module(block, f'global_block_{i}')
        
        # Heads
        if hasattr(self.model, 'camera_head') and self.model.camera_head is not None:
            self._hook_module(self.model.camera_head, 'camera_head')
        
        if hasattr(self.model, 'depth_head') and self.model.depth_head is not None:
            self._hook_module(self.model.depth_head, 'depth_head')
        
        if hasattr(self.model, 'track_head') and self.model.track_head is not None:
            self._hook_module(self.model.track_head, 'track_head')
    
    def _hook_module(self, module: torch.nn.Module, name: str):
        """Attach timing hooks to a module."""
        
        def forward_pre_hook(m, input):
            torch.cuda.synchronize()
            m._time_start = time.time()
        
        def forward_hook(m, input, output):
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - m._time_start) * 1000
            if name not in self.module_times:
                self.module_times[name] = []
            self.module_times[name].append(elapsed_ms)
        
        h1 = module.register_forward_pre_hook(forward_pre_hook)
        h2 = module.register_forward_hook(forward_hook)
        self.handles.append(h1)
        self.handles.append(h2)
    
    def reset(self):
        """Clear timing history for current run, but preserve accumulated data."""
        # Store current run's times into accumulated_times before clearing
        for name, times in self.module_times.items():
            if name not in self.accumulated_times:
                self.accumulated_times[name] = []
            if times:  # If there are times from current run, store their sum
                self.accumulated_times[name].append(sum(times))
        self.module_times.clear()
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for h in self.handles:
            h.remove()
        self.handles.clear()
    
    def get_avg_times(self, num_runs: int = 1) -> Dict[str, float]:
        """Get average times for each module across runs, aggregating frame/global blocks."""
        result = {}

        # First, store the current run's data if any
        for name, times in self.module_times.items():
            if name not in self.accumulated_times:
                self.accumulated_times[name] = []
            if times:
                self.accumulated_times[name].append(sum(times))

        # Aggregate frame_block and global_block times
        frame_block_totals = []
        global_block_totals = []
        
        for name, run_totals in self.accumulated_times.items():
            if not run_totals:
                continue
            avg_time = np.mean(run_totals)
            
            if name.startswith('frame_block_'):
                frame_block_totals.append(avg_time)
            elif name.startswith('global_block_'):
                global_block_totals.append(avg_time)
            else:
                # Keep other modules as-is
                result[name] = avg_time
        
        # Add aggregated statistics
        if frame_block_totals:
            result['frame_blocks_total'] = sum(frame_block_totals)
            result['frame_blocks_avg'] = np.mean(frame_block_totals)
        if global_block_totals:
            result['global_blocks_total'] = sum(global_block_totals)
            result['global_blocks_avg'] = np.mean(global_block_totals)

        return result

    def clear_accumulated(self):
        """Clear accumulated times for a new test configuration."""
        self.accumulated_times.clear()


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_7scenes_data(
    data_dir: str,
    num_frames: int,
    resolution: Tuple[int, int] = (518, 392),
    num_samples: int = 1
) -> torch.Tensor:
    """Load 7Scenes data."""
    sys.path.insert(0, os.path.join(ROOT_DIR, "eval"))
    from data import SevenScenes
    
    dataset = SevenScenes(
        split="test",
        ROOT=data_dir,
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=1,
    )
    
    all_images = []
    for idx in range(min(num_samples, len(dataset))):
        views = dataset[idx]
        actual_frames = min(num_frames, len(views))
        selected_views = views[:actual_frames]
        
        # Normalize and stack
        imgs = torch.stack([v["img"] for v in selected_views])
        imgs = (imgs + 1.0) / 2.0  # [-1, 1] -> [0, 1]
        all_images.append(imgs.unsqueeze(0))
    
    return torch.cat(all_images, dim=0)


def load_scannet_data(
    data_dir: str,
    num_frames: int,
    num_samples: int = 1
) -> torch.Tensor:
    """Load ScanNet data."""
    from vggt.utils.eval_utils import (
        get_sorted_image_paths, 
        load_images_rgb, 
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)])
    
    all_images = []
    for scene in scenes[:num_samples]:
        scene_dir = data_dir / scene
        images_dir = scene_dir / "color"
        image_paths = get_sorted_image_paths(images_dir)
        
        actual_frames = min(num_frames, len(image_paths))
        selected_paths = image_paths[:actual_frames]
        
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0)


def load_images(
    data_dir: str,
    num_frames: int,
    num_samples: int = 1
) -> torch.Tensor:
    """Load generic images."""
    from vggt.utils.eval_utils import (
        get_sorted_image_paths,
        load_images_rgb,
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    image_paths = get_sorted_image_paths(data_dir)
    
    all_images = []
    total_images = len(image_paths)
    stride = max(1, total_images // num_samples)
    
    for idx in range(num_samples):
        start_idx = idx * stride
        if start_idx >= total_images:
            break
        end_idx = min(start_idx + num_frames, total_images)
        selected_paths = image_paths[start_idx:end_idx]
        
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0)

# ============================================================================
# Global Configuration Variables
# ============================================================================

FRAME_COUNTS = [5, 10, 20, 50, 100]
MERGE_RATIOS = [0.0]
NUM_RUNS = 3  # Number of averaging runs

PATH_FOR_7SCENES = "/home/hba/Documents/Dataset/7_scenes"
PATH_FOR_SCANNET = "/home/hba/Documents/Dataset/ScanNet/scans"

# ============================================================================
# Main Script
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Measure module latencies")
    parser.add_argument("--dataset_type", choices=["7scenes", "scannet", "images"], 
                       default="7scenes")
    parser.add_argument("--data_dir", type=str,
                       default=PATH_FOR_7SCENES,
                       help="Path to dataset")
    parser.add_argument("--ckpt_path", type=str, 
                       default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt")
    parser.add_argument("--resolution", type=int, nargs=2, default=[518, 392],
                       help="Resolution for 7Scenes (H W)")
    parser.add_argument("--num_samples", type=int, default=1,
                       help="Number of samples per config")
    parser.add_argument("--frame_counts", type=int, nargs="+", default=None)
    args = parser.parse_args()
    
    if not os.path.exists(args.data_dir):
        raise ValueError(f"Data dir not found: {args.data_dir}")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    
    # Load model
    model = VGGT(merging=0, merge_ratio=0.9, enable_point=True, 
                 enable_depth=True, enable_camera=True)
    
    if os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint: {args.ckpt_path}")
    
    model = model.to(device).eval().to(torch.float16)
    
    # Output path (different for each dataset)
    output_dir = "/home/hba/Documents/FastVGGT/tests/tests_result/module_latency"
    os.makedirs(output_dir, exist_ok=True)
    output_csv = os.path.join(output_dir, f"module_latency_{args.dataset_type}.csv")
    
    # Initialize CSV file (will write header on first append)
    csv_initialized = False
    
    frame_counts = args.frame_counts if args.frame_counts else FRAME_COUNTS
    
    print(f"\n{'='*60}")
    print(f"Dataset: {args.dataset_type}")
    print(f"Output: {output_csv}")
    print(f"Frame counts: {frame_counts}")
    print(f"Merge ratios: {MERGE_RATIOS}")
    print(f"{'='*60}\n")
    
    # Main loop
    results = []
    
    for seq_len in tqdm(frame_counts, desc="Testing"):
        # Load data
        try:
            if args.dataset_type == "7scenes":
                images = load_7scenes_data(args.data_dir, seq_len, 
                                          tuple(args.resolution), args.num_samples)
            elif args.dataset_type == "scannet":
                images = load_scannet_data(args.data_dir, seq_len, args.num_samples)
            else:  # images
                images = load_images(args.data_dir, seq_len, args.num_samples)
        except Exception as e:
            print(f"Error loading data: {e}")
            continue
        
        # Test each merge ratio
        for merge_ratio in MERGE_RATIOS:
            merging_threshold = 0 if merge_ratio > 0 else 25
            
            mode = 'with_merge' if merge_ratio > 0 else 'no_merge'
            
            # Configure model
            model.aggregator.merging = merging_threshold
            for block in model.aggregator.frame_blocks:
                if hasattr(block, 'attn'):
                    block.attn.merge_ratio = merge_ratio
            for block in model.aggregator.global_blocks:
                if hasattr(block, 'attn'):
                    block.attn.merge_ratio = merge_ratio
            
            # Create profiler
            profiler = LatencyProfiler(model)
            profiler.register_hooks()
            
            images_device = images.to(device, dtype=torch.float16)
            
            # Warmup
            warmup_oom = False
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    for _ in range(2):
                        try:
                            _ = model(images_device)
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            print(f"  OOM at frame={seq_len}, {mode}")
                            profiler.remove_hooks()
                            warmup_oom = True
                            break
            
            if warmup_oom:
                continue
            
            # Discard warmup data without saving to accumulated_times
            profiler.module_times.clear()
            
            # Measurement runs - accumulate times across all runs
            total_times = []
            oom_occurred = False
            
            for run_idx in range(NUM_RUNS):
                profiler.reset()
                try:
                    with torch.no_grad():
                        with torch.cuda.amp.autocast(dtype=torch.float16):
                            torch.cuda.synchronize()
                            start = time.time()
                            _ = model(images_device)
                            torch.cuda.synchronize()
                            elapsed_ms = (time.time() - start) * 1000
                            total_times.append(elapsed_ms)
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    print(f"  OOM at frame={seq_len}, {mode}")
                    profiler.remove_hooks()
                    oom_occurred = True
                    break
            
            if oom_occurred:
                continue
            
            # Calculate average total time across all runs
            avg_total_ms = np.mean(total_times) if total_times else 0.0
            
            # Get average times for each module
            avg_times = profiler.get_avg_times()
            profiler.remove_hooks()
            
            if not avg_times:
                continue
            
            # Build result row
            row = {
                'frame_count': seq_len,
                'merge_ratio': merge_ratio,
                'total_ms': avg_total_ms,
            }
            row.update(avg_times)
            results.append(row)
            
            print(f"  ✓ frame={seq_len}, {mode:12s} total={avg_total_ms:7.2f}ms")
            
            # Append to CSV immediately (after timing is complete)
            df_row = pd.DataFrame([row])
            if not csv_initialized:
                df_row.to_csv(output_csv, index=False, mode='w')
                csv_initialized = True
            else:
                df_row.to_csv(output_csv, index=False, mode='a', header=False)
            
            # Clear GPU memory and run garbage collection
            del images_device, profiler, avg_times, total_times, row, df_row
            torch.cuda.empty_cache()
            gc.collect()
        
        # Clear loaded images after all merge_ratio tests
        del images
        torch.cuda.empty_cache()
        gc.collect()
    
    # Print summary
    if results:
        print(f"\n{'='*60}")
        print(f"✓ Results saved to: {output_csv}")
        print(f"{'='*60}")
        df = pd.DataFrame(results)
        print(df.to_string(index=False))
    else:
        print("No results to save")


if __name__ == "__main__":
    main()
