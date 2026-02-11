"""
Module Latency Profiling Script for FastVGGT

This script measures the latency of different modules in the FastVGGT model
across various frame counts and datasets (7Scenes, ScanNet, dummy).

It helps identify bottlenecks and measure the impact of token merging on
different model components.

Usage:
    python tests/measure_module_latency.py --dataset_type 7scenes --data_dir /path/to/7scenes
    python tests/measure_module_latency.py --dataset_type scannet --data_dir /path/to/scannet
    python tests/measure_module_latency.py --dataset_type dummy
"""

import os
import sys
import torch
import time
import pandas as pd
import numpy as np
import argparse
from tqdm import tqdm
from collections import defaultdict
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union

# Ensure project root is on sys.path
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT

# ============================================================================
# Global Configuration Variables - Modify these to change testing parameters
# ============================================================================

# Different frame counts to test
FRAME_COUNTS = [5, 10, 20]

# Test batch sizes per frame count (adjust for GPU memory)
BATCH_SIZES = {
    5: 3,
    10: 2,
    20: 1,
    30: 1,
    50: 1,
    100: 1,
}

# Merge ratios: 0.9 = with merging (fast), 0.0 = no merging (baseline)
MERGE_RATIOS = [0.9, 0.0]

# Number of runs for averaging latency measurements
NUM_RUNS = 5


# ============================================================================
# Timing Utilities
# ============================================================================

class CudaTimer:
    """Context manager for CUDA timing with synchronization."""
    
    def __init__(self, name: str, timing_dict: Optional[Dict] = None):
        self.name = name
        self.timing_dict = timing_dict
        self.start_time = None
        
    def __enter__(self):
        torch.cuda.synchronize()
        self.start_time = time.time()
        return self
        
    def __exit__(self, *args):
        torch.cuda.synchronize()
        elapsed_ms = (time.time() - self.start_time) * 1000
        if self.timing_dict is not None:
            self.timing_dict[self.name] = elapsed_ms


def inject_timing_hooks(model: torch.nn.Module) -> None:
    """
    Inject timing hooks into model modules to measure latency.
    
    This function wraps forward methods of key modules with timing.
    """
    # Wrap aggregator components
    original_aggregator_forward = model.aggregator.forward
    
    def timed_aggregator_forward(images, timing_info=None):
        if timing_info is None:
            timing_info = {}
        with CudaTimer("aggregator_total", timing_info):
            return original_aggregator_forward(images, timing_info=timing_info)
    
    model.aggregator.forward = timed_aggregator_forward
    
    # Wrap head components
    if model.camera_head is not None:
        original_camera_forward = model.camera_head.forward
        def timed_camera_forward(tokens):
            start = time.time()
            torch.cuda.synchronize()
            result = original_camera_forward(tokens)
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - start) * 1000
            # Store timing in model's timing_dict if available
            return result
        model.camera_head.forward = timed_camera_forward
    
    if model.depth_head is not None:
        original_depth_forward = model.depth_head.forward
        def timed_depth_forward(aggregated_tokens_list, images, patch_start_idx):
            start = time.time()
            torch.cuda.synchronize()
            result = original_depth_forward(aggregated_tokens_list, images, patch_start_idx)
            torch.cuda.synchronize()
            elapsed_ms = (time.time() - start) * 1000
            return result
        model.depth_head.forward = timed_depth_forward


# ============================================================================
# Data Loading Functions
# ============================================================================

def load_scannet_data(
    data_dir: Union[str, Path],
    num_frames: int,
    num_samples: int = 1
) -> torch.Tensor:
    """
    Load ScanNet data for testing.
    
    Args:
        data_dir: Path to ScanNet root directory
        num_frames: Number of frames per sample
        num_samples: Number of different scenes/samples to load
        
    Returns:
        Tensor of shape [B, S, 3, H, W] where B=num_samples, S=num_frames
    """
    from vggt.utils.eval_utils import (
        get_sorted_image_paths, 
        load_images_rgb, 
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    scenes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(data_dir / d)])
    
    if not scenes:
        raise ValueError(f"No scenes found in {data_dir}")
    
    num_samples = min(num_samples, len(scenes))
    selected_scenes = scenes[:num_samples]
    
    all_images = []
    for scene in selected_scenes:
        scene_dir = data_dir / scene
        images_dir = scene_dir / "color"
        image_paths = get_sorted_image_paths(images_dir)
        
        actual_frames = min(num_frames, len(image_paths))
        if len(image_paths) < num_frames:
            print(f"  Scene {scene}: requested {num_frames} frames, got {actual_frames}")
        
        selected_paths = image_paths[:actual_frames]
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0)  # [num_samples, S, 3, H, W]


def load_7scenes_data(
    data_dir: Union[str, Path],
    num_frames: int,
    resolution: Tuple[int, int] = (518, 392),
    num_samples: int = 1
) -> torch.Tensor:
    """
    Load 7Scenes data for testing.
    
    Args:
        data_dir: Path to 7Scenes root directory
        num_frames: Number of frames per sample
        resolution: Input resolution (H, W)
        num_samples: Number of different sequences to load
        
    Returns:
        Tensor of shape [B, S, 3, H, W] where B=num_samples, S=num_frames
    """
    sys.path.append(os.path.join(ROOT_DIR, "eval"))
    from data import SevenScenes
    
    dataset = SevenScenes(
        split="test",
        ROOT=str(data_dir),
        resolution=resolution,
        num_seq=1,
        full_video=True,
        kf_every=1,
    )
    
    if len(dataset) == 0:
        raise ValueError(f"No data found in 7scenes dataset at {data_dir}")
    
    num_samples = min(num_samples, len(dataset))
    all_images = []
    
    for sample_idx in range(num_samples):
        views = dataset[sample_idx]
        actual_frames = min(num_frames, len(views))
        
        if len(views) < num_frames:
            print(f"  Sequence {sample_idx}: requested {num_frames} frames, got {len(views)}")
        
        selected_views = views[:actual_frames]
        
        # Extract and stack images [S, 3, H, W]
        # Dataset returns images normalized to [-1, 1], convert to [0, 1]
        imgs = torch.stack([v["img"] for v in selected_views])
        imgs = (imgs + 1.0) / 2.0
        
        all_images.append(imgs.unsqueeze(0))  # [1, S, 3, H, W]
    
    return torch.cat(all_images, dim=0)  # [num_samples, S, 3, H, W]


def load_generic_images(
    data_dir: Union[str, Path],
    num_frames: int,
    num_samples: int = 1
) -> torch.Tensor:
    """
    Load images from a directory with flexible sampling.
    
    Args:
        data_dir: Path to image directory
        num_frames: Number of frames per sample
        num_samples: Number of different subsets to sample
        
    Returns:
        Tensor of shape [B, S, 3, H, W] where B=num_samples, S=num_frames
    """
    from vggt.utils.eval_utils import (
        get_sorted_image_paths,
        load_images_rgb,
        get_vgg_input_imgs
    )
    
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise ValueError(f"Provided path is not a directory: {data_dir}")
    
    image_paths = get_sorted_image_paths(data_dir)
    if not image_paths:
        raise ValueError(f"No images found in {data_dir}")
    
    all_images = []
    total_images = len(image_paths)
    stride = max(1, total_images // num_samples) if num_samples > 1 else 1
    
    for sample_idx in range(num_samples):
        start_idx = sample_idx * stride
        if start_idx >= total_images:
            break
        
        end_idx = min(start_idx + num_frames, total_images)
        selected_paths = image_paths[start_idx:end_idx]
        
        if len(selected_paths) < num_frames:
            print(f"  Sample {sample_idx}: requested {num_frames} frames, got {len(selected_paths)}")
        
        images = load_images_rgb(selected_paths)
        images_array = np.stack(images)
        vgg_input, _, _ = get_vgg_input_imgs(images_array)
        all_images.append(vgg_input)
    
    return torch.cat(all_images, dim=0)  # [num_samples, S, 3, H, W]


def generate_dummy_data(
    num_frames: int,
    batch_size: int = 1,
    resolution: Tuple[int, int] = (518, 392)
) -> torch.Tensor:
    """
    Generate dummy random data for quick testing.
    
    Args:
        num_frames: Number of frames
        batch_size: Batch size
        resolution: Image resolution (H, W)
        
    Returns:
        Tensor of shape [B, S, 3, H, W] with random values in [0, 1]
    """
    H, W = resolution
    return torch.rand(batch_size, num_frames, 3, H, W)


# ============================================================================
# Result Aggregation and Processing
# ============================================================================

def aggregate_timing_info(
    timing_dict: Dict[str, float],
    num_frames: int,
    seq_len: int,
    merge_ratio: float,
    merging: int,
    dataset_type: str,
    mode: str,
    batch_size: int
) -> Dict:
    """
    Aggregate and summarize raw timing measurements.
    
    Args:
        timing_dict: Raw timing dictionary from measurement
        num_frames: Actual frames processed
        seq_len: Requested sequence length
        merge_ratio: Token merge ratio (0.0-1.0)
        merging: Merging threshold
        dataset_type: Dataset name
        mode: 'with_merge' or 'no_merge'
        batch_size: Batch size used
        
    Returns:
        Dictionary with aggregated metrics
    """
    if not timing_dict:
        return {
            'seq_len': seq_len,
            'actual_frames': num_frames,
            'batch_size': batch_size,
            'merge_ratio': merge_ratio,
            'dataset': dataset_type,
            'mode': mode,
            'total_time_ms': 0.0,
            'throughput_fps': 0.0,
            'top1_module': 'N/A',
            'top1_time_ms': 0.0,
            'top1_percent': 0.0,
            'top5_summary': 'N/A',
        }
    
    total_ms = sum(timing_dict.values())
    sorted_modules = sorted(timing_dict.items(), key=lambda x: x[1], reverse=True)
    
    top1_module = sorted_modules[0][0] if sorted_modules else 'N/A'
    top1_time = sorted_modules[0][1] if sorted_modules else 0.0
    top1_percent = (top1_time / total_ms * 100) if total_ms > 0 else 0.0
    
    top5_summary = "; ".join([
        f"{name}:{time:.1f}ms:{time/total_ms*100:.1f}%"
        for name, time in sorted_modules[:5]
    ]) if sorted_modules else "N/A"
    
    result = {
        'seq_len': seq_len,
        'actual_frames': num_frames,
        'batch_size': batch_size,
        'merge_ratio': merge_ratio,
        'merging_threshold': merging,
        'dataset': dataset_type,
        'mode': mode,
        'total_time_ms': round(total_ms, 2),
        'throughput_fps': round(num_frames / (total_ms / 1000.0), 2) if total_ms > 0 else 0.0,
        'top1_module': top1_module,
        'top1_time_ms': round(top1_time, 2),
        'top1_percent': round(top1_percent, 1),
        'top5_summary': top5_summary,
    }
    
    return result


def save_result_incremental(result_dict: Dict, output_file: str) -> None:
    """
    Save result to CSV file incrementally (append mode).
    
    Args:
        result_dict: Single result dictionary to append
        output_file: Path to CSV file
    """
    df = pd.DataFrame([result_dict])
    file_exists = os.path.isfile(output_file)
    df.to_csv(
        output_file,
        mode='a',
        header=not file_exists,
        index=False
    )
    print(f"    ✓ Result saved to {output_file}")


# ============================================================================
# Model Latency Measurement
# ============================================================================

def measure_latency(
    model: torch.nn.Module,
    images: torch.Tensor,
    device: str,
    merge_ratio: float = 0.9,
    merging: int = 0
) -> Dict[str, float]:
    """
    Measure module latencies during model forward pass.
    
    Args:
        model: VGGT model
        images: Input images [B, S, 3, H, W]
        device: Device to run on ('cuda' or 'cpu')
        merge_ratio: Token merge ratio
        merging: Merging threshold
        
    Returns:
        Dictionary of module latencies {module_name: latency_ms}
    """
    # Configure merging parameters
    model.aggregator.merging = merging
    for block in model.aggregator.frame_blocks:
        if hasattr(block, 'attn'):
            block.attn.merge_ratio = merge_ratio
    for block in model.aggregator.global_blocks:
        if hasattr(block, 'attn'):
            block.attn.merge_ratio = merge_ratio
    
    # Convert images to device (model is already on device)
    images = images.to(device).float()
    
    # Warmup runs
    with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        for _ in range(3):
            try:
                _ = model(images)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"    ⚠ OOM during warmup, skipping this configuration")
                return {}
    
    # Measurement runs - measure overall latency
    total_time_ms = 0.0
    num_runs = NUM_RUNS
    
    for run_idx in range(num_runs):
        try:
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                torch.cuda.synchronize()
                start_time = time.time()
                _ = model(images)
                torch.cuda.synchronize()
                elapsed_ms = (time.time() - start_time) * 1000
                total_time_ms += elapsed_ms
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"    ⚠ OOM during run {run_idx+1}/{num_runs}, skipping")
            return {}
    
    avg_time_ms = total_time_ms / num_runs
    
    # For now, return a simple dictionary with total time
    # In future, can add per-module timing by wrapping forward methods
    timing_info = {
        'model_total': avg_time_ms
    }
    
    return timing_info


# ============================================================================
# Main Testing Loop
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Measure FastVGGT module latency with different frame counts"
    )
    parser.add_argument(
        "--dataset_type",
        type=str,
        choices=["7scenes", "scannet", "dummy", "images"],
        default="dummy",
        help="Type of dataset to use. Default: dummy"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Path to dataset root directory (required for non-dummy datasets)"
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/hba/Documents/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default="./tests_result/module_latency_report.csv",
        help="Path to save results CSV"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        nargs=2,
        default=[518, 392],
        help="Input resolution (H W) for 7Scenes. Options: 518 392, 512 384, 224 224"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=1,
        help="Number of different scenes/samples to test"
    )
    args = parser.parse_args()
    
    # Validate resolution for 7Scenes
    valid_resolutions = [(518, 392), (512, 384), (224, 224)]
    resolution_tuple = tuple(args.resolution)
    if resolution_tuple not in valid_resolutions and args.dataset_type == "7scenes":
        print(f"⚠ Warning: resolution {resolution_tuple} not in standard options. Using as-is.")
    
    # Validate data_dir for non-dummy datasets
    if args.dataset_type != "dummy" and not args.data_dir:
        print(f"⚠ Warning: dataset_type '{args.dataset_type}' requires --data_dir. Falling back to dummy data.")
        args.dataset_type = "dummy"
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Initialize model
    model = VGGT(
        merging=0,
        merge_ratio=0.9,
        enable_point=True,
        enable_depth=True,
        enable_camera=True
    )
    
    if args.ckpt_path and os.path.exists(args.ckpt_path):
        ckpt = torch.load(args.ckpt_path, map_location="cpu")
        model.load_state_dict(ckpt, strict=False)
        print(f"Loaded checkpoint from {args.ckpt_path}")
    else:
        print(f"⚠ Warning: checkpoint not found at {args.ckpt_path}, using random initialization")
    
    model = model.to(device).eval()
    
    model_dtype = next(model.parameters()).dtype
    print(f"Model dtype: {model_dtype}")
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    
    # Main testing loop over frame counts
    for seq_len in tqdm(FRAME_COUNTS, desc="Testing different frame counts"):
        print(f"\n>>> Processing sequence length: {seq_len}")
        
        # Load data
        try:
            if args.dataset_type == "7scenes" and args.data_dir:
                images = load_7scenes_data(
                    args.data_dir,
                    seq_len,
                    resolution=tuple(args.resolution),
                    num_samples=args.num_samples
                )
                dataset_label = f"7scenes(res={tuple(args.resolution)})"
            elif args.dataset_type == "scannet" and args.data_dir:
                images = load_scannet_data(
                    args.data_dir,
                    seq_len,
                    num_samples=args.num_samples
                )
                dataset_label = "scannet"
            elif args.dataset_type == "images" and args.data_dir:
                images = load_generic_images(
                    args.data_dir,
                    seq_len,
                    num_samples=args.num_samples
                )
                dataset_label = "images"
            else:
                # Default: dummy data
                batch_size = BATCH_SIZES.get(seq_len, 1)
                images = generate_dummy_data(seq_len, batch_size=batch_size)
                dataset_label = "dummy"
        except Exception as e:
            print(f"✗ Error loading data for {args.dataset_type}: {e}")
            continue
        
        actual_seq_len = images.shape[1]
        batch_size = images.shape[0]
        
        print(f"  Loaded data: shape={images.shape}, dtype={images.dtype}")
        
        # Test with different merge ratios
        for merge_ratio in MERGE_RATIOS:
            mode = 'with_merge' if merge_ratio > 0 else 'no_merge'
            merging_threshold = 0 if merge_ratio > 0 else 25
            
            print(f"  Measuring {mode} (ratio={merge_ratio})...")
            
            timing_dict = measure_latency(
                model,
                images,
                device,
                merge_ratio=merge_ratio,
                merging=merging_threshold
            )
            
            if not timing_dict:
                print(f"    ✓ Skipped due to OOM or other error")
                continue
            
            # Aggregate results
            result = aggregate_timing_info(
                timing_dict,
                actual_seq_len,
                seq_len,
                merge_ratio=merge_ratio,
                merging=merging_threshold,
                dataset_type=dataset_label,
                mode=mode,
                batch_size=batch_size
            )
            
            # Save incrementally
            save_result_incremental(result, args.output_csv)
            
            # Print summary
            print(f"    Total: {result['total_time_ms']:.2f}ms, "
                  f"FPS: {result['throughput_fps']:.2f}, "
                  f"Top module: {result['top1_module']} ({result['top1_percent']:.1f}%)")
    
    print(f"\n✓ All tests completed! Results saved to: {args.output_csv}")


if __name__ == "__main__":
    main()
