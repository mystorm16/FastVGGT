"""
Analysis utilities for module latency profiling results.

This script provides helper functions to analyze and visualize the profiling results.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional


def load_results(csv_path: str) -> pd.DataFrame:
    """Load profiling results from CSV file."""
    return pd.read_csv(csv_path)


def calculate_speedup(df: pd.DataFrame) -> Dict[int, float]:
    """
    Calculate Token Merging speedup for each frame count.
    
    Args:
        df: Results dataframe
        
    Returns:
        Dictionary mapping frame count to speedup ratio
    """
    speedups = {}
    
    for seq_len in df['seq_len'].unique():
        subset = df[df['seq_len'] == seq_len]
        with_merge = subset[subset['mode'] == 'with_merge']['total_time_ms'].values
        no_merge = subset[subset['mode'] == 'no_merge']['total_time_ms'].values
        
        if len(with_merge) > 0 and len(no_merge) > 0 and with_merge[0] > 0:
            speedup = no_merge[0] / with_merge[0]
            speedups[seq_len] = speedup
        else:
            speedups[seq_len] = np.nan
    
    return speedups


def analyze_by_dataset(df: pd.DataFrame) -> Dict:
    """
    Analyze results grouped by dataset.
    
    Args:
        df: Results dataframe
        
    Returns:
        Dictionary with per-dataset analysis
    """
    analysis = {}
    
    for dataset in df['dataset'].unique():
        subset = df[df['dataset'] == dataset]
        analysis[dataset] = {
            'num_configs': len(subset),
            'avg_speedup': calculate_speedup(subset)['mean'],
            'min_time_ms': subset['total_time_ms'].min(),
            'max_time_ms': subset['total_time_ms'].max(),
        }
    
    return analysis


def print_speedup_summary(df: pd.DataFrame):
    """Print Token Merging speedup summary."""
    print("\n" + "="*60)
    print("Token Merging Speedup Analysis")
    print("="*60)
    
    speedups = calculate_speedup(df)
    
    print(f"\n{'Frames':<10} {'Speedup':<15} {'with_merge (ms)':<20} {'no_merge (ms)':<20}")
    print("-" * 65)
    
    for seq_len in sorted(speedups.keys()):
        subset = df[df['seq_len'] == seq_len]
        with_merge_time = subset[subset['mode'] == 'with_merge']['total_time_ms'].values
        no_merge_time = subset[subset['mode'] == 'no_merge']['total_time_ms'].values
        
        speedup = speedups[seq_len]
        with_time_str = f"{with_merge_time[0]:.2f}" if len(with_merge_time) > 0 else "N/A"
        no_time_str = f"{no_merge_time[0]:.2f}" if len(no_merge_time) > 0 else "N/A"
        
        if not np.isnan(speedup):
            print(f"{seq_len:<10} {speedup:<15.2f}x {with_time_str:<20} {no_time_str:<20}")
        else:
            print(f"{seq_len:<10} {'N/A':<15} {with_time_str:<20} {no_time_str:<20}")


def print_bottleneck_analysis(df: pd.DataFrame):
    """Print bottleneck module analysis."""
    print("\n" + "="*60)
    print("Bottleneck Module Analysis (Top 5)")
    print("="*60)
    
    for mode in ['with_merge', 'no_merge']:
        print(f"\n{mode.upper()} Mode:")
        print("-" * 60)
        
        subset = df[df['mode'] == mode]
        
        for seq_len in sorted(subset['seq_len'].unique()):
            config = subset[subset['seq_len'] == seq_len].iloc[0]
            print(f"\n  Seq {seq_len} frames:")
            print(f"    Total: {config['total_time_ms']:.2f}ms")
            print(f"    Top modules: {config['top5_summary']}")


def print_throughput_analysis(df: pd.DataFrame):
    """Print throughput (FPS) analysis."""
    print("\n" + "="*60)
    print("Throughput (FPS) Analysis")
    print("="*60)
    
    print(f"\n{'Dataset':<20} {'Frames':<10} {'with_merge':<15} {'no_merge':<15}")
    print("-" * 60)
    
    for dataset in sorted(df['dataset'].unique()):
        dataset_df = df[df['dataset'] == dataset]
        
        for seq_len in sorted(dataset_df['seq_len'].unique()):
            subset = dataset_df[dataset_df['seq_len'] == seq_len]
            
            with_merge = subset[subset['mode'] == 'with_merge']['throughput_fps'].values
            no_merge = subset[subset['mode'] == 'no_merge']['throughput_fps'].values
            
            with_str = f"{with_merge[0]:.2f}" if len(with_merge) > 0 else "N/A"
            no_str = f"{no_merge[0]:.2f}" if len(no_merge) > 0 else "N/A"
            
            print(f"{dataset:<20} {seq_len:<10} {with_str:<15} {no_str:<15}")


def export_speedup_table(df: pd.DataFrame, output_file: Optional[str] = None) -> pd.DataFrame:
    """
    Create a table comparing with_merge vs no_merge performance.
    
    Args:
        df: Results dataframe
        output_file: Optional file path to save table
        
    Returns:
        Comparison dataframe
    """
    results = []
    
    for seq_len in sorted(df['seq_len'].unique()):
        subset = df[df['seq_len'] == seq_len]
        
        with_merge = subset[subset['mode'] == 'with_merge'].iloc[0] if len(subset[subset['mode'] == 'with_merge']) > 0 else None
        no_merge = subset[subset['mode'] == 'no_merge'].iloc[0] if len(subset[subset['mode'] == 'no_merge']) > 0 else None
        
        if with_merge is not None and no_merge is not None:
            speedup = no_merge['total_time_ms'] / with_merge['total_time_ms']
            results.append({
                'Seq_Len': seq_len,
                'with_merge_ms': round(with_merge['total_time_ms'], 2),
                'no_merge_ms': round(no_merge['total_time_ms'], 2),
                'Speedup_x': round(speedup, 2),
                'with_merge_fps': round(with_merge['throughput_fps'], 2),
                'no_merge_fps': round(no_merge['throughput_fps'], 2),
            })
    
    result_df = pd.DataFrame(results)
    
    if output_file:
        result_df.to_csv(output_file, index=False)
        print(f"\n✓ Comparison table saved to: {output_file}")
    
    return result_df


def plot_latency_curve(df: pd.DataFrame, output_file: Optional[str] = None):
    """
    Plot latency curves comparing with_merge vs no_merge.
    
    Args:
        df: Results dataframe
        output_file: Optional file path to save plot
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("⚠ matplotlib not installed. Skipping plot generation.")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Absolute latency
    for mode in ['with_merge', 'no_merge']:
        subset = df[df['mode'] == mode].sort_values('seq_len')
        axes[0].plot(subset['seq_len'], subset['total_time_ms'], 'o-', label=mode)
    
    axes[0].set_xlabel('Sequence Length (frames)')
    axes[0].set_ylabel('Latency (ms)')
    axes[0].set_title('Model Latency vs Sequence Length')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Throughput
    for mode in ['with_merge', 'no_merge']:
        subset = df[df['mode'] == mode].sort_values('seq_len')
        axes[1].plot(subset['seq_len'], subset['throughput_fps'], 'o-', label=mode)
    
    axes[1].set_xlabel('Sequence Length (frames)')
    axes[1].set_ylabel('Throughput (FPS)')
    axes[1].set_title('Model Throughput vs Sequence Length')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=150, bbox_inches='tight')
        print(f"✓ Plot saved to: {output_file}")
    else:
        plt.show()


def main():
    """Main analysis function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze module latency profiling results")
    parser.add_argument(
        '--csv',
        type=str,
        default='tests_result/module_latency_report.csv',
        help='Path to profiling results CSV'
    )
    parser.add_argument(
        '--comparison-output',
        type=str,
        default='tests_result/speedup_comparison.csv',
        help='Output file for comparison table'
    )
    parser.add_argument(
        '--plot-output',
        type=str,
        default='tests_result/latency_curves.png',
        help='Output file for plot'
    )
    
    args = parser.parse_args()
    
    # Load results
    if not Path(args.csv).exists():
        print(f"✗ CSV file not found: {args.csv}")
        return
    
    df = load_results(args.csv)
    print(f"✓ Loaded {len(df)} results from {args.csv}")
    
    # Print analysis
    print_speedup_summary(df)
    print_bottleneck_analysis(df)
    print_throughput_analysis(df)
    
    # Export comparison table
    export_speedup_table(df, args.comparison_output)
    
    # Generate plots
    plot_latency_curve(df, args.plot_output)


if __name__ == '__main__':
    main()
