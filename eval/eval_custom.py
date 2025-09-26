import argparse
from pathlib import Path
import numpy as np
import torch
import os
import sys
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Ensure project root is in sys.path for absolute imports like `vggt.*`
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    load_poses,
    get_vgg_input_imgs,
    get_sorted_image_paths,
    build_frame_selection,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    evaluate_scene_and_save,
)

# Import pose visualization libraries (optional EVO support)
try:
    from evo.core.trajectory import PoseTrajectory3D
    import evo.tools.plot as plot

    EVO_AVAILABLE = True
except ImportError:
    # EVO is optional; we have a matplotlib-based fallback
    EVO_AVAILABLE = False


def visualize_predicted_poses(
    all_cam_to_world_mat, frame_ids, output_scene_dir, scene_name="custom_dataset"
):
    """
    Visualize the predicted camera pose trajectory (no GT comparison required).

    Args:
        all_cam_to_world_mat: List of camera-to-world transform matrices
        frame_ids: List of frame IDs
        output_scene_dir: Output directory
        scene_name: Scene name
    """
    # Provide basic pose visualization even without EVO
    if not EVO_AVAILABLE:
        print("⚠️  EVO not installed; using basic matplotlib visualization")

    try:
        # Convert to numpy array
        poses_est = np.array(all_cam_to_world_mat)

        if len(poses_est) < 2:
            print("⚠️  Not enough poses to generate trajectory plot")
            return

        print(f"🎨 Generating pose trajectory visualization...")

        # Extract translation part
        positions = poses_est[:, :3, 3]  # shape: (N, 3)

        # Create figure - show XZ-plane projection only
        fig, ax = plt.subplots(1, 1, figsize=(10, 8))

        # XZ-plane projection
        ax.plot(
            positions[:, 0],
            positions[:, 2],
            "b-",
            linewidth=2,
            label="Predicted Trajectory",
        )
        ax.scatter(
            positions[0, 0], positions[0, 2], color="green", s=100, label="Start"
        )
        ax.scatter(positions[-1, 0], positions[-1, 2], color="red", s=100, label="End")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_title(f"{scene_name} - XZ-plane projection")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # Save image
        pose_plot_path = output_scene_dir / "predicted_trajectory.png"
        plt.savefig(pose_plot_path, dpi=300, bbox_inches="tight")
        plt.close()

        print(f"📊 Trajectory visualization saved: {pose_plot_path}")

    except Exception as e:
        print(f"⚠️  Failed to generate pose visualization: {e}")
        import traceback

        traceback.print_exc()


def main():
    """
    Evaluation script for a Custom Dataset.
    Supports optional evaluation and custom dataset structure.
    """
    parser = argparse.ArgumentParser(
        description="Run FastVGGT evaluation on a Custom Dataset"
    )

    # Required: dataset path
    parser.add_argument(
        "--data_path",
        type=Path,
        required=True,
        help="Dataset path containing subfolders: color, depth, gt_ply, pose",
    )

    # Optional: enable evaluation
    parser.add_argument(
        "--enable_evaluation",
        action="store_true",
        help="Enable evaluation (requires pose and ply data)",
    )

    # Output path
    parser.add_argument(
        "--output_path",
        type=Path,
        default="./eval_results_custom",
        help="Output path for evaluation results",
    )

    # Model parameters
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="/home/sy/code/FastVGGT/ckpt/model_tracker_fixed_e20.pt",
        help="Model checkpoint file path",
    )

    parser.add_argument("--merging", type=int, default=0, help="Merging parameter")

    # Processing parameters
    parser.add_argument(
        "--input_frame",
        type=int,
        default=200,
        help="Maximum number of frames to process per scene",
    )

    parser.add_argument(
        "--depth_conf_thresh",
        type=float,
        default=3.0,
        help="Depth confidence threshold to filter low-confidence depth values",
    )

    # Evaluation parameters (only used when evaluation is enabled)
    parser.add_argument(
        "--chamfer_max_dist",
        type=float,
        default=0.5,
        help="Maximum distance threshold used in Chamfer Distance computation",
    )

    parser.add_argument("--plot", action="store_true", help="Whether to generate plots")

    parser.add_argument(
        "--vis_attn_map",
        action="store_true",
        help="Visualize attention maps during inference",
    )

    args = parser.parse_args()
    torch.manual_seed(33)

    # Check data path exists
    if not args.data_path.exists():
        print(f"❌ Error: Data path does not exist: {args.data_path}")
        return

    # Check required subdirectories
    color_dir = args.data_path / "images"
    pose_dir = args.data_path / "pose"

    if not color_dir.exists():
        print(f"❌ Error: color directory does not exist: {color_dir}")
        return

    print(f"📁 Dataset path: {args.data_path}")
    # print(f"🔧 Enable evaluation: {'Yes' if args.enable_evaluation else 'No'}")

    # If evaluation is enabled, check pose and gt_ply directories
    if args.enable_evaluation:
        if not pose_dir.exists():
            print(f"❌ Error: Evaluation requires pose directory: {pose_dir}")
            return

        gt_ply_dir = args.data_path / "gt_ply"
        if not gt_ply_dir.exists():
            print(f"❌ Error: Evaluation requires gt_ply directory: {gt_ply_dir}")
            return
        print(f"📊 Evaluation will use Ground Truth")
    else:
        print(f"🏃 Inference only, no evaluation")

    # Create output directory
    args.output_path.mkdir(parents=True, exist_ok=True)
    output_scene_dir = args.output_path / "custom_dataset"

    # Check if already processed
    if (output_scene_dir / "metrics.json").exists() and args.enable_evaluation:
        print(
            f"⚠️  Results already exist, skipping: {output_scene_dir / 'metrics.json'}"
        )
        return

    # Force use of bf16 dtype
    # dtype = torch.bfloat16
    dtype = torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load VGGT model
    print(f"🔄 Loading model: {args.ckpt_path}")
    model = VGGT(merging=args.merging, vis_attn_map=args.vis_attn_map, dtype=dtype, device=device)
    ckpt = torch.load(args.ckpt_path, map_location="cpu")
    incompat = model.load_state_dict(ckpt, strict=False)
    # if incompat.missing_keys or incompat.unexpected_keys:
    #     print(f"⚠️  Partially incompatible keys when loading model: {incompat}")
    # model = model.cuda().eval()
    model = model.eval()
    model = model.to(dtype)
    print(f"✅ Model loaded")

    # Load scene data
    image_paths = get_sorted_image_paths(color_dir)
    if len(image_paths) == 0:
        print(f"❌ Error: No images found in {color_dir}")
        return

    print(f"🖼️  Found {len(image_paths)} images")

    # Process pose data (if evaluation is enabled)
    poses_gt = None
    first_gt_pose = None
    available_pose_frame_ids = None
    c2ws = None

    if args.enable_evaluation:
        poses_gt, first_gt_pose, available_pose_frame_ids = load_poses(pose_dir)
        if (
            poses_gt is None
            or first_gt_pose is None
            or available_pose_frame_ids is None
        ):
            print(f"❌ Error: Failed to load pose data")
            return
        print(f"📐 Loaded {len(poses_gt)} poses")

    # Frame selection
    if args.enable_evaluation and available_pose_frame_ids is not None:
        # Use pose data for frame selection
        selected_frame_ids, selected_image_paths, selected_pose_indices = (
            build_frame_selection(
                image_paths, available_pose_frame_ids, args.input_frame
            )
        )
        c2ws = poses_gt[selected_pose_indices]
        image_paths = selected_image_paths
    else:
        # Simply take the first N frames
        num_frames = min(len(image_paths), args.input_frame)
        selected_frame_ids = list(range(num_frames))
        image_paths = image_paths[:num_frames]

    print(f"📋 Selected {len(image_paths)} frames for processing")

    try:
        # Load images
        print(f"🔄 Loading images...")
        images = load_images_rgb(image_paths)

        if not images or len(images) < 3:
            print(f"❌ Error: Not enough valid images (need at least 3)")
            return

        frame_ids = selected_frame_ids
        images_array = np.stack(images)
        vgg_input, patch_width, patch_height = get_vgg_input_imgs(images_array)
        print(f"📐 Image patch dimensions: {patch_width}x{patch_height}")

        # Update attention layer patch dimensions in the model
        model.update_patch_dimensions(patch_width, patch_height)

        # Inference + Reconstruction
        print(f"🚀 Start inference and reconstruction...")
        (
            extrinsic_np,
            intrinsic_np,
            all_world_points,
            all_point_colors,
            all_cam_to_world_mat,
            inference_time_ms,
        ) = infer_vggt_and_reconstruct(
            model, vgg_input, dtype, args.depth_conf_thresh, image_paths
        )
        print(f"⏱️  Inference time: {inference_time_ms:.2f}ms")

        # Check results
        if not all_cam_to_world_mat or not all_world_points:
            print(f"❌ Error: Failed to obtain valid camera poses or point clouds")
            return

        # print(f"✅ Inference done, obtained {len(all_world_points)} point sets")

        # Evaluation and saving
        if args.enable_evaluation:
            print(f"📊 Start evaluation...")
            gt_ply_dir = args.data_path / "gt_ply"
            metrics = evaluate_scene_and_save(
                "custom_dataset",
                c2ws,
                first_gt_pose,
                frame_ids,
                all_cam_to_world_mat,
                all_world_points,
                output_scene_dir,
                gt_ply_dir,
                args.chamfer_max_dist,
                inference_time_ms,
                args.plot,
            )
            if metrics is not None:
                print("📈 Evaluation results:")
                for key, value in metrics.items():
                    if key in [
                        "chamfer_distance",
                        "ate",
                        "are",
                        "rpe_rot",
                        "rpe_trans",
                        "inference_time_ms",
                    ]:
                        print(f"  {key}: {float(value):.4f}")

            # Also visualize predicted poses in evaluation branch
            if args.plot:
                visualize_predicted_poses(
                    all_cam_to_world_mat, frame_ids, output_scene_dir, "custom_dataset"
                )
        else:
            # Save reconstruction only, no evaluation
            print(f"💾 Saving reconstruction...")
            output_scene_dir.mkdir(parents=True, exist_ok=True)

            # Save camera poses
            poses_output_path = output_scene_dir / "estimated_poses.txt"
            with open(poses_output_path, "w") as f:
                for i, pose in enumerate(all_cam_to_world_mat):
                    f.write(f"# Frame {frame_ids[i]}\n")
                    for row in pose:
                        f.write(" ".join(map(str, row)) + "\n")
                    f.write("\n")

            # Save point cloud
            if all_world_points:
                points_output_path = output_scene_dir / "reconstructed_points.ply"

                # Merge all frames' point clouds and colors
                try:
                    merged_point_cloud = np.vstack(all_world_points)
                    merged_colors = (
                        np.vstack(all_point_colors).astype(np.uint8)
                        if all_point_colors is not None and len(all_point_colors) > 0
                        else None
                    )
                    print(
                        f"📊 Merged point clouds: {len(all_world_points)} frames, total {len(merged_point_cloud)} points"
                    )

                    # If too many points, randomly sample 100000 points
                    max_points = 100000
                    if len(merged_point_cloud) > max_points:
                        print(
                            f"🔽 Too many points, randomly sampling {max_points} points..."
                        )
                        # Randomly choose indices
                        indices = np.random.choice(
                            len(merged_point_cloud), size=max_points, replace=False
                        )
                        merged_point_cloud = merged_point_cloud[indices]
                        if merged_colors is not None:
                            merged_colors = merged_colors[indices]
                        print(
                            f"✅ Sampling done, kept {len(merged_point_cloud)} points"
                        )

                    # Save as PLY (with color)
                    with open(points_output_path, "w") as f:
                        f.write("ply\n")
                        f.write("format ascii 1.0\n")
                        f.write(f"element vertex {len(merged_point_cloud)}\n")
                        f.write("property float x\n")
                        f.write("property float y\n")
                        f.write("property float z\n")
                        if merged_colors is not None:
                            f.write("property uchar red\n")
                            f.write("property uchar green\n")
                            f.write("property uchar blue\n")
                        f.write("end_header\n")
                        if merged_colors is None:
                            for point in merged_point_cloud:
                                if not (np.isnan(point).any() or np.isinf(point).any()):
                                    f.write(
                                        f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n"
                                    )
                        else:
                            for point, color in zip(merged_point_cloud, merged_colors):
                                # Check point validity
                                if not (np.isnan(point).any() or np.isinf(point).any()):
                                    r = int(np.clip(color[0], 0, 255))
                                    g = int(np.clip(color[1], 0, 255))
                                    b = int(np.clip(color[2], 0, 255))
                                    f.write(
                                        f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} {r} {g} {b}\n"
                                    )

                    print(f"💾 Point cloud saved to: {points_output_path}")

                except Exception as e:
                    print(f"⚠️  Error saving point cloud: {e}")
                    # If merge fails, try to log per-frame info
                    print(f"🔍 Point cloud debug info:")
                    for i, frame_points in enumerate(all_world_points):
                        print(
                            f"  Frame {i}: {frame_points.shape if hasattr(frame_points, 'shape') else type(frame_points)}"
                        )
                        if (
                            hasattr(frame_points, "shape")
                            and len(frame_points.shape) >= 2
                        ):
                            print(
                                f"    Shape: {frame_points.shape}, Dtype: {frame_points.dtype}"
                            )
                            if frame_points.shape[0] > 0:
                                print(
                                    f"    Range: x[{np.min(frame_points[:, 0]):.3f}, {np.max(frame_points[:, 0]):.3f}] "
                                    f"y[{np.min(frame_points[:, 1]):.3f}, {np.max(frame_points[:, 1]):.3f}] "
                                    f"z[{np.min(frame_points[:, 2]):.3f}, {np.max(frame_points[:, 2]):.3f}]"
                                )

            print(f"📁 Results saved to: {output_scene_dir}")

            # Visualize predicted pose trajectory
            if args.plot:
                visualize_predicted_poses(
                    all_cam_to_world_mat, frame_ids, output_scene_dir, "custom_dataset"
                )

        print(f"🎉 Done!")

    except Exception as e:
        print(f"❌ Error occurred during processing: {e}")
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    main()
