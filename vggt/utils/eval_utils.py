import io
import time
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from copy import deepcopy
import evo.main_ape as main_ape
import evo.main_rpe as main_rpe
import matplotlib.pyplot as plt
import numpy as np
import torch
from evo.core.metrics import PoseRelation, Unit
from evo.core.trajectory import PoseTrajectory3D
from scipy.linalg import svd
import open3d as o3d  # for point cloud processing and Chamfer Distance computation
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms as TF
import evo.tools.plot as plot
import cv2
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


# Import umeyama_alignment for internal use in eval_trajectory
def umeyama_alignment(src, dst, estimate_scale=True):
    # Ensure inputs have correct shape
    assert (
        src.shape == dst.shape
    ), f"Input shapes don't match: src {src.shape}, dst {dst.shape}"
    assert src.shape[0] == 3, f"Expected point cloud dimension (3,N), got {src.shape}"

    # Compute centroids
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)

    # Center the point clouds
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    cov = dst_centered @ src_centered.T

    try:
        # Singular Value Decomposition
        U, D, Vt = svd(cov)
        V = Vt.T

        # Handle reflection case
        det_UV = np.linalg.det(U @ V.T)
        S = np.eye(3)
        if det_UV < 0:
            S[2, 2] = -1

        # Compute rotation matrix
        R = U @ S @ V.T

        if estimate_scale:
            # Compute scale factor - fix dimension issue
            src_var = np.sum(src_centered * src_centered)
            if src_var < 1e-10:
                print(
                    "Warning: Source point cloud variance close to zero, setting scale factor to 1.0"
                )
                scale = 1.0
            else:
                # Fix potential dimension issue with np.diag(S)
                # Use diagonal elements directly
                scale = np.sum(D * np.diag(S)) / src_var
        else:
            scale = 1.0

        # Compute translation vector
        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()

        return scale, R, t

    except Exception as e:
        print(f"Error in umeyama_alignment computation: {e}")
        print(
            "Returning default transformation: scale=1.0, rotation=identity matrix, translation=centroid difference"
        )
        # Return default transformation
        scale = 1.0
        R = np.eye(3)
        t = (dst_mean - src_mean).ravel()
        return scale, R, t


def compute_chamfer_distance(points_pred, points_gt, max_dist=1.0):
    # Ensure point cloud size is not too large, which would cause slow computation
    MAX_POINTS = 100000
    if points_pred.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(points_pred.shape[0], MAX_POINTS, replace=False)
        points_pred = points_pred[indices]

    if points_gt.shape[0] > MAX_POINTS:
        np.random.seed(33)  # Fix random seed
        indices = np.random.choice(points_gt.shape[0], MAX_POINTS, replace=False)
        points_gt = points_gt[indices]

    # Convert numpy point clouds to open3d point cloud objects
    pcd_pred = o3d.geometry.PointCloud()
    pcd_gt = o3d.geometry.PointCloud()
    pcd_pred.points = o3d.utility.Vector3dVector(points_pred)
    pcd_gt.points = o3d.utility.Vector3dVector(points_gt)

    # Downsample point clouds to accelerate computation
    voxel_size = 0.05  # 5cm voxel size
    pcd_pred = pcd_pred.voxel_down_sample(voxel_size)
    pcd_gt = pcd_gt.voxel_down_sample(voxel_size)

    # Compute distances from predicted point cloud to GT point cloud
    distances1 = np.asarray(pcd_pred.compute_point_cloud_distance(pcd_gt))
    # Compute distances from GT point cloud to predicted point cloud
    distances2 = np.asarray(pcd_gt.compute_point_cloud_distance(pcd_pred))

    # Apply distance clipping
    distances1 = np.clip(distances1, 0, max_dist)
    distances2 = np.clip(distances2, 0, max_dist)

    # Chamfer Distance is the sum of mean distances in both directions
    chamfer_dist = np.mean(distances1) + np.mean(distances2)

    return chamfer_dist


def load_gt_pointcloud(scene_id, gt_ply_dir):
    scene_dir = gt_ply_dir / scene_id
    ply_path = scene_dir / (scene_id + "_vh_clean_2.ply")
    pcd = o3d.io.read_point_cloud(str(ply_path))

    # Convert to numpy arrays
    points = np.asarray(pcd.points)
    colors = None
    try:
        if pcd.has_colors():
            colors = np.asarray(pcd.colors)
    except Exception:
        colors = None

    return points, colors


def eval_trajectory(poses_est, poses_gt, frame_ids, align=False):
    # Build reference trajectory object
    traj_ref = PoseTrajectory3D(
        positions_xyz=poses_gt[:, :3, 3],  # Extract translation part
        orientations_quat_wxyz=Rotation.from_matrix(poses_gt[:, :3, :3]).as_quat(
            scalar_first=True
        ),  # Convert rotation matrix to quaternion
        timestamps=frame_ids,
    )

    # Build estimated trajectory object
    traj_est = PoseTrajectory3D(
        positions_xyz=poses_est[:, :3, 3],
        orientations_quat_wxyz=Rotation.from_matrix(poses_est[:, :3, :3]).as_quat(
            scalar_first=True
        ),
        timestamps=frame_ids,
    )

    # Calculate Absolute Trajectory Error (ATE)
    ate_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        align_origin=True,
    )
    ate = ate_result.stats["rmse"]

    # Get alignment transformation matrix
    transform = np.eye(4)
    if align:
        try:
            # Use umeyama algorithm to compute optimal rigid transformation (including rotation, translation and scaling)
            aligned_xyz = ate_result.trajectories["traj"].positions_xyz
            original_xyz = traj_est.positions_xyz

            # At least 3 points needed to compute reliable transformation
            if len(aligned_xyz) >= 3 and len(original_xyz) >= 3:
                # Ensure point count matches
                min_points = min(len(aligned_xyz), len(original_xyz))
                aligned_xyz = aligned_xyz[:min_points]
                original_xyz = original_xyz[:min_points]

                # Compute transformation matrix (scaling, rotation and translation)
                try:
                    s, R, t = umeyama_alignment(
                        original_xyz.T,  # Source point cloud (3xN)
                        aligned_xyz.T,  # Target point cloud (3xN)
                        True,  # Whether to estimate scaling
                    )

                    # Build complete transformation matrix
                    transform = np.eye(4)
                    transform[:3, :3] = s * R  # Scaling and rotation
                    transform[:3, 3] = t  # Translation

                except Exception as e:
                    print(f"umeyama_alignment failed: {e}")
            else:
                print(
                    "Insufficient points, cannot reliably compute transformation matrix"
                )
        except Exception as e:
            print(f"Error computing transformation matrix: {e}")

        # If the above method fails, fallback to simple translation transformation
        if np.array_equal(transform, np.eye(4)) and hasattr(ate_result, "trajectories"):
            try:
                # Get original and aligned first position
                orig_pos = traj_est.positions_xyz[0]
                aligned_pos = ate_result.trajectories["traj"].positions_xyz[0]

                # Calculate translation part
                translation = aligned_pos - orig_pos

                # Update translation part of transformation matrix
                transform[:3, 3] = translation
                print(f"Fallback to simple translation transformation: {transform}")
            except Exception as e:
                print(f"Error building translation transformation: {e}")
                print("Will use identity matrix")

    # Calculate Absolute Rotation Error (ARE)
    are_result = main_ape.ape(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        align_origin=True,
    )
    are = are_result.stats["rmse"]

    # Calculate Relative Pose Error (RPE) - rotation part
    rpe_rots_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.rotation_angle_deg,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_rot = rpe_rots_result.stats["rmse"]

    # Calculate Relative Pose Error (RPE) - translation part
    rpe_transs_result = main_rpe.rpe(
        deepcopy(traj_ref),
        deepcopy(traj_est),
        est_name="traj",
        pose_relation=PoseRelation.translation_part,
        align=align,
        correct_scale=align,
        delta=1,
        delta_unit=Unit.frames,
        rel_delta_tol=0.01,
        all_pairs=True,
        align_origin=True,
    )
    rpe_trans = rpe_transs_result.stats["rmse"]

    # Plot trajectory graph
    plot_mode = plot.PlotMode.xz  # Use correct PlotMode reference
    fig = plt.figure()
    ax = plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE: {round(ate, 3)}, ARE: {round(are, 3)}")

    # Use reference trajectory (GT) for plotting
    plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")

    # Use aligned trajectory for visualization
    if align:
        traj_est_aligned = ate_result.trajectories["traj"]
        plot.traj_colormap(
            ax,
            traj_est_aligned,
            ate_result.np_arrays["error_array"],
            plot_mode,
            min_map=ate_result.stats["min"],
            max_map=ate_result.stats["max"],
        )
    else:
        plot.traj_colormap(
            ax,
            traj_est,
            ate_result.np_arrays["error_array"],
            plot_mode,
            min_map=ate_result.stats["min"],
            max_map=ate_result.stats["max"],
        )

    ax.legend()

    # Save image to memory buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format="png", dpi=90)
    buffer.seek(0)

    pillow_image = Image.open(buffer)
    pillow_image.load()
    buffer.close()
    plt.close(fig)

    return (
        {"ate": ate, "are": are, "rpe_rot": rpe_rot, "rpe_trans": rpe_trans},
        pillow_image,
        transform,
    )


def load_poses(path):
    # Read all txt files from pose directory
    pose_files = sorted(
        path.glob("*.txt"), key=lambda x: int(x.stem)
    )  # Sort by numerical order

    # Check if pose files exist
    if len(pose_files) == 0:
        print(f"Warning: No pose files (.txt) found in directory {path}")
        return None, None, None

    c2ws = []
    available_frame_ids = []

    for pose_file in pose_files:
        try:
            with open(pose_file, "r") as f:
                # Each file contains 16 numbers representing a 4x4 transformation matrix
                nums = [float(x) for x in f.read().strip().split()]
                pose = np.array(nums).reshape(4, 4)
                # Check if pose is valid (no infinite or NaN values)
                if not (np.isinf(pose).any() or np.isnan(pose).any()):
                    c2ws.append(pose)
                    available_frame_ids.append(int(pose_file.stem))
                else:
                    continue
        except Exception as e:
            print(f"Error reading pose file {pose_file}: {e}")
            continue

    if len(c2ws) == 0:
        print(f"Warning: No valid pose files found in directory {path}")
        return None, None, None

    c2ws = np.stack(c2ws)
    available_frame_ids = np.array(available_frame_ids)

    # Transform all poses to first frame coordinate system
    first_gt_pose = c2ws[0].copy()  # Save original pose of first frame
    c2ws = np.linalg.inv(c2ws[0]) @ c2ws
    return c2ws, first_gt_pose, available_frame_ids


def get_vgg_input_imgs(images: np.ndarray):
    to_tensor = TF.ToTensor()
    vgg_input_images = []
    for image in images:
        img = Image.fromarray(image, mode="RGB")
        width, height = img.size
        # Resize image, maintain aspect ratio, ensure height is multiple of 14
        new_width = 532
        new_height = round(height * (new_width / width) / 14) * 14
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # If height exceeds 518, perform center cropping
        if new_height > new_width:
            start_y = (new_height - new_width) // 2
            img = img[:, start_y : start_y + new_width, :]

        vgg_input_images.append(img)
    vgg_input_images = torch.stack(vgg_input_images)
    return vgg_input_images


def get_sorted_image_paths(images_dir):
    image_paths = []
    for ext in ["*.jpg", "*.png", "*.jpeg"]:
        image_paths.extend(sorted(images_dir.glob(ext)))
    image_paths.sort(key=lambda x: int(x.stem))
    return image_paths


def to_homogeneous(extrinsics):
    n = extrinsics.shape[0]
    homogeneous_extrinsics = np.eye(4)[None, :, :].repeat(
        n, axis=0
    )  # Create identity matrix
    homogeneous_extrinsics[:, :3, :4] = extrinsics  # Copy [R|t] part
    return homogeneous_extrinsics


def umeyama_alignment(src, dst, estimate_scale=True):
    # Ensure inputs have correct shape
    assert (
        src.shape == dst.shape
    ), f"Input shapes don't match: src {src.shape}, dst {dst.shape}"
    assert src.shape[0] == 3, f"Expected point cloud dimension (3,N), got {src.shape}"

    # Compute centroids
    src_mean = src.mean(axis=1, keepdims=True)
    dst_mean = dst.mean(axis=1, keepdims=True)

    # Center the point clouds
    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    # Compute covariance matrix
    cov = dst_centered @ src_centered.T

    try:
        # Singular Value Decomposition
        U, D, Vt = svd(cov)
        V = Vt.T

        # Handle reflection case
        det_UV = np.linalg.det(U @ V.T)
        S = np.eye(3)
        if det_UV < 0:
            S[2, 2] = -1

        # Compute rotation matrix
        R = U @ S @ V.T

        if estimate_scale:
            # Compute scale factor - fix dimension issue
            src_var = np.sum(src_centered * src_centered)
            if src_var < 1e-10:
                print(
                    "Warning: Source point cloud variance close to zero, setting scale factor to 1.0"
                )
                scale = 1.0
            else:
                # Fix potential dimension issue with np.diag(S)
                # Use diagonal elements directly
                scale = np.sum(D * np.diag(S)) / src_var
        else:
            scale = 1.0

        # Compute translation vector
        t = dst_mean.ravel() - scale * (R @ src_mean).ravel()

        return scale, R, t

    except Exception as e:
        print(f"Error in umeyama_alignment computation: {e}")
        print(
            "Returning default transformation: scale=1.0, rotation=identity matrix, translation=centroid difference"
        )
        # Return default transformation
        scale = 1.0
        R = np.eye(3)
        t = (dst_mean - src_mean).ravel()
        return scale, R, t


def align_point_clouds_scale(source_pc, target_pc):
    # Compute bounding box sizes of point clouds
    source_min = np.min(source_pc, axis=0)
    source_max = np.max(source_pc, axis=0)
    target_min = np.min(target_pc, axis=0)
    target_max = np.max(target_pc, axis=0)

    source_size = source_max - source_min
    target_size = target_max - target_min

    # Compute point cloud centers
    source_center = (source_max + source_min) / 2
    target_center = (target_max + target_min) / 2

    # Compute overall scale factor (using diagonal length)
    source_diag = np.sqrt(np.sum(source_size**2))
    target_diag = np.sqrt(np.sum(target_size**2))

    if source_diag < 1e-8:
        print("Warning: Source point cloud size close to zero")
        scale_factor = 1.0
    else:
        scale_factor = target_diag / source_diag

    # Apply scaling (with source point cloud center as reference)
    centered_source = source_pc - source_center
    scaled_centered = centered_source * scale_factor
    scaled_aligned_source = scaled_centered + target_center

    return scaled_aligned_source, scale_factor


def get_all_scenes(data_dir: Path, num_scenes: int) -> List[str]:
    all_scenes = sorted([d.name for d in data_dir.iterdir() if d.is_dir()])
    if len(all_scenes) > num_scenes:
        sample_interval = max(1, len(all_scenes) // num_scenes)
        return all_scenes[::sample_interval][:num_scenes]
    return all_scenes


def build_frame_selection(
    image_paths: List[Path],
    available_pose_frame_ids: np.ndarray,
    input_frame: int,
) -> Tuple[List[int], List[Path], List[int]]:
    all_image_frame_ids = [int(path.stem) for path in image_paths]
    valid_frame_ids = sorted(
        list(set(all_image_frame_ids) & set(available_pose_frame_ids))
    )
    if len(valid_frame_ids) > input_frame:
        first_frame = valid_frame_ids[0]
        remaining_frames = valid_frame_ids[1:]
        step = max(1, len(remaining_frames) // (input_frame - 1))
        selected_remaining = remaining_frames[::step][: input_frame - 1]
        selected_frame_ids = [first_frame] + selected_remaining
    else:
        selected_frame_ids = valid_frame_ids

    frame_id_to_path = {int(path.stem): path for path in image_paths}
    selected_image_paths = [
        frame_id_to_path[fid] for fid in selected_frame_ids if fid in frame_id_to_path
    ]

    pose_frame_to_idx = {fid: idx for idx, fid in enumerate(available_pose_frame_ids)}
    selected_pose_indices = [
        pose_frame_to_idx[fid] for fid in selected_frame_ids if fid in pose_frame_to_idx
    ]

    return selected_frame_ids, selected_image_paths, selected_pose_indices


def load_images_rgb(image_paths: List[Path]) -> List[np.ndarray]:
    images: List[np.ndarray] = []
    for image_path in image_paths:
        img = cv2.imread(str(image_path))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
    return images


@torch.no_grad()
def infer_vggt_and_reconstruct(
    model: torch.nn.Module,
    vgg_input: torch.Tensor,
    dtype: torch.dtype,
    depth_conf_thresh: float,
    image_paths: list = None,
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], float]:
    torch.cuda.synchronize()
    start = time.time()
    with torch.cuda.amp.autocast(dtype=dtype):
        vgg_input_cuda = vgg_input.cuda().to(torch.bfloat16)
        predictions = model(vgg_input_cuda, image_paths=image_paths)
    torch.cuda.synchronize()
    end = time.time()
    inference_time_ms = (end - start) * 1000.0

    extrinsic, intrinsic = pose_encoding_to_extri_intri(
        predictions["pose_enc"], (vgg_input.shape[2], vgg_input.shape[3])
    )

    depth_tensor = predictions["depth"]
    depth_conf = predictions["depth_conf"]
    depth_conf_np = depth_conf[0].detach().float().cpu().numpy()
    depth_mask = depth_conf_np >= depth_conf_thresh
    depth_filtered = depth_tensor[0].detach().float().cpu().numpy()
    depth_filtered[~depth_mask] = np.nan
    depth_np = depth_filtered

    extrinsic_np = extrinsic[0].detach().float().cpu().numpy()
    intrinsic_np = intrinsic[0].detach().float().cpu().numpy()

    world_points = unproject_depth_map_to_point_map(
        depth_np, extrinsic_np, intrinsic_np
    )
    all_points: List[np.ndarray] = []
    for frame_idx in range(world_points.shape[0]):
        points = world_points[frame_idx].reshape(-1, 3)
        valid_mask = ~np.isnan(points).any(axis=1) & ~np.isinf(points).any(axis=1)
        valid_points = points[valid_mask]
        if len(valid_points) > 0:
            all_points.append(valid_points)

    camera_poses = to_homogeneous(extrinsic_np)
    all_cam_to_world_mat = list(camera_poses)

    return (
        extrinsic_np,
        intrinsic_np,
        all_points,
        all_cam_to_world_mat,
        inference_time_ms,
    )


def evaluate_scene_and_save(
    scene: str,
    c2ws: np.ndarray,
    first_gt_pose: np.ndarray,
    frame_ids: List[int],
    all_cam_to_world_mat: List[np.ndarray],
    all_world_points: List[np.ndarray],
    output_scene_dir: Path,
    gt_ply_dir: Path,
    chamfer_max_dist: float,
    inference_time_ms: float,
    plot_flag: bool,
) -> Optional[Dict[str, float]]:
    if not all_cam_to_world_mat or not all_world_points:
        print(f"Skipping {scene}: failed to obtain valid camera poses or point clouds")
        return None

    output_scene_dir.mkdir(parents=True, exist_ok=True)

    poses_gt = c2ws
    w2cs = np.linalg.inv(poses_gt)
    traj_est_poses = np.array(all_cam_to_world_mat)
    n = min(len(traj_est_poses), len(w2cs))
    timestamps = frame_ids[:n]
    stats_aligned, traj_plot, _ = eval_trajectory(
        traj_est_poses[:n], w2cs[:n], timestamps, align=True
    )

    try:
        merged_point_cloud = np.vstack(all_world_points)
        gt_point_cloud, _ = load_gt_pointcloud(scene, gt_ply_dir)
        if gt_point_cloud is not None:
            homogeneous_points = np.hstack(
                [merged_point_cloud, np.ones((merged_point_cloud.shape[0], 1))]
            )
            world_points_raw = np.dot(homogeneous_points, first_gt_pose.T)[:, :3]

            world_points_scaled, scale_factor = align_point_clouds_scale(
                world_points_raw, gt_point_cloud
            )

            cd_value = compute_chamfer_distance(
                world_points_scaled, gt_point_cloud, max_dist=chamfer_max_dist
            )
            stats_aligned["chamfer_distance"] = float(cd_value)
            stats_aligned["scale_factor"] = float(scale_factor)
    except Exception as e:
        print(f"Error computing Chamfer Distance for {scene}: {e}")

    all_metrics: Dict[str, float] = deepcopy(stats_aligned)
    for metric_name, metric_value in list(stats_aligned.items()):
        all_metrics[f"aligned_{metric_name}"] = metric_value
    all_metrics["inference_time_ms"] = float(inference_time_ms)

    with open(output_scene_dir / "metrics.json", "w") as f:
        import json

        json.dump(all_metrics, f, indent=4)
    if plot_flag:
        try:
            traj_plot.save(output_scene_dir / "plot.png")
        except Exception:
            pass

    return all_metrics


def compute_average_metrics_and_save(
    all_scenes_metrics: Dict[str, Dict[str, Dict[str, float]]],
    output_path: Path,
    input_frame: int,
) -> Dict[str, float]:
    metric_names = [
        "chamfer_distance",
        "ate",
        "are",
        "rpe_rot",
        "rpe_trans",
        "inference_time_ms",
    ]
    average_metrics_list: Dict[str, List[float]] = {
        metric: [] for metric in metric_names
    }
    for _, metrics in all_scenes_metrics["scenes"].items():
        for metric_name, metric_value in metrics.items():
            if metric_name in average_metrics_list:
                average_metrics_list[metric_name].append(float(metric_value))

    average_metrics: Dict[str, float] = {}
    for metric_name, values in average_metrics_list.items():
        average_metrics[metric_name] = float(np.mean(values)) if values else 0.0

    all_scenes_metrics["average"] = average_metrics
    output_path.mkdir(parents=True, exist_ok=True)

    input_frame_dir = output_path / f"input_frame_{input_frame}"
    input_frame_dir.mkdir(parents=True, exist_ok=True)

    with open(input_frame_dir / "all_scenes_metrics.json", "w") as f:
        import json

        json.dump(all_scenes_metrics, f, indent=4)

    with open(input_frame_dir / "average_metrics.json", "w") as f:
        import json

        json.dump(average_metrics, f, indent=4)

    print("\nAverage metrics:")
    for metric_name, value in average_metrics.items():
        print(f"{metric_name}: {value:.6f}")

    return average_metrics
