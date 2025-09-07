import numpy as np
import torch
import os
import cv2
from pathlib import Path
from vggt.models.vggt import VGGT
from vggt.utils.eval_utils import (
    get_vgg_input_imgs,
    load_images_rgb,
    infer_vggt_and_reconstruct,
    get_sorted_image_paths
)

# force bfloat16
dtype = torch.bfloat16

# load VGGT
model = VGGT(merging=0, vis_attn_map=None)
ckpt = torch.load("./checkpoints/model_tracker_fixed_e20.pt", map_location="cpu")
incompat = model.load_state_dict(ckpt, strict=False)
model = model.cuda().eval()
model = model.to(torch.bfloat16)

# load Images
images_dir = Path('./kitchen/images')
image_paths = get_sorted_image_paths(images_dir)
images = load_images_rgb(image_paths)
print(f"Loaded {len(images)} images with shape: {images[0].shape if images else 'N/A'}")

images_array = np.stack(images)
vgg_input = get_vgg_input_imgs(images_array)


# Inference
(
    extrinsic_np,
    intrinsic_np,
    all_world_points,
    all_cam_to_world_mat,
    inference_time_ms,
) = infer_vggt_and_reconstruct(
    model, vgg_input, dtype, 3.0
)


