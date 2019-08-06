import numpy as np
import os.path as osp
import cv2

import torch
import torch.nn.functional as F

from pointmvsnet.utils.io import mkdir
from pointmvsnet.functions.functions import get_pixel_grids


def file_logger(data_batch, preds, step, output_dir, prefix):
    step_dir = osp.join(output_dir, "{}_step{:05d}".format(prefix, step))
    mkdir(step_dir)
    print("start saving files in ", step_dir)

    img_list = data_batch["img_list"]
    batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())

    cam_params_list = data_batch["cam_params_list"]

    for i in range(num_view):
        np.savetxt(osp.join(step_dir, "img{}.txt".format(i)), img_list[0, i, 0].detach().cpu().numpy(), fmt="%.4f")
        np.savetxt(osp.join(step_dir, "cam{}_extrinsic.txt".format(i)), cam_params_list[0, i, 0].detach().cpu().numpy(), fmt="%.4f")
        np.savetxt(osp.join(step_dir, "cam{}_intrinsic.txt".format(i)), cam_params_list[0, i, 1].detach().cpu().numpy(), fmt="%.4f")
    np.savetxt(osp.join(step_dir, "gt_depth_img.txt"), data_batch["gt_depth_img"][0, 0].detach().cpu().numpy(), fmt="%.4f")
    np.savetxt(osp.join(step_dir, "coarse_depth_img.txt"), preds["coarse_depth_map"][0, 0].detach().cpu().numpy(), fmt="%.4f")

    cam_extrinsic = cam_params_list[0, 0, 0, :3, :4].clone()  # (3, 4)

    cam_intrinsic = cam_params_list[0, 0, 1, :3, :3].clone()

    world_points = preds["world_points"]
    world_points = world_points[0].cpu().numpy().transpose()
    save_points(osp.join(step_dir, "world_points.xyz"), world_points)

    prob_map = preds["coarse_prob_map"][0][0].cpu().numpy()

    coarse_points = depth2pts(preds["coarse_depth_map"], prob_map,
                              cam_intrinsic, cam_extrinsic, (img_height, img_width))
    save_points(osp.join(step_dir, "coarse_point.xyz"), coarse_points)

    gt_points = depth2pts(data_batch["gt_depth_img"], prob_map,
                          cam_intrinsic, cam_extrinsic, (img_height, img_width))
    save_points(osp.join(step_dir, "gt_points.xyz"), gt_points)

    if "flow1" in preds.keys():
        flow1_points = depth2pts(preds["flow1"], prob_map,
                                 cam_intrinsic, cam_extrinsic, (img_height, img_width))
        save_points(osp.join(step_dir, "flow1_points.xyz"), flow1_points)

    if "flow2" in preds.keys():
        flow2_points = depth2pts(preds["flow2"], prob_map,
                                 cam_intrinsic, cam_extrinsic, (img_height, img_width))
        save_points(osp.join(step_dir, "flow2_points.xyz"), flow2_points)

    print("saving finished.")


def depth2pts(depth_map, prob_map, cam_intrinsic, cam_extrinsic, img_size):
    feature_map_indices_grid = get_pixel_grids(depth_map.size(2), depth_map.size(3)).to(depth_map.device)  # (3, H*W)

    curr_cam_intrinsic = cam_intrinsic.clone()
    scale = (depth_map.size(2) + 0.0) / (img_size[0] + 0.0) * 4.0
    curr_cam_intrinsic[:2, :3] *= scale

    uv = torch.matmul(torch.inverse(curr_cam_intrinsic), feature_map_indices_grid)
    cam_points = uv * depth_map[0].view(1, -1)

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3].unsqueeze(-1)
    R_inv = torch.inverse(R)

    world_points = torch.matmul(R_inv, cam_points - t).detach().cpu().numpy().transpose()

    curr_prob_map = prob_map.copy()
    if curr_prob_map.shape[0] != depth_map.size(2):
        curr_prob_map = cv2.resize(curr_prob_map, (depth_map.size(3), depth_map.size(2)),
                                   interpolation=cv2.INTER_LANCZOS4)
    curr_prob_map = np.reshape(curr_prob_map, (-1, 1))

    world_points = np.concatenate([world_points, curr_prob_map], axis=1)

    return world_points


def save_points(path, points):
    np.savetxt(path, points, delimiter=' ', fmt='%.4f')
