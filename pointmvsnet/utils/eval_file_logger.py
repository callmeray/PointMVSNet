import numpy as np
import os.path as osp
import cv2
import scipy

import torch
import torch.nn.functional as F

from pointmvsnet.utils.io import mkdir, write_cam_dtu, write_pfm


def eval_file_logger(data_batch, preds, ref_img_path, folder):
    l = ref_img_path.split("/")
    eval_folder = "/".join(l[:-3])

    scene = l[-2]

    scene_folder = osp.join(eval_folder, folder, scene)

    if not osp.isdir(scene_folder):
        mkdir(scene_folder)
        print("**** {} ****".format(scene))

    out_index = int(l[-1][5:8]) - 1

    cam_params_list = data_batch["cam_params_list"].cpu().numpy()

    ref_cam_paras = cam_params_list[0, 0, :, :, :]

    init_depth_map_path = scene_folder + ('/%08d_init.pfm' % out_index)
    init_prob_map_path = scene_folder + ('/%08d_init_prob.pfm' % out_index)
    out_ref_image_path = scene_folder + ('/%08d.jpg' % out_index)

    init_depth_map = preds["coarse_depth_map"].cpu().numpy()[0, 0]
    init_prob_map = preds["coarse_prob_map"].cpu().numpy()[0, 0]
    ref_image = data_batch["ref_img"][0].cpu().numpy()

    write_pfm(init_depth_map_path, init_depth_map)
    write_pfm(init_prob_map_path, init_prob_map)
    cv2.imwrite(out_ref_image_path, ref_image)

    out_init_cam_path = scene_folder + ('/cam_%08d_init.txt' % out_index)
    init_cam_paras = ref_cam_paras.copy()
    init_cam_paras[1, :2, :3] *= (float(init_depth_map.shape[0]) / ref_image.shape[0])
    write_cam_dtu(out_init_cam_path, init_cam_paras)

    interval_list = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
    interval_list = np.reshape(interval_list, [1, 1, -1])

    for i, k in enumerate(preds.keys()):
        if "flow" in k:
            if "prob" in k:
                out_flow_prob_map = preds[k][0].cpu().permute(1, 2, 0).numpy()
                num_interval = out_flow_prob_map.shape[-1]
                assert num_interval == interval_list.size
                pred_interval = np.sum(out_flow_prob_map * interval_list, axis=-1) + 2.0
                pred_floor = np.floor(pred_interval).astype(np.int)[..., np.newaxis]
                pred_ceil = pred_floor + 1
                pred_ceil = np.clip(pred_ceil, 0, num_interval - 1)
                prob_height, prob_width = pred_floor.shape[:2]
                prob_height_ind = np.tile(np.reshape(np.arange(prob_height), [-1, 1, 1]), [1, prob_width, 1])
                prob_width_ind = np.tile(np.reshape(np.arange(prob_width), [1, -1, 1]), [prob_height, 1, 1])
                floor_prob = np.squeeze(out_flow_prob_map[prob_height_ind, prob_width_ind, pred_floor], -1)
                ceil_prob = np.squeeze(out_flow_prob_map[prob_height_ind, prob_width_ind, pred_ceil], -1)
                flow_prob = floor_prob + ceil_prob
                flow_prob_map_path = scene_folder + "/{:08d}_{}.pfm".format(out_index, k)
                write_pfm(flow_prob_map_path, flow_prob)

            else:
                out_flow_depth_map = preds[k][0, 0].cpu().numpy()
                flow_depth_map_path = scene_folder + "/{:08d}_{}.pfm".format(out_index, k)
                write_pfm(flow_depth_map_path, out_flow_depth_map)
                out_flow_cam_path = scene_folder + "/cam_{:08d}_{}.txt".format(out_index, k)
                flow_cam_paras = ref_cam_paras.copy()
                flow_cam_paras[1, :2, :3] *= (float(out_flow_depth_map.shape[0]) / float(ref_image.shape[0]))
                write_cam_dtu(out_flow_cam_path, flow_cam_paras)

                world_pts = depth2pts_np(out_flow_depth_map, flow_cam_paras[1][:3, :3], flow_cam_paras[0])
                save_points(osp.join(scene_folder, "{:08d}_{}pts.xyz".format(out_index, k)), world_pts)


def depth2pts_np(depth_map, cam_intrinsic, cam_extrinsic):
    feature_grid = get_pixel_grids_np(depth_map.shape[0], depth_map.shape[1])

    uv = np.matmul(np.linalg.inv(cam_intrinsic), feature_grid)
    cam_points = uv * np.reshape(depth_map, (1, -1))

    R = cam_extrinsic[:3, :3]
    t = cam_extrinsic[:3, 3:4]
    R_inv = np.linalg.inv(R)

    world_points = np.matmul(R_inv, cam_points - t).transpose()
    return world_points


def get_pixel_grids_np(height, width):
    x_linspace = np.linspace(0.5, width - 0.5, width)
    y_linspace = np.linspace(0.5, height - 0.5, height)
    x_coordinates, y_coordinates = np.meshgrid(x_linspace, y_linspace)
    x_coordinates = np.reshape(x_coordinates, (1, -1))
    y_coordinates = np.reshape(y_coordinates, (1, -1))
    ones = np.ones_like(x_coordinates).astype(np.float)
    grid = np.concatenate([x_coordinates, y_coordinates, ones], axis=0)

    return grid


def save_points(path, points):
    np.savetxt(path, points, delimiter=' ', fmt='%.4f')
