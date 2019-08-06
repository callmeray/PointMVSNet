#!/usr/bin/env python
"""
Copyright 2018, Yao Yao, HKUST.
Edited by Rui Chen.
convert Point-MVSNet output to Gipuma format for post-processing.
"""

from __future__ import print_function

import argparse
import os.path as osp
from struct import *

from pointmvsnet.utils.io import *


def mkdir(path):
    import errno
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_gipuma_dmb(path):
    '''read Gipuma .dmb format image'''

    with open(path, "rb") as fid:
        image_type = unpack('<i', fid.read(4))[0]
        height = unpack('<i', fid.read(4))[0]
        width = unpack('<i', fid.read(4))[0]
        channel = unpack('<i', fid.read(4))[0]

        array = np.fromfile(fid, np.float32)
    array = array.reshape((width, height, channel), order="F")
    return np.transpose(array, (1, 0, 2)).squeeze()


def write_gipuma_dmb(path, image):
    '''write Gipuma .dmb format image'''

    image_shape = np.shape(image)
    width = image_shape[1]
    height = image_shape[0]
    if len(image_shape) == 3:
        channels = image_shape[2]
    else:
        channels = 1

    if len(image_shape) == 3:
        image = np.transpose(image, (2, 0, 1)).squeeze()

    with open(path, "wb") as fid:
        # fid.write(pack(1))
        fid.write(pack('<i', 1))
        fid.write(pack('<i', height))
        fid.write(pack('<i', width))
        fid.write(pack('<i', channels))
        image.tofile(fid)
    return


def mvsnet_to_gipuma_dmb(in_path, out_path):
    '''convert mvsnet .pfm output to Gipuma .dmb format'''

    image = load_pfm(in_path)[0]
    write_gipuma_dmb(out_path, image)

    return


def mvsnet_to_gipuma_cam(in_path, out_path):
    '''convert mvsnet camera to gipuma camera format'''

    cam = load_cam_dtu(open(in_path))

    extrinsic = cam[0:4][0:4][0]
    intrinsic = cam[0:4][0:4][1]
    intrinsic[3][0] = 0
    intrinsic[3][1] = 0
    intrinsic[3][2] = 0
    intrinsic[3][3] = 0
    projection_matrix = np.matmul(intrinsic, extrinsic)
    projection_matrix = projection_matrix[0:3][:]

    f = open(out_path, "w")
    for i in range(0, 3):
        for j in range(0, 4):
            f.write(str(projection_matrix[i][j]) + ' ')
        f.write('\n')
    f.write('\n')
    f.close()

    return


def fake_colmap_normal(in_depth_path, out_normal_path):
    depth_image = read_gipuma_dmb(in_depth_path)
    image_shape = np.shape(depth_image)

    normal_image = np.ones_like(depth_image)
    normal_image = np.reshape(normal_image, (image_shape[0], image_shape[1], 1))
    normal_image = np.tile(normal_image, [1, 1, 3])
    normal_image = normal_image / 1.732050808

    mask_image = np.squeeze(np.where(depth_image > 0, 1, 0))
    mask_image = np.reshape(mask_image, (image_shape[0], image_shape[1], 1))
    mask_image = np.tile(mask_image, [1, 1, 3])
    mask_image = np.float32(mask_image)

    normal_image = np.multiply(normal_image, mask_image)
    normal_image = np.float32(normal_image)

    write_gipuma_dmb(out_normal_path, normal_image)
    return


def mvsnet_to_gipuma(scene_folder, gipuma_point_folder, name, view_num):
    gipuma_cam_folder = os.path.join(gipuma_point_folder, 'cams')
    gipuma_image_folder = os.path.join(gipuma_point_folder, 'images')
    mkdir(gipuma_cam_folder)
    mkdir(gipuma_image_folder)

    for v in range(view_num):
        # convert cameras
        in_cam_file = os.path.join(scene_folder, 'cam_{:08d}_{}.txt'.format(v, name))
        out_cam_file = os.path.join(gipuma_cam_folder, '{:08d}.jpg.P'.format(v))
        mvsnet_to_gipuma_cam(in_cam_file, out_cam_file)

        # convert depth maps and fake normal maps
        gipuma_prefix = '2333__'
        sub_depth_folder = os.path.join(gipuma_point_folder, gipuma_prefix + "{:08d}".format(v))
        mkdir(sub_depth_folder)
        in_depth_pfm = os.path.join(scene_folder, "{:08d}_{}_prob_filtered.pfm".format(v, name))
        out_depth_dmb = os.path.join(sub_depth_folder, 'disp.dmb')
        fake_normal_dmb = os.path.join(sub_depth_folder, 'normals.dmb')
        mvsnet_to_gipuma_dmb(in_depth_pfm, out_depth_dmb)
        fake_colmap_normal(out_depth_dmb, fake_normal_dmb)

        # copy images to gipuma image folder
        in_image_file = os.path.join(scene_folder, '{:08d}.jpg'.format(v))
        out_image_file = os.path.join(gipuma_image_folder, '{:08d}.jpg'.format(v))
        in_image = cv2.imread(in_image_file)

        depth_image = load_pfm(in_depth_pfm)[0]
        if in_image.shape[:2] != depth_image.shape[:2]:
            in_image = cv2.resize(in_image, (depth_image.shape[1], depth_image.shape[0]),
                                  interpolation=cv2.INTER_NEAREST)
        cv2.imwrite(out_image_file, in_image)


def probability_filter(scene_folder, init_prob_threshold, flow_prob_threshold, name, view_num, mode):
    for v in range(view_num):
        init_prob_map_path = os.path.join(scene_folder, "{:08d}_init_prob.pfm".format(v))
        prob_map_path = os.path.join(scene_folder, "{:08d}_{}_prob.pfm".format(v, name))
        init_depth_map_path = os.path.join(scene_folder, "{:08d}_{}.pfm".format(v, name))
        out_depth_map_path = os.path.join(scene_folder, "{:08d}_{}_prob_filtered.pfm".format(v, name))

        depth_map = load_pfm(init_depth_map_path)[0]
        prob_map = load_pfm(prob_map_path)[0]
        init_prob_map = load_pfm(init_prob_map_path)[0]
        if prob_map.shape != depth_map.shape:
            prob_map = cv2.resize(prob_map, (depth_map.shape[1], depth_map.shape[0]), interpolation=mode)
        if init_prob_map.shape != depth_map.shape:
            init_prob_map = cv2.resize(init_prob_map, (depth_map.shape[1], depth_map.shape[0]), interpolation=mode)

        depth_map[prob_map < flow_prob_threshold] = 0
        depth_map[init_prob_map < init_prob_threshold] = 0
        write_pfm(out_depth_map_path, depth_map)


def depth_map_fusion(point_folder, fusibile_exe_path, disp_thresh, num_consistent):
    cam_folder = os.path.join(point_folder, 'cams')
    image_folder = os.path.join(point_folder, 'images')
    depth_min = 0.001
    depth_max = 100000
    normal_thresh = 360

    cmd = fusibile_exe_path
    cmd = cmd + ' -input_folder ' + point_folder + '/'
    cmd = cmd + ' -p_folder ' + cam_folder + '/'
    cmd = cmd + ' -images_folder ' + image_folder + '/'
    cmd = cmd + ' --depth_min=' + str(depth_min)
    cmd = cmd + ' --depth_max=' + str(depth_max)
    cmd = cmd + ' --normal_thresh=' + str(normal_thresh)
    cmd = cmd + ' --disp_thresh=' + str(disp_thresh)
    cmd = cmd + ' --num_consistent=' + str(num_consistent)
    print(cmd)
    os.system(cmd)

    return


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--eval_folder', type=str,
                        default='/home/rayc/Projects/PointMVSNet/data/dtu/Eval/')
    parser.add_argument('--fusibile_exe_path', type=str, default='/home/rayc/Projects/new_fusible/fusibile/fusibile')
    parser.add_argument('--init_prob_threshold', type=float, default=0.2)
    parser.add_argument('--flow_prob_threshold', type=float, default=0.1)
    parser.add_argument('--disp_threshold', type=float, default=0.12)
    parser.add_argument('--num_consistent', type=int, default=3)
    parser.add_argument("-v", '--view_num', type=int, default=49)
    parser.add_argument("-n", '--name', type=str)
    parser.add_argument("-m", '--inter_mode', type=str, default='LANCZOS4')
    parser.add_argument("-f", '--depth_folder', type=str)
    args = parser.parse_args()

    eval_folder = args.eval_folder
    fusibile_exe_path = args.fusibile_exe_path
    init_prob_threshold = args.init_prob_threshold
    flow_prob_threshold = args.flow_prob_threshold
    disp_threshold = args.disp_threshold
    num_consistent = args.num_consistent
    view_num = args.view_num
    name = args.name

    if args.inter_mode == "NEAREST":
        mode = cv2.INTER_NEAREST
    elif args.inter_mode == "BILINEAR":
        mode = cv2.INTER_LINEAR
    elif args.inter_mode == "CUBIC":
        mode = cv2.INTER_CUBIC
    elif args.inter_mode == "LANCZOS4":
        mode = cv2.INTER_LANCZOS4
    else:
        raise ValueError("Unknown interpolation mode: {}.".format(args.inter_mode))

    DEPTH_FOLDER = args.depth_folder

    out_point_folder = os.path.join(eval_folder, DEPTH_FOLDER, '{}_3ITER_{}_ip{}_fp{}_d{}_nc{}'
                                    .format(args.inter_mode, name, init_prob_threshold, flow_prob_threshold,
                                            disp_threshold, num_consistent))
    mkdir(out_point_folder)

    scene_list = ["scan1", "scan4", "scan9", "scan10", "scan11", "scan12", "scan13", "scan15", "scan23",
                  "scan24", "scan29", "scan32", "scan33", "scan34", "scan48", "scan49", "scan62", "scan75",
                  "scan77", "scan110", "scan114", "scan118"]

    for scene in scene_list:
        scene_folder = osp.join(eval_folder, DEPTH_FOLDER, scene)
        if not osp.isdir(scene_folder):
            continue
        if scene[:4] != "scan":
            continue
        print("**** Fusion for {} ****".format(scene))

        # probability filter
        print('filter depth map with probability map')
        probability_filter(scene_folder, init_prob_threshold, flow_prob_threshold, name, view_num, mode)

        # convert to gipuma format
        print('Convert mvsnet output to gipuma input')
        point_folder = osp.join(out_point_folder, scene)
        mkdir(point_folder)
        mvsnet_to_gipuma(scene_folder, point_folder, name, view_num)

        # depth map fusion with gipuma
        print('Run depth map fusion & filter')
        depth_map_fusion(point_folder, fusibile_exe_path, disp_threshold, num_consistent)

        rename_cmd = "cp " + point_folder + "/final3d_model.ply {}/{}_ip{}_fp{}_d{}_nc{}_{}.ply".format(
            point_folder, scene, init_prob_threshold, flow_prob_threshold, disp_threshold, num_consistent,
            args.inter_mode
        )
        os.system(rename_cmd)
