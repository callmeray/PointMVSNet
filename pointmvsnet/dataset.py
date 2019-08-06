import os
import os.path as osp
from tqdm import tqdm
import cv2
import numpy as np
import scipy.io

import torch
from torch.utils.data import Dataset, DataLoader

from pointmvsnet.utils.preprocess import mask_depth_image, norm_image, scale_dtu_input, crop_dtu_input
import pointmvsnet.utils.io as io


class DTU_Train_Val_Set(Dataset):
    training_set = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                    45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                    74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                    101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                    121, 122, 123, 124, 125, 126, 127, 128]
    validation_set = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

    training_lighting_set = [0, 1, 2, 3, 4, 5, 6]
    validation_lighting_set = [3]

    mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
    std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    cluster_file_path = "Cameras/pair.txt"

    def __init__(self, root_dir, dataset_name,
                 num_view=3,
                 num_virtual_plane=128,
                 interval_scale=1.6,
                 ):

        self.root_dir = root_dir
        self.num_view = num_view
        self.interval_scale = interval_scale
        self.num_virtual_plane = num_virtual_plane

        self.cluster_file_path = osp.join(root_dir, self.cluster_file_path)
        self.cluster_list = open(self.cluster_file_path).read().split()
        # self.cluster_list =
        assert (dataset_name in ["train", "valid"]), "Unknown dataset_name: {}".format(dataset_name)

        if dataset_name == "train":
            self.data_set = self.training_set
            self.lighting_set = self.training_lighting_set
        elif dataset_name == "valid":
            self.data_set = self.validation_set
            self.lighting_set = self.validation_lighting_set

        self.path_list = self._load_dataset(self.data_set, self.lighting_set)

    def _load_dataset(self, dataset, lighting_set):
        path_list = []
        for ind in dataset:
            image_folder = osp.join(self.root_dir, "Rectified/scan{}_train".format(ind))
            cam_folder = osp.join(self.root_dir, "Cameras/train")
            depth_folder = osp.join(self.root_dir, "Depths/scan{}_train".format(ind))

            for lighting_ind in lighting_set:
                # for each reference image
                for p in range(0, int(self.cluster_list[0])):
                    paths = {}
                    pts_paths = []
                    view_image_paths = []
                    view_cam_paths = []
                    view_depth_paths = []

                    # ref image
                    ref_index = int(self.cluster_list[22 * p + 1])
                    ref_image_path = osp.join(
                        image_folder, "rect_{:03d}_{}_r5000.png".format(ref_index + 1, lighting_ind))
                    ref_cam_path = osp.join(cam_folder, "{:08d}_cam.txt".format(ref_index))
                    ref_depth_path = osp.join(depth_folder, "depth_map_{:04d}.pfm".format(ref_index))

                    view_image_paths.append(ref_image_path)
                    view_cam_paths.append(ref_cam_path)
                    view_depth_paths.append(ref_depth_path)

                    # view images
                    for view in range(self.num_view - 1):
                        view_index = int(self.cluster_list[22 * p + 2 * view + 3])
                        view_image_path = osp.join(
                            image_folder, "rect_{:03d}_{}_r5000.png".format(view_index + 1, lighting_ind))
                        view_cam_path = osp.join(cam_folder, "{:08d}_cam.txt".format(view_index))
                        view_depth_path = osp.join(depth_folder, "depth_map_{:04d}.pfm".format(view_index))
                        view_image_paths.append(view_image_path)
                        view_cam_paths.append(view_cam_path)
                        view_depth_paths.append(view_depth_path)
                    paths["view_image_paths"] = view_image_paths
                    paths["view_cam_paths"] = view_cam_paths
                    paths["view_depth_paths"] = view_depth_paths

                    path_list.append(paths)

        return path_list

    def __getitem__(self, index):
        paths = self.path_list[index]
        images = []
        cams = []
        for view in range(self.num_view):
            while True:
                try:
                    image = cv2.imread(paths["view_image_paths"][view])
                    image = norm_image(image)
                except Exception:
                    print(paths["view_image_paths"][view])
                    continue
                break
            cam = io.load_cam_dtu(open(paths["view_cam_paths"][view]),
                                  num_depth=self.num_virtual_plane,
                                  interval_scale=self.interval_scale)
            images.append(image)
            cams.append(cam)

        depth_images = []
        for depth_path in paths["view_depth_paths"]:
            depth_image = io.load_pfm(depth_path)[0]
            depth_images.append(depth_image)

        # mask out-of-range depth pixels (in a relaxed range)
        ref_depth = depth_images[0]
        depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
        depth_end = cams[0][1, 3, 0] + (self.num_virtual_plane - 2) * cams[0][1, 3, 1]
        ref_depth = mask_depth_image(ref_depth, depth_start, depth_end)

        depth_list = np.stack(depth_images, axis=0)
        img_list = np.stack(images, axis=0)
        cam_params_list = np.stack(cams, axis=0)

        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).type(torch.float)
        cam_params_list = torch.tensor(cam_params_list).type(torch.float)
        ref_depth = torch.tensor(ref_depth).permute(2, 0, 1).type(torch.float)
        depth_list = torch.tensor(depth_list).unsqueeze(-1).permute(0, 3, 1, 2).type(torch.float)
        depth_list = depth_list * (depth_list > depth_start).float() * (depth_list < depth_end).float()

        return {
            "img_list": img_list,
            "cam_params_list": cam_params_list,
            "gt_depth_img": ref_depth,
            "depth_list": depth_list,
            "ref_img_path": paths["view_image_paths"][0],
            "mean": self.mean,
            "std": self.std,
        }

    def __len__(self):
        return len(self.path_list)


class DTU_Test_Set(Dataset):
    test_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77,
                110, 114, 118]
    # test_set = [1]
    test_lighting_set = [3]

    mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
    std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    cluster_file_path = "Cameras/pair.txt"

    def __init__(self, root_dir, dataset_name,
                 num_view=3,
                 height=1152, width=1600,
                 num_virtual_plane=128,
                 interval_scale=1.6,
                 base_image_size=64,
                 depth_folder=""):

        self.root_dir = root_dir
        self.num_view = num_view
        self.interval_scale = interval_scale
        self.num_virtual_plane = num_virtual_plane
        self.base_image_size = base_image_size
        self.height = height
        self.width = width
        self.depth_folder = depth_folder

        self.cluster_file_path = osp.join(root_dir, self.cluster_file_path)
        self.cluster_list = open(self.cluster_file_path).read().split()
        # self.cluster_list =
        assert (dataset_name in ["test"]), "Unknown dataset_name: {}".format(dataset_name)

        self.data_set = self.test_set
        self.lighting_set = self.test_lighting_set

        self.path_list = self._load_dataset(self.data_set, self.lighting_set)

    def _load_dataset(self, dataset, lighting_set):
        path_list = []
        for ind in dataset:
            image_folder = osp.join(self.root_dir, "Eval/Rectified/scan{}".format(ind))
            cam_folder = osp.join(self.root_dir, "Cameras")
            depth_folder = osp.join(self.depth_folder, "scan{}".format(ind))

            for lighting_ind in lighting_set:
                # for each reference image
                for p in range(0, int(self.cluster_list[0])):
                    paths = {}
                    # pts_paths = []
                    view_image_paths = []
                    view_cam_paths = []
                    view_depth_paths = []

                    # ref image
                    ref_index = int(self.cluster_list[22 * p + 1])
                    ref_image_path = osp.join(
                        image_folder, "rect_{:03d}_{}_r5000.png".format(ref_index + 1, lighting_ind))
                    ref_cam_path = osp.join(cam_folder, "{:08d}_cam.txt".format(ref_index))
                    ref_depth_path = osp.join(depth_folder, "depth_map_{:04d}.pfm".format(ref_index))

                    view_image_paths.append(ref_image_path)
                    view_cam_paths.append(ref_cam_path)
                    view_depth_paths.append(ref_depth_path)

                    # view images
                    for view in range(self.num_view - 1):
                        view_index = int(self.cluster_list[22 * p + 2 * view + 3])
                        view_image_path = osp.join(
                            image_folder, "rect_{:03d}_{}_r5000.png".format(view_index + 1, lighting_ind))
                        view_cam_path = osp.join(cam_folder, "{:08d}_cam.txt".format(view_index))
                        view_depth_path = osp.join(depth_folder, "depth_map_{:04d}.pfm".format(view_index))
                        view_image_paths.append(view_image_path)
                        view_cam_paths.append(view_cam_path)
                        view_depth_paths.append(view_depth_path)
                    paths["view_image_paths"] = view_image_paths
                    paths["view_cam_paths"] = view_cam_paths
                    paths["view_depth_paths"] = view_depth_paths

                    path_list.append(paths)

        return path_list

    def __getitem__(self, index):
        paths = self.path_list[index]
        depth_images = []

        images = []
        cams = []
        for view in range(self.num_view):
            while True:
                try:
                    image = cv2.imread(paths["view_image_paths"][view])
                except Exception:
                    print(paths["view_image_paths"][view])
                    continue
                break

            cam = io.load_cam_dtu(open(paths["view_cam_paths"][view]),
                                  num_depth=self.num_virtual_plane,
                                  interval_scale=self.interval_scale)
            images.append(image)
            cams.append(cam)

        if self.depth_folder:
            for depth_path in paths["view_depth_paths"]:
                depth_image = io.load_pfm(depth_path)[0]
                depth_images.append(depth_image)
        else:
            for depth_path in paths["view_depth_paths"]:
                depth_images.append(np.zeros((self.height, self.width), np.float))

        ref_depth = depth_images[0].copy()

        h_scale = float(self.height) / images[0].shape[0]
        w_scale = float(self.width) / images[0].shape[1]
        if h_scale > 1 or w_scale > 1:
            print("max_h, max_w should < W and H!")
            exit()
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
        scaled_input_images, scaled_input_cams, ref_depth = scale_dtu_input(images, cams, depth_image=ref_depth,
                                                                            scale=resize_scale)

        # crop to fit network
        croped_images, croped_cams, ref_depth = crop_dtu_input(scaled_input_images, scaled_input_cams,
                                                               height=self.height, width=self.width,
                                                               base_image_size=self.base_image_size,
                                                               depth_image=ref_depth)
        ref_image = croped_images[0].copy()
        for i, image in enumerate(croped_images):
            croped_images[i] = norm_image(image)

        depth_list = np.stack(depth_images, axis=0)
        img_list = np.stack(croped_images, axis=0)
        cam_params_list = np.stack(croped_cams, axis=0)
        # cam_pos_list = np.stack(camspos, axis=0)

        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
        cam_params_list = torch.tensor(cam_params_list).float()
        depth_list = torch.tensor(depth_list).unsqueeze(-1).permute(0, 3, 1, 2).float()

        return {
            "img_list": img_list,
            "cam_params_list": cam_params_list,
            "gt_depth_img": ref_depth,
            "depth_list": depth_list,
            "ref_img_path": paths["view_image_paths"][0],
            "ref_img": ref_image,
            "mean": self.mean,
            "std": self.std,
        }

    def __len__(self):
        return len(self.path_list)


def build_data_loader(cfg, mode="train"):
    if mode == "train":
        dataset = DTU_Train_Val_Set(
            root_dir=cfg.DATA.TRAIN.ROOT_DIR,
            dataset_name="train",
            num_view=cfg.DATA.TRAIN.NUM_VIEW,
            interval_scale=cfg.DATA.TRAIN.INTER_SCALE,
            num_virtual_plane=cfg.DATA.TRAIN.NUM_VIRTUAL_PLANE,
        )
    elif mode == "val":
        dataset = DTU_Train_Val_Set(
            root_dir=cfg.DATA.VAL.ROOT_DIR,
            dataset_name="val",
            num_view=cfg.DATA.VAL.NUM_VIEW,
            interval_scale=cfg.DATA.TRAIN.INTER_SCALE,
            num_virtual_plane=cfg.DATA.TRAIN.NUM_VIRTUAL_PLANE,
        )
    elif mode == "test":
        dataset = DTU_Test_Set(
            root_dir=cfg.DATA.TEST.ROOT_DIR,
            dataset_name="test",
            num_view=cfg.DATA.TEST.NUM_VIEW,
            height=cfg.DATA.TEST.IMG_HEIGHT,
            width=cfg.DATA.TEST.IMG_WIDTH,
            interval_scale=cfg.DATA.TEST.INTER_SCALE,
            num_virtual_plane=cfg.DATA.TEST.NUM_VIRTUAL_PLANE,
        )
    else:
        raise ValueError("Unknown mode: {}.".format(mode))

    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
    )

    return data_loader
