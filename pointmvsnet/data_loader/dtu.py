import cv2
import numpy as np
from path import Path
import logging

import torch
from torch.utils.data import Dataset

from pointmvsnet.utils.preprocess import mask_depth_image, norm_image, scale_input, crop_input
import pointmvsnet.utils.io as io


class DTUTrainValSet(Dataset):
    training_scene_list = [2, 6, 7, 8, 14, 16, 18, 19, 20, 22, 30, 31, 36, 39, 41, 42, 44,
                           45, 46, 47, 50, 51, 52, 53, 55, 57, 58, 60, 61, 63, 64, 65, 68, 69, 70, 71, 72,
                           74, 76, 83, 84, 85, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100,
                           101, 102, 103, 104, 105, 107, 108, 109, 111, 112, 113, 115, 116, 119, 120,
                           121, 122, 123, 124, 125, 126, 127, 128]
    validation_scene_list = [3, 5, 17, 21, 28, 35, 37, 38, 40, 43, 56, 59, 66, 67, 82, 86, 106, 117]

    training_lighting_set = [0, 1, 2, 3, 4, 5, 6]
    validation_lighting_set = [3]

    mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
    std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    cluster_file_path = "Cameras/pair.txt"

    def __init__(self, root_dir, mode,
                 num_view=3,
                 num_depth=128,
                 interval_scale=1.6,
                 ):

        self.root_dir = Path(root_dir)
        self.num_view = num_view
        self.num_depth = num_depth
        self.interval_scale = interval_scale

        self.cluster_file_path = self.root_dir / self.cluster_file_path
        self.cluster_list = open(self.cluster_file_path).read().split()
        assert (mode in ["train", "val"]), "Unknown mode: {}".format(mode)

        if mode == "train":
            self.scene_list = self.training_scene_list
            self.lighting_list = self.training_lighting_set
        elif mode == "val":
            self.scene_list = self.validation_scene_list
            self.lighting_list = self.validation_lighting_set

        self.path_list = self._load_dataset(self.scene_list, self.lighting_list)
        logger = logging.getLogger("pointmvsnet.dataset")
        logger.info("DTU dataset: mode: [{}]; length: [{}].".format(mode, len(self.path_list)))

    def _load_dataset(self, scene_list, lighting_list):
        path_list = []
        for ind in scene_list:
            image_folder = self.root_dir / "Rectified/scan{}_train".format(ind)
            cam_folder = self.root_dir / "Cameras/train"
            depth_folder = self.root_dir / "Depths/scan{}_train".format(ind)

            for lighting_ind in lighting_list:
                # for each reference image
                for p in range(0, int(self.cluster_list[0])):
                    paths = {}
                    view_image_paths = []
                    view_cam_paths = []
                    view_depth_paths = []

                    # ref image
                    ref_index = int(self.cluster_list[22 * p + 1])
                    ref_image_path = image_folder / "rect_{:03d}_{}_r5000.png".format(ref_index + 1, lighting_ind)
                    ref_cam_path = cam_folder / "{:08d}_cam.txt".format(ref_index)
                    ref_depth_path = depth_folder / "depth_map_{:04d}.pfm".format(ref_index)

                    view_image_paths.append(ref_image_path)
                    view_cam_paths.append(ref_cam_path)
                    view_depth_paths.append(ref_depth_path)

                    # view images
                    for view in range(self.num_view - 1):
                        view_index = int(self.cluster_list[22 * p + 2 * view + 3])
                        view_image_path = image_folder / "rect_{:03d}_{}_r5000.png".format(view_index + 1, lighting_ind)
                        view_cam_path = cam_folder / "{:08d}_cam.txt".format(view_index)
                        view_depth_path = depth_folder / "depth_map_{:04d}.pfm".format(view_index)
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
            cam = io.load_cam_dtu(paths["view_cam_paths"][view], self.num_depth, self.interval_scale)
            images.append(image)
            cams.append(cam)

        # mask out-of-range depth pixels
        ref_depth = io.load_pfm(paths["view_depth_paths"][0])
        depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
        depth_end = cams[0][1, 3, 3] - cams[0][1, 3, 1]
        ref_depth = mask_depth_image(ref_depth, depth_start, depth_end)

        img_list = np.stack(images, axis=0)
        cam_params_list = np.stack(cams, axis=0)

        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
        cam_params_list = torch.tensor(cam_params_list).float()
        ref_depth = torch.tensor(ref_depth).permute(2, 0, 1).float()

        return {
            "img_list": img_list,
            "cam_params_list": cam_params_list,
            "gt_depth_img": ref_depth,
            "ref_img_path": paths["view_image_paths"][0],
            "mean": self.mean,
            "std": self.std,
        }

    def __len__(self):
        return len(self.path_list)


class DTUTestSet(Dataset):
    test_set = [1, 4, 9, 10, 11, 12, 13, 15, 23, 24, 29, 32, 33, 34, 48, 49, 62, 75, 77,
                110, 114, 118]
    # test_set = [1]
    test_lighting_set = [3]

    mean = torch.tensor([1.97145182, -1.52387525, 651.07223895])
    std = torch.tensor([84.45612252, 93.22252387, 80.08551226])

    cluster_file_path = "Cameras/pair.txt"

    def __init__(self, root_dir,
                 num_view=3,
                 num_depth=128,
                 interval_scale=1.6,
                 img_height=1152, img_width=1600,
                 base_image_size=64,
                 ):

        self.root_dir = Path(root_dir)
        self.num_view = num_view
        self.num_depth = num_depth
        self.interval_scale = interval_scale
        self.img_height = img_height
        self.img_width = img_width
        self.base_image_size = base_image_size

        self.cluster_file_path = root_dir / self.cluster_file_path
        self.cluster_list = open(self.cluster_file_path).read().split()

        self.data_set = self.test_set
        self.lighting_set = self.test_lighting_set

        self.path_list = self._load_dataset(self.data_set, self.lighting_set)

    def _load_dataset(self, dataset, lighting_set):
        path_list = []
        for ind in dataset:
            image_folder = self.root_dir / "Eval/Rectified/scan{}".format(ind)
            cam_folder = self.root_dir / "Cameras"

            for lighting_ind in lighting_set:
                # for each reference image
                for p in range(0, int(self.cluster_list[0])):
                    paths = {}
                    view_image_paths = []
                    view_cam_paths = []

                    # ref image
                    ref_index = int(self.cluster_list[22 * p + 1])
                    ref_image_path = image_folder / "rect_{:03d}_{}_r5000.png".format(ref_index + 1, lighting_ind)
                    ref_cam_path = cam_folder / "{:08d}_cam.txt".format(ref_index)

                    view_image_paths.append(ref_image_path)
                    view_cam_paths.append(ref_cam_path)

                    # view images
                    for view in range(self.num_view - 1):
                        view_index = int(self.cluster_list[22 * p + 2 * view + 3])
                        view_image_path = image_folder / "rect_{:03d}_{}_r5000.png".format(view_index + 1, lighting_ind)
                        view_cam_path = cam_folder / "{:08d}_cam.txt".format(view_index)
                        view_image_paths.append(view_image_path)
                        view_cam_paths.append(view_cam_path)
                    paths["view_image_paths"] = view_image_paths
                    paths["view_cam_paths"] = view_cam_paths

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
                except Exception:
                    print(paths["view_image_paths"][view])
                    continue
                break
            cam = io.load_cam_dtu(open(paths["view_cam_paths"][view]), self.num_depth, self.interval_scale)
            images.append(image)
            cams.append(cam)

        h_scale = float(self.img_height) / images[0].shape[0]
        w_scale = float(self.img_width) / images[0].shape[1]
        if h_scale > 1 or w_scale > 1:
            print("max_h, max_w should < W and H!")
            exit()
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
        scaled_input_images, scaled_input_cams, ref_depth = scale_input(images, cams, scale=resize_scale)

        # crop to fit network
        croped_images, croped_cams, ref_depth = crop_input(scaled_input_images, scaled_input_cams,
                                                           height=self.img_height, width=self.img_width,
                                                           base_image_size=self.base_image_size)
        ref_image = croped_images[0].copy()
        for i, image in enumerate(croped_images):
            croped_images[i] = norm_image(image)

        img_list = np.stack(croped_images, axis=0)
        cam_params_list = np.stack(croped_cams, axis=0)

        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
        cam_params_list = torch.tensor(cam_params_list).float()

        return {
            "img_list": img_list,
            "cam_params_list": cam_params_list,
            "ref_img_path": paths["view_image_paths"][0],
            "ref_img": ref_image,
            "mean": self.mean,
            "std": self.std,
        }

    def __len__(self):
        return len(self.path_list)
