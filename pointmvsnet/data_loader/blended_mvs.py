import cv2
import numpy as np
from path import Path
import logging

import torch
from torch.utils.data import Dataset
from pointmvsnet.utils.preprocess import mask_depth_image, norm_image
import pointmvsnet.utils.io as io


class BlendedMVSTrainValSet(Dataset):
    train_list_file = "training_list.txt"
    val_list_file = "validation_list.txt"

    mean = torch.tensor([-1.526, -3.182, 38.524])
    std = torch.tensor([2.0, 2.0, 2.0])

    img_height, img_width = 512, 640

    def __init__(self, root_dir, mode,
                 num_view=3,
                 num_depth=128,
                 interval_scale=1.6,
                 ):

        self.root_dir = Path(root_dir)
        self.num_view = num_view
        self.num_depth = num_depth
        self.interval_scale = interval_scale

        assert (mode in ["train", "val"]), "Unknown mode name: {}".format(mode)
        if mode == "train":
            self.path_list = self._load_dataset(self.train_list_file)
        else:
            self.path_list = self._load_dataset(self.val_list_file)
        logger = logging.getLogger("pointmvsnet.dataset")
        logger.info("Blended MVS dataset: mode: [{}]; length: [{}].".format(mode, len(self.path_list)))

    def _load_dataset(self, list_file):
        folder_list = open(self.root_dir / list_file, "r").read().splitlines()

        path_list = []

        for folder in folder_list:
            folder = self.root_dir / folder
            cluster_path = folder / "cams" / "pair.txt"
            cluster_lines = open(cluster_path).read().splitlines()
            image_num = int(cluster_lines[0])

            for idx in range(image_num):
                ref_idx = int(cluster_lines[2 * idx + 1])
                cluster_info = cluster_lines[2 * idx + 2].split()
                total_view_num = int(cluster_info[0])
                if total_view_num < self.num_view - 1:
                    continue
                paths = {}
                view_image_paths = []
                view_cam_paths = []
                view_depth_paths = []

                # ref image
                ref_image_path = folder / "blended_images" / "{:08d}.jpg".format(ref_idx)
                ref_depth_path = folder / "rendered_depth_maps" / "{:08d}.pfm".format(ref_idx)
                ref_cam_path = folder / "cams" / "{:08d}_cam.txt".format(ref_idx)

                view_image_paths.append(ref_image_path)
                view_depth_paths.append(ref_depth_path)
                view_cam_paths.append(ref_cam_path)

                # view images
                for cidx in range(self.num_view - 1):
                    view_idx = int(cluster_info[2 * cidx + 1])
                    view_image_path = folder / "blended_images" / "{:08d}.jpg".format(view_idx)
                    view_depth_path = folder / "rendered_depth_maps" / "{:08d}.pfm".format(view_idx)
                    view_cam_path = folder / "cams" / "{:08d}_cam.txt".format(view_idx)

                    view_image_paths.append(view_image_path)
                    view_depth_paths.append(view_depth_path)
                    view_cam_paths.append(view_cam_path)

                paths["view_image_paths"] = view_image_paths
                paths["view_depth_paths"] = view_depth_paths
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
                    image = norm_image(image)
                except Exception:
                    print(paths["view_image_paths"][view])
                    continue
                break
            cam = io.load_cam_dtu(paths["view_cam_paths"][view], self.num_depth, self.interval_scale)
            img_heigt, img_width = image.shape[:2]
            cam[1][0, :3] = cam[1][0, :3] / img_width * self.img_width
            cam[1][1, :3] = cam[1][1, :3] / img_heigt * self.img_height
            image = cv2.resize(image, (self.img_width, self.img_height), interpolation=cv2.INTER_LINEAR)
            cam[1][:2, :3] /= 4.0
            images.append(image)
            cams.append(cam)

        # mask out-of-range depth pixels (in a relaxed range)
        ref_depth = io.load_pfm(paths["view_depth_paths"][0])
        ref_depth = cv2.resize(ref_depth, (self.img_width, self.img_height), interpolation=cv2.INTER_NEAREST)
        depth_start = cams[0][1, 3, 0] + cams[0][1, 3, 1]
        depth_end = cams[0][1, 3, 3] - cams[0][1, 3, 1]
        ref_depth = mask_depth_image(ref_depth, depth_start, depth_end)
        ref_depth[np.isnan(ref_depth)] = 0.0

        img_list = np.stack(images, axis=0)
        cam_params_list = np.stack(cams, axis=0)

        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).type(torch.float)
        cam_params_list = torch.tensor(cam_params_list).type(torch.float)
        ref_depth = torch.tensor(ref_depth).permute(2, 0, 1).type(torch.float)

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
