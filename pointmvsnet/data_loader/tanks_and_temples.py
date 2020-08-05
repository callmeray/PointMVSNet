import cv2
import math
import numpy as np
import logging
import torch
from path import Path
from torch.utils.data import Dataset

import pointmvsnet.utils.io as io
from pointmvsnet.utils.preprocess import norm_image, scale_input, crop_input


class TanksAndTemplesTestSet(Dataset):
    mean_dict = {
        "family": torch.tensor([-0.1, 0, -0.15]),
        "francis": torch.tensor([0.1, 0.59, 0.15]),
        "horse": torch.tensor([0.0, 0.0, 0.02]),
        "lighthouse": torch.tensor([0.18, 0.27, -0.62]),
        "m60": torch.tensor([-0.18, 0.0, 0.0]),
        "panther": torch.tensor([-0.05, -0.06, -0.52]),
        "playground": torch.tensor([-0.12, -0.02, 0.18]),
        "train": torch.tensor([-0.57, -0.066, -0.1493]),
        "Barn": torch.tensor([0.0, 0.0, 0.0]),
        "Caterpillar": torch.tensor([0.0, 0.0, 0.0]),
        "Church": torch.tensor([0.0, 0.0, 0.0]),
        "Courthouse": torch.tensor([0.0, 0.0, 0.0]),
        "Ignatius": torch.tensor([0.0, 0.0, 0.0]),
        "Meetingroom": torch.tensor([0.0, 0.0, 0.0]),
        "Truck": torch.tensor([0.0, 0.0, 0.0]),
    }
    std_dict = {
        "family": torch.tensor([0.62, 0.3, 0.61]),
        "francis": torch.tensor([1.2850, 1.0500, 0.7350]),
        "horse": torch.tensor([0.2500, 0.1250, 0.2600]),
        "lighthouse": torch.tensor([1.2500, 0.8350, 1.2450]),
        "m60": torch.tensor([1.1750, 0.4700, 1.1500]),
        "panther": torch.tensor([1.9, 1.06, 3.05]) * 0.5,
        "playground": torch.tensor([0.9500, 0.5300, 1.5250]),
        "train": torch.tensor([1.6, 0.5, 1.8]),
        "Barn": torch.tensor([1.0, 1.0, 1.0]),
        "Caterpillar": torch.tensor([1.0, 1.0, 1.0]),
        "Church": torch.tensor([1.0, 1.0, 1.0]),
        "Courthouse": torch.tensor([1.0, 1.0, 1.0]),
        "Ignatius": torch.tensor([1.0, 1.0, 1.0]),
        "Meetingroom": torch.tensor([1.0, 1.0, 1.0]),
        "Truck": torch.tensor([1.0, 1.0, 1.0]),
    }

    def __init__(self, root_dir,
                 num_view=3,
                 num_depth=128,
                 interval_scale=1.6,
                 img_height=1152, img_width=1600,
                 base_image_size=64,
                 depth_in_folder="",
                 depth_in_name="flow3",
                 prob_in_name="init"):

        self.root_dir = Path(root_dir)
        self.scene_list = [s.strip() for s in open(self.root_dir / "scenes.txt").readlines()]
        self.num_view = num_view
        self.num_depth = num_depth
        self.interval_scale = interval_scale
        self.img_height = img_height
        self.img_width = img_width
        self.base_image_size = base_image_size

        self.depth_folder = Path(depth_in_folder)
        self.depth_in_name = depth_in_name
        self.prob_in_name = prob_in_name

        self.path_list = self._load_dataset(self.scene_list)
        logger = logging.getLogger("pointmvsnet.dataset")
        logger.info("TanksAndTemples dataset: scene_list: [{}]; length: [{}].".format(
            self.scene_list, len(self.path_list)))

    def _load_dataset(self, scene_list):
        path_list = []
        for scene in scene_list:
            image_folder = self.root_dir / scene / "images"
            cam_folder = self.root_dir / scene / "cams"
            cam_pair = open(self.root_dir / scene / "pair.txt").read().split()

            # for each reference image
            for p in range(0, int(cam_pair[0])):
                paths = {}

                view_image_paths = []
                view_cam_paths = []

                # ref image
                ref_index = int(cam_pair[22 * p + 1])
                ref_image_path = image_folder / "{:08d}.jpg".format(ref_index)
                ref_cam_path = cam_folder / "{:08d}_cam.txt".format(ref_index)

                view_image_paths.append(ref_image_path)
                view_cam_paths.append(ref_cam_path)

                # view images
                for view in range(self.num_view - 1):
                    view_index = int(cam_pair[22 * p + 2 * view + 3])
                    view_image_path = image_folder / "{:08d}.jpg".format(view_index)
                    view_cam_path = cam_folder / "{:08d}_cam.txt".format(view_index)

                    view_image_paths.append(view_image_path)
                    view_cam_paths.append(view_cam_path)

                paths["view_image_paths"] = view_image_paths
                paths["view_cam_paths"] = view_cam_paths
                paths["depth_map_path"] = self.depth_folder / scene / "{:08d}_{}.pfm".format(ref_index,
                                                                                             self.depth_in_name)
                paths["prob_map_path"] = self.depth_folder / scene / "{:08d}_{}_prob.pfm".format(ref_index,
                                                                                                 self.prob_in_name)
                paths["mean"] = self.mean_dict[scene]
                paths["std"] = self.std_dict[scene]

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

            cam = io.load_cam_dtu(paths["view_cam_paths"][view], self.num_depth, self.interval_scale)
            images.append(image)
            cams.append(cam)
        if self.depth_folder:
            ref_depth, _ = io.load_pfm(paths["depth_map_path"])
            ref_prob_map, _ = io.load_pfm(paths["prob_map_path"])
            ref_h, ref_w = ref_depth.shape
            ref_prob_h, ref_prob_w = ref_prob_map.shape
            ori_h, ori_w = images[0].shape[0:2]
            h_scale = float(ref_h) / ori_h
            w_scale = float(ref_w) / ori_w

            resize_scale = h_scale
            if w_scale > h_scale:
                resize_scale = w_scale
            ref_depth = cv2.resize(ref_depth, (int(ref_w / resize_scale), int(ref_h / resize_scale)))
            new_h, new_w = ref_depth.shape
            # assert (new_h / ref_prob_h) == (new_w / ref_w)
            ref_prob_map = cv2.resize(ref_prob_map, (new_w, new_h))
            start_h = int(math.floor((ori_h - new_h) / 2))
            start_w = int(math.floor((ori_w - new_w) / 2))
            finish_h = start_h + new_h
            finish_w = start_w + new_w
            for view in range(len(images)):
                images[view] = images[view][start_h:finish_h, start_w:finish_w]
                cams[view][1][0][2] = cams[view][1][0][2] - start_w
                cams[view][1][1][2] = cams[view][1][1][2] - start_h

        else:
            ref_depth = np.zeros((self.img_height // 4, self.img_width // 4))
            ref_prob_map = np.zeros((self.img_height // 4, self.img_width // 4))

        h_scale = float(self.img_height) / images[0].shape[0]
        w_scale = float(self.img_width) / images[0].shape[1]
        if h_scale > 1 or w_scale > 1:
            print("max_h, max_w should < W and H!")
            exit()
        resize_scale = h_scale
        if w_scale > h_scale:
            resize_scale = w_scale
        scaled_input_images, scaled_input_cams, ref_depth, ref_prob_map = scale_input(images, cams, scale=resize_scale,
                                                                                      depth_image=ref_depth,
                                                                                      prob_image=ref_prob_map)

        # crop to fit network
        croped_images, croped_cams, ref_depth, ref_prob_map = crop_input(scaled_input_images, scaled_input_cams,
                                                                         height=self.img_height, width=self.img_width,
                                                                         base_image_size=self.base_image_size,
                                                                         depth_image=ref_depth,
                                                                         prob_image=ref_prob_map)

        ref_image = croped_images[0].copy()
        for i, image in enumerate(croped_images):
            croped_images[i] = norm_image(image)

        img_list = np.stack(croped_images, axis=0)
        cam_params_list = np.stack(croped_cams, axis=0)

        img_list = torch.tensor(img_list).permute(0, 3, 1, 2).float()
        cam_params_list = torch.tensor(cam_params_list).float()

        data_batch = {
            "img_list": img_list,
            "cam_params_list": cam_params_list,
            "ref_img_path": paths["view_image_paths"][0],
            "ref_img": ref_image,
            "mean": paths["mean"].float(),
            "std": paths["std"].float(),
        }
        if self.depth_folder:
            ref_depth = torch.tensor(ref_depth).unsqueeze(0).float()
            ref_prob_map = torch.tensor(ref_prob_map.copy()).unsqueeze(0).float()
            data_batch["init_depth_map"] = ref_depth
            data_batch["init_prob_map"] = ref_prob_map
        return data_batch

    def __len__(self):
        return len(self.path_list)
