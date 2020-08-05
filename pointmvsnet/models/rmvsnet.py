import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointmvsnet.modules.gru import GRU
from pointmvsnet.modules.unet_ds2gn import UNetDS2GN
from pointmvsnet.modules.networks import MAELoss
from pointmvsnet.functions.functions import get_pixel_grids, get_propability_map
from pointmvsnet.utils.feature_fetcher import FeatureFetcher


class RMVSNet(nn.Module):
    def __init__(self, img_base_channels=8,
                 gru_channels_list=(16, 4, 2),
                 inverse_depth=False,
                 ):
        super(RMVSNet, self).__init__()
        # setup network modules
        self.gru_channels_list = gru_channels_list
        self.inverse_depth = inverse_depth
        self.feature_fetcher = FeatureFetcher()

        self.feature_extractor = UNetDS2GN(img_base_channels)

        gru_in_channels = self.feature_extractor.out_channels
        self.gru_list = nn.ModuleList()
        for gru_channels in gru_channels_list:
            self.gru_list.append(GRU(gru_in_channels, gru_channels, 3))
            gru_in_channels = gru_channels

        self.prob_conv = nn.Conv2d(gru_in_channels, 1, 3, 1, 1)

        self.maeloss = MAELoss()

    def compute_cost_volume(self, warped):
        """
        Warped: N x C x M x H x W
        returns: 1 x C x M x H x W
        """
        warped_sq = warped ** 2
        av_warped = warped.mean(0)
        av_warped_sq = warped_sq.mean(0)
        cost = av_warped_sq - (av_warped ** 2)

        return cost.unsqueeze(0)

    def compute_depth(self, prob_volume, depth_start, depth_interval, depth_num):
        """
        prob_volume: 1 x D x H x W
        """
        _, M, H, W = prob_volume.shape
        # prob_indices = HW shaped vector
        probs, indices = prob_volume.max(1)
        depth_range = depth_start + torch.arange(depth_num).float() * depth_interval
        depth_range = depth_range.to(prob_volume.device)
        depths = torch.index_select(depth_range, 0, indices.flatten())
        depth_image = depths.view(H, W)
        prob_image = probs.view(H, W)

        return depth_image, prob_image

    def forward(self, data_batch, img_scales, inter_scales, use_occ_pred, isFlow, isTest=False):

        preds = collections.OrderedDict()
        img_list = data_batch["img_list"]
        cam_params_list = data_batch["cam_params_list"]
        batch_size, num_view, img_channel, img_height, img_width = img_list.shape

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()
        if isTest:
            cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 4.0

        if self.inverse_depth:
            depth_start = cam_params_list[:, 0, 1, 3, 0]
            depth_interval = cam_params_list[:, 0, 1, 3, 1]
            num_depth = cam_params_list[0, 0, 1, 3, 2].long()
            depth_end = cam_params_list[:, 0, 1, 3, 3]
            disp_interval = (1. / depth_start - 1. / depth_end) / (num_depth - 1)
            d = torch.linspace(0, num_depth - 1, num_depth, device=cam_params_list.device).view(1, 1, 1, num_depth,
                                                                                                1).float() + 1e-6
            d = disp_interval.view(batch_size, 1, 1, 1, 1) * d + (1. / depth_end).view(batch_size, 1, 1, 1, 1)
            depths = 1. / d
        else:
            depth_start = cam_params_list[:, 0, 1, 3, 0]
            depth_interval = cam_params_list[:, 0, 1, 3, 1]
            num_depth = cam_params_list[0, 0, 1, 3, 2].long()
            depth_end = cam_params_list[:, 0, 1, 3, 3]
            depths = []
            for i in range(batch_size):
                depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=cam_params_list.device) \
                              .view(1, 1, num_depth, 1))
            depths = torch.stack(depths, dim=0)  # (B, 1, 1, D, 1)

        img_feature_list = self.feature_extractor(img_list)

        feature_channels, feature_height, feature_width = img_feature_list.shape[2:]

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width) \
            .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(cam_params_list.device)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)

        cam_points = (uv.unsqueeze(3) * depths).view(batch_size, 1, 3, -1)  # (B, 1, 3, D*FH*FW)
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

        num_world_points = world_points.size(-1)
        assert num_world_points == feature_height * feature_width * num_depth

        point_features, _, _ = self.feature_fetcher(img_feature_list, world_points, cam_intrinsic, cam_extrinsic)
        ref_feature_map = img_feature_list[:, 0]

        ref_feature = ref_feature_map.unsqueeze(2).expand(-1, -1, num_depth, -1, -1) \
            .contiguous().view(batch_size, feature_channels, -1)
        point_features[:, 0, :, :] = ref_feature
        avg_point_features = torch.mean(point_features, dim=1)
        avg_point_features2 = torch.mean(point_features ** 2, dim=1)

        point_features = avg_point_features2 - (avg_point_features ** 2)

        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height, feature_width)
        cost_list = []
        for i in range(len(self.gru_channels_list)):
            cost_list.append(None)
        depth_costs = []

        for d in range(num_depth):
            cost_d = cost_volume[:, :, d, :, :]
            for cost_idx in range(len(cost_list)):
                if cost_idx == 0:
                    cost_list[cost_idx] = self.gru_list[cost_idx](-cost_d, cost_list[cost_idx])
                else:
                    cost_list[cost_idx] = self.gru_list[cost_idx](cost_list[cost_idx - 1], cost_list[cost_idx])

            reg_cost = self.prob_conv(cost_list[-1])
            depth_costs.append(reg_cost)

        prob_volume = torch.cat(depth_costs, 1)
        softmax_probs = torch.softmax(prob_volume, 1)

        depth_volume = depths.view(batch_size, num_depth, 1, 1).expand(softmax_probs.shape)
        pred_depth_img = torch.sum(depth_volume * softmax_probs, dim=1).unsqueeze(1)  # (B, 1, FH, FW)

        preds["coarse_depth_map"] = pred_depth_img
        prob_map = get_propability_map(softmax_probs, pred_depth_img, depth_start, depth_interval)
        preds["coarse_prob_map"] = prob_map

        # loss function
        if "gt_depth_img" in data_batch.keys():
            gt_depth_img = data_batch["gt_depth_img"]
            resized_gt_depth = F.interpolate(gt_depth_img, (pred_depth_img.shape[2], pred_depth_img.shape[3]))
            coarse_loss = self.maeloss(pred_depth_img, resized_gt_depth, depth_interval)
            preds["cor_loss"] = coarse_loss

        # compute depth map from prob / depth values
        return preds


def build_rmvsnet(cfg):
    torch.backends.cudnn.benchmark = True
    net = RMVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        gru_channels_list=cfg.MODEL.GRU_CHANNELS_LIST,
        inverse_depth=cfg.MODEL.INVERSE_DEPTH
    )

    return net
