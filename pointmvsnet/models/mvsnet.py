import collections
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/..')

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointmvsnet.nn.conv import Conv2d, Conv3d, Deconv3d
from pointmvsnet.modules.networks import MAELoss
from pointmvsnet.functions.functions import get_pixel_grids, get_propability_map
from pointmvsnet.utils.feature_fetcher import FeatureFetcher


class FeatureNet(nn.Module):
    def __init__(self, base_channels):
        super(FeatureNet, self).__init__()
        self.base_channels = base_channels
        self.out_channels = 4 * base_channels
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, 1, padding=1),
            Conv2d(base_channels, base_channels, 3, 1, padding=1),
        )
        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
            Conv2d(base_channels * 2, base_channels * 2, 3, 1, padding=1),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2),
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
            nn.Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, bias=False)
        )

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class CostRegNet(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(CostRegNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = 1
        self.base_channels = base_channels
        self.conv1_0 = Conv3d(in_channels, base_channels * 2, 3, stride=2, padding=1)
        self.conv2_0 = Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1)
        self.conv3_0 = Conv3d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1)

        self.conv0_1 = Conv3d(in_channels, base_channels, 3, 1, padding=1)

        self.conv1_1 = Conv3d(base_channels * 2, base_channels * 2, 3, 1, padding=1)
        self.conv2_1 = Conv3d(base_channels * 4, base_channels * 4, 3, 1, padding=1)

        self.conv3_1 = Conv3d(base_channels * 8, base_channels * 8, 3, 1, padding=1)
        self.conv4_0 = Deconv3d(base_channels * 8, base_channels * 4, 3, 2, padding=1, output_padding=1)
        self.conv5_0 = Deconv3d(base_channels * 4, base_channels * 2, 3, 2, padding=1, output_padding=1)
        self.conv6_0 = Deconv3d(base_channels * 2, base_channels, 3, 2, padding=1, output_padding=1)

        self.conv6_2 = nn.Conv3d(base_channels, 1, 3, padding=1, bias=False)

    def forward(self, x):
        conv0_1 = self.conv0_1(x)

        conv1_0 = self.conv1_0(x)
        conv2_0 = self.conv2_0(conv1_0)
        conv3_0 = self.conv3_0(conv2_0)

        conv1_1 = self.conv1_1(conv1_0)
        conv2_1 = self.conv2_1(conv2_0)
        conv3_1 = self.conv3_1(conv3_0)

        conv4_0 = self.conv4_0(conv3_1)

        conv5_0 = self.conv5_0(conv4_0 + conv2_1)
        conv6_0 = self.conv6_0(conv5_0 + conv1_1)

        conv6_2 = self.conv6_2(conv6_0 + conv0_1)

        return conv6_2


class RefineNet(nn.Module):
    def __init__(self):
        super(RefineNet, self).__init__()

        self.conv = nn.Sequential(
            Conv2d(4, 32, 3, padding=1),
            Conv2d(32, 32, 3, padding=1),
            Conv2d(32, 32, 3, padding=1))
        self.res = nn.Conv2d(32, 1, 3, padding=1, bias=False)
        nn.init.zeros_(self.res.weight)

    def forward(self, img, depth_init):
        concat = torch.cat((img, depth_init), dim=1)
        depth_res = self.res(self.conv(concat))
        depth_refined = depth_init + depth_res

        return depth_refined


class MVSNet(nn.Module):
    def __init__(self,
                 img_base_channels,
                 vol_base_channels,
                 inverse_depth=False,
                 refine=True,
                 ):
        super(MVSNet, self).__init__()
        self.inverse_depth = inverse_depth
        self.refine = refine
        self.feature_fetcher = FeatureFetcher()

        self.feature_net = FeatureNet(img_base_channels)
        self.cost_regularization_net = CostRegNet(self.feature_net.out_channels, vol_base_channels)
        self.maeloss = MAELoss()
        if refine:
            self.refine_net = RefineNet()
            self.refine_maeloss = MAELoss()

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

        coarse_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :].contiguous()
            curr_feature_map = self.feature_net(curr_img)
            coarse_feature_maps.append(curr_feature_map)
        feature_list = torch.stack(coarse_feature_maps, dim=1)

        feature_channels, feature_height, feature_width = curr_feature_map.shape[1:]

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width) \
            .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(cam_params_list.device)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)

        cam_points = (uv.unsqueeze(3) * depths).view(batch_size, 1, 3, -1)  # (B, 1, 3, D*FH*FW)
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

        if not isTest:
            preds["world_points"] = world_points

        num_world_points = world_points.size(-1)
        assert num_world_points == feature_height * feature_width * num_depth

        point_features, _, _ = self.feature_fetcher(feature_list, world_points, cam_intrinsic, cam_extrinsic)
        ref_feature_map = coarse_feature_maps[0]

        ref_feature = ref_feature_map.unsqueeze(2).expand(-1, -1, num_depth, -1, -1) \
            .contiguous().view(batch_size, feature_channels, -1)
        point_features[:, 0, :, :] = ref_feature
        avg_point_features = torch.mean(point_features, dim=1)
        avg_point_features2 = torch.mean(point_features ** 2, dim=1)

        point_features = avg_point_features2 - (avg_point_features ** 2)

        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height, feature_width)

        filtered_cost_volume = self.cost_regularization_net(cost_volume).squeeze(1)

        probability_volume = F.softmax(-filtered_cost_volume, dim=1)

        depth_volume = depths.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape)
        pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW)

        preds["coarse_depth_map"] = pred_depth_img
        prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval)
        preds["coarse_prob_map"] = prob_map

        if self.refine:
            norm_pred_depth = (pred_depth_img - depth_start.view(batch_size, 1, 1, 1)) \
                              / (depth_end.view(batch_size, 1, 1, 1) - depth_start.view(batch_size, 1, 1, 1))
            ref_img = img_list[:, 0, :, :, :]
            ref_img = F.interpolate(ref_img, (feature_height, feature_width))
            refined_depth = self.refine_net(ref_img, norm_pred_depth)
            refined_depth = (refined_depth * (depth_end.view(batch_size, 1, 1, 1) -
                                              depth_start.view(batch_size, 1, 1, 1))) \
                            + depth_start.view(batch_size, 1, 1, 1)
            preds["flow1"] = refined_depth

        # loss function
        if "gt_depth_img" in data_batch.keys():
            gt_depth_img = data_batch["gt_depth_img"]
            resized_gt_depth = F.interpolate(gt_depth_img, (pred_depth_img.shape[2], pred_depth_img.shape[3]))
            coarse_loss = self.maeloss(pred_depth_img, resized_gt_depth, depth_interval)
            preds["cor_loss"] = coarse_loss

            if self.refine:
                refine_loss = self.refine_maeloss(refined_depth, resized_gt_depth, depth_interval)
                preds["ref_loss"] = refine_loss

        return preds


def build_mvsnet(cfg):
    torch.backends.cudnn.benchmark = True
    net = MVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
        inverse_depth=cfg.MODEL.INVERSE_DEPTH,
        refine=cfg.MODEL.MVSNET_REFINE
    )

    return net
