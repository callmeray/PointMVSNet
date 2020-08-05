import collections
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__) + '/..')

from pointmvsnet.functions.functions import get_pixel_grids, get_propability_map
from pointmvsnet.utils.torch_utils import get_knn_3d
from pointmvsnet.utils.feature_fetcher import FeatureFetcher
from pointmvsnet.modules.networks import ImageConv, VolumeConv, EdgeConv, EdgeConvNoC, MAELoss, Valid_MAELoss
from pointmvsnet.modules.networks_gn import EdgeConvGN, EdgeConvNoCGN, ImageConvGN, VolumeConvGN
from pointmvsnet.nn.mlp import SharedMLP
from pointmvsnet.nn_gn.mlp import SharedMLP as SharedMLPGN


class VarPointMVSNet(nn.Module):
    def __init__(self,
                 img_base_channels=8,
                 vol_base_channels=8,
                 edge_channels=(32, 32, 64),
                 flow_channels=(64, 64, 16, 1),
                 k=16,
                 vis_model="var",
                 occ_depth_channels=0,
                 occ_shared_channels=(128, 128, 128),
                 occ_global_channels=(64, 16, 4),
                 use_gn=False,
                 num_group=4,
                 inverse_depth=False,
                 masked_loss=True,
                 end2end=True,
                 valid_threshold=2.0,
                 occ_interpolation="nearest"):
        super(VarPointMVSNet, self).__init__()
        self.k = k
        self.inverse_depth = inverse_depth
        self.end2end = end2end
        self.masked_loss = masked_loss
        self.occ_interpolation = occ_interpolation
        self.feature_fetcher = FeatureFetcher()

        if use_gn:
            self.coarse_img_conv = ImageConvGN(img_base_channels, num_group)
        else:
            self.coarse_img_conv = ImageConv(img_base_channels)

        image_feature_channels = self.coarse_img_conv.out_channels

        assert vis_model in ["var", "vis-var", "vis-var-gt"]
        self.vis_model = vis_model
        if vis_model == "vis-var":
            self.occ_depth_channels = occ_depth_channels
            if use_gn:
                self.occ_shared_mlp = SharedMLPGN(2 * image_feature_channels + occ_depth_channels, occ_shared_channels,
                                                  ndim=2,
                                                  num_group=num_group)
                self.occ_global_mlp = SharedMLPGN(occ_shared_channels[-1], occ_global_channels,
                                                  num_group=num_group)
                self.occ_pred = nn.Conv1d(occ_global_channels[-1], 1, 1)
            else:
                self.occ_shared_mlp = SharedMLP(2 * image_feature_channels + occ_depth_channels, occ_shared_channels,
                                                ndim=2)
                self.occ_global_mlp = SharedMLP(occ_shared_channels[-1], occ_global_channels)
                self.occ_pred = nn.Conv1d(occ_global_channels[-1], 1, 1)

        if use_gn:
            self.coarse_vol_conv = VolumeConvGN(image_feature_channels, vol_base_channels, num_group)
        else:
            self.coarse_vol_conv = VolumeConv(image_feature_channels, vol_base_channels)

        self.chosen_conv = ["conv1", "conv2", "conv3"]
        multiple = 0
        if "conv0" in self.chosen_conv:
            multiple += 1
        if "conv1" in self.chosen_conv:
            multiple += 2
        if "conv2" in self.chosen_conv:
            multiple += 4
        if "conv3" in self.chosen_conv:
            multiple += 8

        if use_gn:
            self.flow_img_conv = ImageConvGN(img_base_channels, num_group)
            self.flow_edge_conv = nn.ModuleList()
            edge_conv_out_channels = 0
            self.flow_edge_conv.append(
                EdgeConvNoCGN(self.flow_img_conv.base_channels * multiple + 24, edge_channels[0], num_group))
            edge_conv_out_channels += edge_channels[0]

            for k in range(1, len(edge_channels)):
                if k == 1:
                    self.flow_edge_conv.append(
                        EdgeConvGN(edge_channels[k - 1], edge_channels[k], num_group)
                    )
                else:
                    self.flow_edge_conv.append(
                        EdgeConvGN(edge_channels[k - 1] * 2, edge_channels[k], num_group)
                    )
                edge_conv_out_channels += edge_channels[k] * 2
            self.flow_mlp = SharedMLPGN(edge_conv_out_channels, flow_channels[:-1], num_group=num_group)
        else:
            self.flow_img_conv = ImageConv(img_base_channels)
            self.flow_edge_conv = nn.ModuleList()
            edge_conv_out_channels = 0
            self.flow_edge_conv.append(
                EdgeConvNoC(self.flow_img_conv.base_channels * multiple + 24, edge_channels[0]))
            edge_conv_out_channels += edge_channels[0]

            for k in range(1, len(edge_channels)):
                if k == 1:
                    self.flow_edge_conv.append(
                        EdgeConv(edge_channels[k - 1], edge_channels[k])
                    )
                else:
                    self.flow_edge_conv.append(
                        EdgeConv(edge_channels[k - 1] * 2, edge_channels[k])
                    )
                edge_conv_out_channels += edge_channels[k] * 2
            self.flow_mlp = SharedMLP(edge_conv_out_channels, flow_channels[:-1])
        self.flow_out = nn.Conv1d(flow_channels[-2], flow_channels[-1], 1, bias=False)

        self.maeloss = MAELoss()
        self.valid_maeloss = Valid_MAELoss(valid_threshold)
        self._init_weight()

    def _init_weight(self):
        nn.init.zeros_(self.flow_out.weight)
        if self.vis_model == "pred":
            nn.init.zeros_(self.occ_pred.weight)
            nn.init.zeros_(self.occ_pred.bias)

    def forward(self, data_batch, img_scales, inter_scales, use_occ_pred, isFlow, isTest=False):
        def norm(pts):
            norm_pts = (pts - data_batch["mean"].unsqueeze(-1)) / data_batch["std"].unsqueeze(-1)
            return norm_pts

        preds = collections.OrderedDict()
        img_list = data_batch["img_list"]
        cam_params_list = data_batch["cam_params_list"]
        batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()
        cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 2.0
        if isTest:
            cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 4.0

        if self.inverse_depth:
            min_depth = cam_params_list[0, 0, 1, 3, 0]
            depth_interval = cam_params_list[:, 0, 1, 3, 1]
            num_depth = cam_params_list[0, 0, 1, 3, 2].long()
            d = torch.linspace(1, num_depth, num_depth, device=cam_params_list.device).view(1, 1, num_depth,
                                                                                            1).float() + 1e-6
            d = torch.ones_like(d, device=cam_params_list.device) * min_depth * num_depth / d
            depths = d.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
            depth_min = min_depth
            depth_max = min_depth * num_depth
        else:
            depth_start = cam_params_list[:, 0, 1, 3, 0]
            depth_interval = cam_params_list[:, 0, 1, 3, 1]
            num_depth = cam_params_list[0, 0, 1, 3, 2].long()
            depth_end = depth_start + (num_depth - 1) * depth_interval
            depths = []
            for i in range(batch_size):
                depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=cam_params_list.device) \
                              .view(1, 1, num_depth, 1))
            depths = torch.stack(depths, dim=0)  # (B, 1, 1, D, 1)
            depth_min = depth_start[0]
            depth_max = depth_end[0]

        coarse_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :].contiguous()
            curr_feature_map = self.coarse_img_conv(curr_img)["conv3"]
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
        if not "mean" in data_batch.keys():
            data_batch["mean"] = torch.mean(world_points, dim=-1)
            data_batch["std"] = (torch.max(world_points, dim=-1)[0] - torch.min(world_points, dim=-1)[0]) / 2.0

        if not isTest:
            preds["world_points"] = world_points

        num_world_points = world_points.size(-1)
        assert num_world_points == feature_height * feature_width * num_depth

        point_features, _, z = self.feature_fetcher(feature_list, world_points, cam_intrinsic, cam_extrinsic)

        ref_feature_map = coarse_feature_maps[0]

        if "gt_depth_img" in data_batch.keys() and "depth_list" in data_batch.keys():
            # fetch depths from different views to compute occlusions
            if "highres_depth_list" in data_batch.keys():
                hd_cam_intrinsic = data_batch["highres_cam_params_list"][:, :, 1, :3, :3].clone()
                hd_depth_list = data_batch["highres_depth_list"]
                hd_height, hd_width = hd_depth_list.shape[3:]
                hd_map_indices_grid = get_pixel_grids(hd_height, hd_width) \
                    .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)
                hd_uv = torch.matmul(torch.inverse(hd_cam_intrinsic[:, 0, :, :].contiguous()).unsqueeze(1),
                                     hd_map_indices_grid)  # (B, 1, 3, FH*FW)

                hd_cam_points = (hd_uv.view(batch_size, 3, hd_height, hd_width) * hd_depth_list[:, 0]).view(batch_size,
                                                                                                            1, 3,
                                                                                                            -1)  # (B, 1, 3, D*FH*FW)
                hd_world_points = torch.matmul(R_inv[:, 0:1, :, :], hd_cam_points - t[:, 0:1, :, :]) \
                    .transpose(1, 2).contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                depth_per_view_fetch, _, depth_per_view_proj = self.feature_fetcher(hd_depth_list, hd_world_points,
                                                                                    hd_cam_intrinsic, cam_extrinsic)
                depth_per_view_fetch = depth_per_view_fetch.view(batch_size, num_view, 1, hd_height, hd_width)
                depth_per_view_proj = depth_per_view_proj.view(batch_size, num_view, 1, hd_height, hd_width)

                valid_mask = (depth_per_view_proj > 0.01).float() * (depth_per_view_fetch > 0.01).float() \
                             * ((hd_depth_list[:, 0:1] > 0.01).float().expand(batch_size, num_view, 1, hd_height,
                                                                              hd_width))
                valid_mask = valid_mask[:, 1:].contiguous()
                gt_occ_mask = (torch.abs(depth_per_view_fetch - depth_per_view_proj) > 0.01).float()
                gt_occ_mask = gt_occ_mask[:, 1:].contiguous() * valid_mask  # (B, V-1, 1, H, W)
                gt_noocc_mask = (torch.abs(depth_per_view_fetch - depth_per_view_proj) < 0.01).float()
                gt_noocc_mask = gt_noocc_mask[:, 1:].contiguous() * valid_mask

                preds["occ_gt"] = gt_occ_mask
                preds["noocc_gt"] = gt_noocc_mask
                preds["2view_mask"] = (gt_noocc_mask.sum(1) > 0.5).float()
                preds["nofull_mask"] = (gt_noocc_mask.sum(1) < (num_view - 1.5)).float()
                occ_mask = torch.cat([gt_occ_mask.new_ones((batch_size, 1, 1, hd_height, hd_width)), gt_noocc_mask],
                                     dim=1)
            else:
                coarse_cam_points = uv.view(batch_size, 3, feature_height, feature_width) * \
                                    F.interpolate(data_batch["gt_depth_img"], (feature_height, feature_width))
                coarse_cam_points = coarse_cam_points.view(batch_size, 3, -1)
                coarse_world_points = torch.matmul(R_inv[:, 0, :, :].contiguous(), coarse_cam_points - t[:, 0, :, :]) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, FH*FW)

                depth_list = data_batch["depth_list"].view(batch_size * num_view, 1, 2 * feature_height,
                                                           2 * feature_width)
                depth_list = F.interpolate(depth_list, (feature_height, feature_width)) \
                    .view(batch_size, num_view, 1, feature_height, feature_width)
                depth_per_view_fetch, _, depth_per_view_proj = self.feature_fetcher(depth_list,
                                                                                    coarse_world_points,
                                                                                    cam_intrinsic, cam_extrinsic)

                depth_per_view_fetch = depth_per_view_fetch.view(batch_size, num_view, 1, feature_height, feature_width)
                depth_per_view_proj = depth_per_view_proj.view(batch_size, num_view, 1, feature_height, feature_width)
                preds["depth_per_view_fetch"] = depth_per_view_fetch
                preds["depth_per_view_proj"] = depth_per_view_proj

                valid_mask = (depth_per_view_proj > 0.1).float() * (depth_per_view_fetch > 0.1).float() \
                             * ((depth_list[:, 0] > 0.1).unsqueeze(1).float().expand(batch_size, num_view, 1, -1, -1))
                valid_mask = valid_mask[:, 1:].contiguous()
                gt_occ_mask = (torch.abs(depth_per_view_fetch - depth_per_view_proj) > 0.01).float()
                gt_occ_mask = gt_occ_mask[:, 1:].contiguous() * valid_mask  # (B, V-1, 1, H, W)
                gt_noocc_mask = (torch.abs(depth_per_view_fetch - depth_per_view_proj) < 0.01).float()
                gt_noocc_mask = gt_noocc_mask[:, 1:].contiguous() * valid_mask

                preds["occ_gt"] = gt_occ_mask
                preds["noocc_gt"] = gt_noocc_mask
                preds["2view_mask"] = (gt_noocc_mask.sum(1) > 0.5).float()
                occ_mask = torch.cat([gt_occ_mask.new_ones((batch_size, 1, 1, feature_height, feature_width)),
                                      gt_noocc_mask], dim=1)

        if self.vis_model == "var":
            ref_feature = ref_feature_map.unsqueeze(2).expand(-1, -1, num_depth, -1, -1) \
                .contiguous().view(batch_size, feature_channels, -1)
            point_features[:, 0, :, :] = ref_feature
            avg_point_features = torch.mean(point_features, dim=1)
            avg_point_features2 = torch.mean(point_features ** 2, dim=1)

            point_features = avg_point_features2 - (avg_point_features ** 2)

        else:
            if self.vis_model == "vis-var":
                if isTest:
                    ref_feature_map = ref_feature_map.view(batch_size, feature_channels, 1,
                                                           feature_height * feature_width)
                    if self.occ_depth_channels > 0:
                        norm_depth = (depths - depth_min) / (depth_max - depth_min)
                        norm_depth = norm_depth.expand(1, 1, self.occ_depth_channels, num_depth,
                                                       feature_height * feature_width) \
                            .squeeze(0)
                    global_features = []
                    for b in range(num_view - 1):
                        d_features = []
                        source_feature = point_features[:, b, :].contiguous().view(1, feature_channels, num_depth,
                                                                                   feature_height * feature_width)
                        for d in range(num_depth):
                            if self.occ_depth_channels > 0:
                                cat_feature = torch.cat((ref_feature_map, source_feature[:, :, d:d + 1, :].contiguous(),
                                                         norm_depth[:, :, d:d + 1, :].contiguous()), dim=1)
                            else:
                                cat_feature = torch.cat(
                                    (ref_feature_map, source_feature[:, :, d:d + 1, :].contiguous()),
                                    dim=1)
                            d_features.append(torch.detach(self.occ_shared_mlp(cat_feature)))
                        d_features = torch.cat(d_features, dim=2)
                        g, _ = torch.max(d_features, dim=2)
                        g = self.occ_global_mlp(g)
                        global_features.append(g)
                    global_features = torch.cat(global_features, dim=0)
                    del source_feature
                    del cat_feature
                    torch.cuda.empty_cache()
                else:
                    ref_feature_map = ref_feature_map.unsqueeze(1).unsqueeze(3) \
                        .expand(-1, num_view - 1, -1, num_depth, -1, -1).contiguous() \
                        .view(batch_size * (num_view - 1), feature_channels, num_depth, feature_height * feature_width)
                    source_features = point_features[:, 1:].contiguous() \
                        .view(batch_size * (num_view - 1), feature_channels, num_depth, feature_height * feature_width)

                    if self.occ_depth_channels > 0:
                        norm_depth = (depths - depth_min) / (depth_max - depth_min)
                        norm_depth = norm_depth.expand(batch_size, num_view - 1, self.occ_depth_channels, num_depth,
                                                       feature_height * feature_width).contiguous().view(
                            batch_size * (num_view - 1), self.occ_depth_channels, num_depth,
                            feature_height * feature_width)
                        cat_features = torch.cat((ref_feature_map, source_features, norm_depth), dim=1).contiguous()
                    else:
                        cat_features = torch.cat((ref_feature_map, source_features), dim=1).contiguous()
                    shared_features = self.occ_shared_mlp(cat_features)
                    global_features, _ = torch.max(shared_features, dim=2)
                    global_features = self.occ_global_mlp(global_features)
                occ_pred = self.occ_pred(global_features)
                occ_pred = torch.sigmoid(occ_pred).view(batch_size, num_view - 1, 1, feature_height, feature_width)
                if not isTest:
                    preds["occ_pred"] = occ_pred

                if use_occ_pred:
                    occ_mask = torch.cat([occ_pred.new_ones((batch_size, 1, 1, feature_height, feature_width)),
                                          occ_pred], dim=1)
                    occ_mask_coarse = occ_mask.unsqueeze(3).expand((-1, -1, -1, num_depth, -1, -1)).contiguous() \
                        .view(batch_size, num_view, 1, -1)
                else:
                    occ_mask_coarse = F.interpolate(occ_mask, (1, feature_height, feature_width), mode="trilinear")
                    occ_mask_coarse = occ_mask_coarse.unsqueeze(3).expand((-1, -1, -1, num_depth, -1, -1)).contiguous() \
                        .view(batch_size, num_view, 1, -1)
            else:  #"vis-var-gt"
                occ_mask_coarse = F.interpolate(occ_mask, (1, feature_height, feature_width), mode="trilinear")
                occ_mask_coarse = occ_mask_coarse.unsqueeze(3).expand((-1, -1, -1, num_depth, -1, -1)).contiguous() \
                    .view(batch_size, num_view, 1, -1)

            if isTest:
                del global_features
                del occ_pred
                del d_features
                del g
                torch.cuda.empty_cache()

            point_features = torch.sum((point_features ** 2) * occ_mask_coarse, dim=1) \
                             / torch.sum(occ_mask_coarse, dim=1) - \
                             torch.pow(torch.sum(point_features * occ_mask_coarse, dim=1)
                                       / torch.sum(occ_mask_coarse, dim=1), 2)
            if isTest:
                del occ_mask_coarse
                torch.cuda.empty_cache()

        if "init_depth_map" in data_batch.keys():
            pred_depth_img = data_batch["init_depth_map"]
            prob_map = data_batch["init_prob_map"]
            preds["coarse_depth_map"] = pred_depth_img
            preds["coarse_prob_map"] = prob_map

        else:
            cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height, feature_width)

            filtered_cost_volume = self.coarse_vol_conv(cost_volume).squeeze(1)

            probability_volume = F.softmax(-filtered_cost_volume, dim=1)
            depth_volume = []
            for i in range(batch_size):
                if self.inverse_depth:
                    depth_array = torch.linspace(1, num_depth, num_depth, device=cam_params_list.device).float()
                    depth_array = min_depth * num_depth / depth_array
                else:
                    depth_array = torch.linspace(depth_start[i], depth_end[i], num_depth, device=cam_params_list.device)
                depth_volume.append(depth_array)
            depth_volume = torch.stack(depth_volume, dim=0)  # (B, D)
            depth_volume = depth_volume.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape)
            pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW)

            preds["coarse_depth_map"] = pred_depth_img

            prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval)

            preds["coarse_prob_map"] = prob_map
            if isTest:
                del cost_volume
                del filtered_cost_volume
                del probability_volume
                del depth_volume
                torch.cuda.empty_cache()

        if isFlow:
            feature_pyramids = {}
            for conv in self.chosen_conv:
                feature_pyramids[conv] = []
            for i in range(num_view):
                curr_img = img_list[:, i, :, :, :]
                curr_feature_pyramid = self.flow_img_conv(curr_img)
                for conv in self.chosen_conv:
                    feature_pyramids[conv].append(curr_feature_pyramid[conv])
            for conv in self.chosen_conv:
                feature_pyramids[conv] = torch.stack(feature_pyramids[conv], dim=1)

            if isTest:
                del img_list
                torch.cuda.empty_cache()

            def point_flow(estimated_depth_map, interval, image_scale, it):
                feature_collection = []
                xyz_collection = []
                flow_height, flow_width = list(estimated_depth_map.size())[2:]
                if flow_height != int(img_height * image_scale):
                    flow_height = int(img_height * image_scale)
                    flow_width = int(img_width * image_scale)
                    estimated_depth_map = F.interpolate(estimated_depth_map, (flow_height, flow_width),
                                                        mode="nearest")
                cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()
                if isTest:
                    cam_intrinsic[:, :, :2, :3] *= image_scale
                else:
                    cam_intrinsic[:, :, :2, :3] *= (4 * image_scale)

                ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
                feature_map_indices_grid = get_pixel_grids(flow_height, flow_width) \
                    .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(cam_params_list.device)

                uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                                  feature_map_indices_grid)  # (B, 1, 3, FH*FW)

                interval_list = [-2, -1, 0, 1, 2]
                for i in interval_list:
                    interval_depth_map = estimated_depth_map + interval.view(-1, 1, 1, 1) * i
                    interval_feature_collection = []
                    cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                    world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                        .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                    point_features_collection = []
                    for conv in self.chosen_conv:
                        curr_feature = feature_pyramids[conv]
                        c, h, w = list(curr_feature.size())[2:]
                        curr_feature = curr_feature.contiguous().view(-1, c, h, w)
                        curr_feature = F.interpolate(curr_feature, (flow_height, flow_width), mode="bilinear")
                        curr_feature = curr_feature.contiguous().view(batch_size, num_view, c, flow_height, flow_width)

                        point_features, _, _ = self.feature_fetcher(curr_feature, world_points, cam_intrinsic,
                                                                    cam_extrinsic)
                        point_features_collection.append(point_features)
                    point_features = torch.cat(point_features_collection, dim=2)

                    if self.vis_model == "var":
                        avg_point_features = torch.mean(point_features, dim=1)
                        avg_point_features_2 = torch.mean(point_features ** 2, dim=1)
                        point_features = avg_point_features_2 - (avg_point_features ** 2)
                    else:
                        if self.occ_interpolation == "bilinear":
                            occ_mask_flow = F.interpolate(occ_mask, (1, flow_height, flow_width),
                                                          mode="trilinear").contiguous().view(batch_size, num_view, 1, flow_height*flow_width)
                        else:
                            occ_mask_flow = F.interpolate(occ_mask, (1, flow_height, flow_width),
                                                          mode="nearest").contiguous().view(batch_size, num_view, 1, flow_height*flow_width)
                        avg_point_features = torch.sum(point_features * occ_mask_flow, dim=1) / torch.sum(
                            occ_mask_flow, dim=1)
                        avg_point_features2 = torch.sum((point_features ** 2) * occ_mask_flow, dim=1) / torch.sum(
                            occ_mask_flow, dim=1)

                        point_features = avg_point_features2 - (avg_point_features ** 2)

                    interval_feature_collection.append(point_features)

                    xyz = norm(world_points)
                    xyz_feature = xyz.repeat(1, 8, 1)

                    interval_feature_collection.append(xyz_feature)
                    interval_feature = torch.cat(interval_feature_collection, dim=1)

                    xyz_collection.append(xyz)
                    feature_collection.append(interval_feature)

                feature = torch.stack(feature_collection, dim=2)
                xyz = torch.stack(xyz_collection, dim=2)
                xyz = xyz.contiguous().view(batch_size, 3, len(interval_list), flow_height, flow_width)

                if isTest:
                    def cal_sub_flow(xyz, feature, sub_flow_height, sub_flow_width):
                        nn_idx = get_knn_3d(xyz, len(interval_list), knn=self.k)
                        feature = feature.contiguous().view(batch_size, -1,
                                                            len(interval_list) * sub_flow_height * sub_flow_width)

                        edge_features = []
                        for edge_conv in self.flow_edge_conv:
                            edge = edge_conv(feature, nn_idx)
                            edge_features.append(edge)
                            feature = edge

                        edge_features = torch.cat(edge_features, dim=1)

                        flow = self.flow_out(self.flow_mlp(edge_features))
                        del edge_features
                        del feature
                        torch.cuda.empty_cache()

                        flow = flow.contiguous().view(batch_size, len(interval_list), sub_flow_height, sub_flow_width)
                        flow_prob = F.softmax(-flow, dim=1)

                        flow_length = torch.tensor(interval_list).float().to(cam_params_list.device)
                        flow_length = flow_length.contiguous().view(1, -1, 1, 1) * interval.view(-1, 1, 1, 1)

                        flow = torch.sum(flow_prob * flow_length, dim=1, keepdim=True)

                        return flow, flow_prob

                    if img_scale in [0.125]:
                        flow, flow_prob = cal_sub_flow(xyz, feature, flow_height, flow_width)
                        preds["flow{}_prob".format(it + 1)] = flow_prob
                        flow_result = estimated_depth_map + flow

                    elif img_scale in [0.25, 0.5, 1.0]:
                        ratio = int(img_scale * 8)
                        sub_flow_height = flow_height // ratio
                        sub_flow_width = flow_width // ratio
                        feature = feature.view(batch_size, -1, len(interval_list), sub_flow_height, ratio,
                                               sub_flow_width, ratio)
                        xyz = xyz.view(batch_size, 3, len(interval_list), sub_flow_height, ratio,
                                       sub_flow_width, ratio)

                        flow = []
                        flow_prob = []
                        for i in range(ratio):
                            flow_i = []
                            flow_prob_i = []
                            for j in range(ratio):
                                feature_ij = feature[:, :, :, :, i, :, j]
                                xyz_ij = xyz[:, :, :, :, i, :, j]
                                flow_ij, flow_prob_ij = cal_sub_flow(xyz_ij, feature_ij, sub_flow_height,
                                                                     sub_flow_width)
                                flow_i.append(flow_ij)
                                flow_prob_i.append(flow_prob_ij)
                            flow_i = torch.stack(flow_i, dim=4)
                            flow_prob_i = torch.stack(flow_prob_i, dim=4)
                            flow.append(flow_i)
                            flow_prob.append(flow_prob_i)
                        flow = torch.stack(flow, dim=3)
                        flow_prob = torch.stack(flow_prob, dim=3)

                        flow = flow.contiguous().view(batch_size, 1, flow_height, flow_width)
                        flow_result = estimated_depth_map + flow

                        flow_prob = flow_prob.contiguous().view(batch_size, len(interval_list), flow_height, flow_width)
                        preds["flow{}_prob".format(it + 1)] = flow_prob
                    else:
                        raise NotImplementedError

                else:
                    nn_idx = get_knn_3d(xyz, len(interval_list), knn=self.k)
                    feature = feature.contiguous().view(batch_size, -1, len(interval_list) * flow_height * flow_width)

                    edge_features = []
                    for edge_conv in self.flow_edge_conv:
                        edge = edge_conv(feature, nn_idx)
                        edge_features.append(edge)
                        feature = edge

                    edge_features = torch.cat(edge_features, dim=1)

                    flow = self.flow_out(self.flow_mlp(edge_features))
                    flow = flow.contiguous().view(batch_size, len(interval_list), flow_height, flow_width)
                    flow_prob = F.softmax(-flow, dim=1)
                    preds["flow{}_prob".format(it + 1)] = flow_prob

                    flow_length = torch.tensor(interval_list).float().to(cam_params_list.device)
                    flow_length = flow_length.contiguous().view(1, -1, 1, 1) * interval.view(-1, 1, 1, 1)

                    flow = torch.sum(flow_prob * flow_length, dim=1, keepdim=True)

                    flow_result = estimated_depth_map + flow

                return flow_result

            for i, (img_scale, inter_scale) in enumerate(zip(img_scales, inter_scales)):
                if isTest or (not self.end2end):
                    pred_depth_img = torch.detach(pred_depth_img)
                flow = point_flow(pred_depth_img.clone(), inter_scale * depth_interval, img_scale, i)
                preds["flow{}".format(i + 1)] = flow
                pred_depth_img = flow

        if "gt_depth_img" in data_batch.keys():
            gt_depth_img = data_batch["gt_depth_img"]
            coarse_depth_map = preds["coarse_depth_map"]
            resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
            if self.masked_loss:
                view_mask = F.interpolate(preds["2view_mask"], (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
                resize_gt_depth = resize_gt_depth * view_mask

            coarse_loss = self.maeloss(coarse_depth_map, resize_gt_depth, depth_interval)

            preds["cor_loss"] = coarse_loss
            if torch.sum(torch.isnan(coarse_loss).float()) > 0.0:
                print("Nan in coarse_loss")
                exit()

            if self.vis_model == "pred" and "occ_gt" in preds.keys():
                occ_pred = preds["occ_pred"]
                occ_gt = preds["occ_gt"]
                occ_gt = F.interpolate(occ_gt, (occ_pred.shape[2], occ_pred.shape[3], occ_pred.shape[4]),
                                       mode="trilinear")
                noocc_gt = preds["noocc_gt"]
                noocc_gt = F.interpolate(noocc_gt, (occ_pred.shape[2], occ_pred.shape[3], occ_pred.shape[4]),
                                       mode="trilinear")
                preds["occ_loss"] = ((1 - occ_pred) * occ_gt).sum() / torch.clamp(occ_gt.sum(), 1e-6) + (
                        occ_pred * noocc_gt).sum() / torch.clamp(noocc_gt.sum(), 1e-6)

            if isFlow:
                for i, (_, inter_scale) in enumerate(zip(img_scales, inter_scales)):
                    flow = preds["flow{}".format(i + 1)]
                    resize_gt_depth = F.interpolate(gt_depth_img, (flow.shape[2], flow.shape[3]))
                    if self.masked_loss:
                        view_mask = F.interpolate(preds["2view_mask"], (flow.shape[2], flow.shape[3]))
                        resize_gt_depth = resize_gt_depth * view_mask
                    if "flow_mask_{}".format(i + 1) in preds.keys():
                        mask = preds["flow_mask_{}".format(i + 1)]
                        resize_gt_depth = resize_gt_depth * (mask > 0.5).float()
                    flow_loss = self.valid_maeloss(flow, resize_gt_depth, inter_scale * depth_interval,
                                                   coarse_depth_map)
                    preds["flow{}_loss".format(i + 1)] = flow_loss
                    coarse_depth_map = flow

        if isTest:
            del R
            del R_inv
            del cam_extrinsic
            del cam_intrinsic
            del cam_params_list
            del cam_points
            del coarse_feature_maps
            del curr_feature_map
            del curr_feature_pyramid
            del curr_img
            del data_batch
            del depths
            del feature_list
            del feature_map_indices_grid
            del feature_pyramids
            del flow
            del occ_mask
            del point_features
            del pred_depth_img
            del prob_map
            del ref_cam_intrinsic
            del ref_feature_map
            del t
            del uv
            del world_points
            del z
            torch.cuda.empty_cache()
        return preds


def build_var_pointmvsnet(cfg):
    torch.backends.cudnn.benchmark = True
    net = VarPointMVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
        edge_channels=cfg.MODEL.EDGE_CHANNELS,
        flow_channels=cfg.MODEL.FLOW_CHANNELS,
        vis_model=cfg.MODEL.VIS_MODEL,
        occ_depth_channels=cfg.MODEL.OCC_DEPTH_CHANNELS,
        occ_shared_channels=cfg.MODEL.OCC_SHARED_CHANNELS,
        occ_global_channels=cfg.MODEL.OCC_GLOBAL_CHANNELS,
        use_gn=cfg.MODEL.GROUP_NORM,
        num_group=cfg.MODEL.NUM_GROUP,
        inverse_depth=cfg.MODEL.INVERSE_DEPTH,
        masked_loss=cfg.MODEL.MASKED_LOSS,
        end2end=cfg.MODEL.END2END,
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
        occ_interpolation=cfg.MODEL.OCC_INTERPOLATION,
    )

    return net


def test():
    batch_size = 2
    in_channels = 3
    height = 512
    width = 640
    num_view = 3
    num_depth = 64
    images = torch.randn(batch_size, num_view, in_channels, height, width)
    cams = torch.randn(batch_size, num_view, 2, 4, 4)
    gt_depth_img = torch.randn(batch_size, 1, height // 4, width // 4)

    pmvs = VarPointMVSNet()
    loss_fn = PointMVSNetLoss()
    metric_fn = PointMVSNetMetric(3.0)

    test_GPU = True
    if test_GPU:
        images = images.cuda()
        cams = cams.cuda()
        gt_depth_img = gt_depth_img.cuda()
        pmvs = pmvs.cuda()

    data_batch = {"img_list": images,
                  "cam_params_list": cams,
                  "gt_depth_img": gt_depth_img}

    preds = pmvs(data_batch)

    for k, v in preds.items():
        print("Occ Point MVSNet: {}: {}".format(k, v.size()))

    loss = loss_fn(preds, data_batch)
    print("Occ Point MVSNet Loss: {}".format(loss))

    metric = metric_fn(preds, data_batch)
    for k, v in metric.items():
        print("Occ Point MVSNet Metric: {}: {}".format(k, v))


if __name__ == "__main__":
    test()
