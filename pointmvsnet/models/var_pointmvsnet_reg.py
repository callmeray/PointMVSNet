import collections
import os.path as osp
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, osp.dirname(__file__) + '/..')

from pointmvsnet.functions.functions import get_pixel_grids, get_propability_map
from pointmvsnet.utils.torch_utils import get_flat_nn
from pointmvsnet.utils.feature_fetcher import FeatureFetcher
from pointmvsnet.modules.networks import ImageConv, VolumeConv, EdgeConv, EdgeConvNoC, MAELoss, Valid_MAELoss
from pointmvsnet.modules.networks_gn import EdgeConvGN, EdgeConvNoCGN, ImageConvGN, VolumeConvGN
from pointmvsnet.nn.mlp import SharedMLP
from pointmvsnet.nn_gn.mlp import SharedMLP as SharedMLPGN


class VarPointMVSNetReg(nn.Module):
    def __init__(self,
                 img_base_channels=8,
                 vol_base_channels=8,
                 edge_channels=(32, 32, 64),
                 flow_channels=(64, 64, 16, 1),
                 k=16,
                 use_gn=False,
                 num_group=4,
                 inverse_depth=False,
                 masked_loss=True,
                 end2end=True,
                 valid_threshold=2.0,
                 ):
        super(VarPointMVSNetReg, self).__init__()
        self.k = k
        self.inverse_depth = inverse_depth
        self.end2end = end2end
        self.masked_loss = masked_loss

        self.feature_fetcher = FeatureFetcher()

        if use_gn:
            self.coarse_img_conv = ImageConvGN(img_base_channels, num_group)
        else:
            self.coarse_img_conv = ImageConv(img_base_channels)

        image_feature_channels = self.coarse_img_conv.out_channels

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
            d = torch.linspace(1, num_depth, num_depth, device=img_list.device).view(1, 1, num_depth, 1).float() + 1e-6
            d = torch.ones_like(d, device=img_list.device) * min_depth * num_depth / d
            depths = d.unsqueeze(0).expand(batch_size, -1, -1, -1, -1)
        else:
            depth_start = cam_params_list[:, 0, 1, 3, 0]
            depth_interval = cam_params_list[:, 0, 1, 3, 1]
            num_depth = cam_params_list[0, 0, 1, 3, 2].long()
            depth_end = depth_start + (num_depth - 1) * depth_interval
            depths = []
            for i in range(batch_size):
                depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=img_list.device) \
                              .view(1, 1, num_depth, 1))
            depths = torch.stack(depths, dim=0)  # (B, 1, 1, D, 1)

        coarse_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :].contiguous()
            curr_feature_map = self.coarse_img_conv(curr_img)["conv3"]
            coarse_feature_maps.append(curr_feature_map)

        feature_list = torch.stack(coarse_feature_maps, dim=1)

        feature_channels, feature_height, feature_width = curr_feature_map.shape[1:]

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width) \
            .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

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

        ref_feature = ref_feature_map.unsqueeze(2).expand(-1, -1, num_depth, -1, -1) \
            .contiguous().view(batch_size, feature_channels, -1)
        point_features[:, 0, :, :] = ref_feature
        avg_point_features = torch.mean(point_features, dim=1)
        avg_point_features2 = torch.mean(point_features ** 2, dim=1)

        point_features = avg_point_features2 - (avg_point_features ** 2)
        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height, feature_width)

        filtered_cost_volume = self.coarse_vol_conv(cost_volume).squeeze(1)

        probability_volume = F.softmax(-filtered_cost_volume, dim=1)
        depth_volume = []
        for i in range(batch_size):
            if self.inverse_depth:
                depth_array = torch.linspace(1, num_depth, num_depth, device=img_list.device).float()
                depth_array = min_depth * num_depth / depth_array
            else:
                depth_array = torch.linspace(depth_start[i], depth_end[i], num_depth, device=img_list.device)
            depth_volume.append(depth_array)
        depth_volume = torch.stack(depth_volume, dim=0)  # (B, D)
        depth_volume = depth_volume.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape)
        pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW)

        preds["coarse_depth_map"] = pred_depth_img

        prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval)

        preds["coarse_prob_map"] = prob_map

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
                for conv in self.chosen_conv:
                    feature_pyramids[conv] = torch.detach(feature_pyramids[conv])

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
                    .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

                uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1),
                                  feature_map_indices_grid)  # (B, 1, 3, FH*FW)

                interval_feature_collection = []
                cam_points = (uv * estimated_depth_map.view(batch_size, 1, 1, -1))
                world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                    .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                point_features = []
                for conv in self.chosen_conv:
                    curr_feature = feature_pyramids[conv]
                    c, h, w = list(curr_feature.size())[2:]
                    curr_feature = curr_feature.contiguous().view(-1, c, h, w)
                    curr_feature = F.interpolate(curr_feature, (flow_height, flow_width), mode="bilinear")
                    curr_feature = curr_feature.contiguous().view(batch_size, num_view, c, flow_height, flow_width)
                    point_features.append(
                        self.feature_fetcher(curr_feature, world_points, cam_intrinsic, cam_extrinsic)[0])
                point_features = torch.cat(point_features, dim=2)

                avg_point_features = torch.mean(point_features, dim=1)
                avg_point_features_2 = torch.mean(point_features ** 2, dim=1)
                point_features = avg_point_features_2 - (avg_point_features ** 2)

                interval_feature_collection.append(point_features)

                xyz = norm(world_points)
                xyz_feature = xyz.repeat(1, 8, 1)

                interval_feature_collection.append(xyz_feature)
                interval_feature = torch.cat(interval_feature_collection, dim=1)

                xyz_collection.append(xyz)
                feature_collection.append(interval_feature)

                feature = torch.stack(feature_collection, dim=2)
                xyz = torch.stack(xyz_collection, dim=2)
                xyz = xyz.contiguous().view(batch_size, 3, 1, flow_height, flow_width)

                if isTest:
                    def cal_sub_flow(xyz, feature, sub_flow_height, sub_flow_width):
                        nn_idx = get_flat_nn(xyz, knn=self.k)
                        feature = feature.contiguous().view(batch_size, -1, sub_flow_height * sub_flow_width)

                        edge_features = []
                        for edge_conv in self.flow_edge_conv:
                            edge = edge_conv(feature, nn_idx)
                            edge_features.append(edge)
                            feature = edge

                        edge_features = torch.cat(edge_features, dim=1)

                        flow = self.flow_out(self.flow_mlp(edge_features))
                        flow = flow.contiguous().view(batch_size, 1, sub_flow_height, sub_flow_width)

                        return flow

                    if img_scale in [0.125]:
                        flow = cal_sub_flow(xyz, feature, flow_height, flow_width)
                        flow_result = estimated_depth_map + flow

                    elif img_scale in [0.25, 0.5, 1.0]:
                        ratio = int(img_scale * 8)
                        sub_flow_height = flow_height // ratio
                        sub_flow_width = flow_width // ratio
                        feature = feature.view(batch_size, -1, 1, sub_flow_height, ratio,
                                               sub_flow_width, ratio)
                        xyz = xyz.view(batch_size, 3, 1, sub_flow_height, ratio,
                                       sub_flow_width, ratio)

                        flow = []
                        for i in range(ratio):
                            flow_i = []
                            for j in range(ratio):
                                feature_ij = feature[:, :, :, :, i, :, j]
                                xyz_ij = xyz[:, :, :, :, i, :, j]
                                flow_ij = cal_sub_flow(xyz_ij, feature_ij, sub_flow_height,
                                                       sub_flow_width)
                                flow_i.append(flow_ij)
                            flow_i = torch.stack(flow_i, dim=4)
                            flow.append(flow_i)
                        flow = torch.stack(flow, dim=3)

                        flow = flow.contiguous().view(batch_size, 1, flow_height, flow_width)
                        flow_result = estimated_depth_map + flow
                    else:
                        raise NotImplementedError

                else:
                    nn_idx = get_flat_nn(xyz, knn=self.k)
                    feature = feature.contiguous().view(batch_size, -1, 1 * flow_height * flow_width)

                    edge_features = []
                    for edge_conv in self.flow_edge_conv:
                        edge = edge_conv(feature, nn_idx)
                        edge_features.append(edge)
                        feature = edge

                    edge_features = torch.cat(edge_features, dim=1)

                    flow = self.flow_out(self.flow_mlp(edge_features))
                    flow = flow.contiguous().view(batch_size, 1, flow_height, flow_width)

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

        return preds


def build_var_pointmvsnet_reg(cfg):
    torch.backends.cudnn.benchmark = True
    net = VarPointMVSNetReg(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
        edge_channels=cfg.MODEL.EDGE_CHANNELS,
        flow_channels=cfg.MODEL.FLOW_CHANNELS,
        use_gn=cfg.MODEL.GROUP_NORM,
        num_group=cfg.MODEL.NUM_GROUP,
        inverse_depth=cfg.MODEL.INVERSE_DEPTH,
        masked_loss=cfg.MODEL.MASKED_LOSS,
        end2end=cfg.MODEL.END2END,
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    return net
