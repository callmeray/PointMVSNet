import numpy as np
import collections

import torch
import torch.nn as nn
import torch.nn.functional as F

from pointmvsnet.networks import *
from pointmvsnet.functions.functions import get_pixel_grids, get_propability_map
from pointmvsnet.utils.feature_fetcher import FeatureFetcher
from pointmvsnet.utils.torch_utils import get_knn_3d
from pointmvsnet.nn.mlp import SharedMLP


class PointMVSNet(nn.Module):
    def __init__(self,
                 img_base_channels=8,
                 vol_base_channels=8,
                 flow_channels=(64, 64, 16, 1),
                 k=16,
                 ):
        super(PointMVSNet, self).__init__()
        self.k = k

        self.feature_fetcher = FeatureFetcher()

        self.coarse_img_conv = ImageConv(img_base_channels)
        self.coarse_vol_conv = VolumeConv(self.coarse_img_conv.out_channels, vol_base_channels)

        self.flow_img_conv = ImageConv(img_base_channels)
        self.flow_edge_conv = nn.ModuleList()
        self.flow_edge_conv.append(
            EdgeConvNoC(136, 32))
        self.flow_edge_conv.append(
            EdgeConv(32, 32)
        )
        self.flow_edge_conv.append(
            EdgeConv(64, 64)
        )
        self.flow_mlp = nn.Sequential(
            SharedMLP(32 + 32 * 2 + 64 * 2, flow_channels[:-1]),
            nn.Conv1d(flow_channels[-2], flow_channels[-1], 1, bias=False),
        )

    def forward(self, data_batch, img_scales, inter_scales, isFlow, isTest=False):
        def norm(pts):
            norm_pts = (pts - data_batch["mean"].unsqueeze(-1)) / data_batch["std"].unsqueeze(-1)
            return norm_pts

        preds = collections.OrderedDict()
        img_list = data_batch["img_list"]
        cam_params_list = data_batch["cam_params_list"]

        cam_extrinsic = cam_params_list[:, :, 0, :3, :4].clone()  # (B, V, 3, 4)
        R = cam_extrinsic[:, :, :3, :3]
        t = cam_extrinsic[:, :, :3, 3].unsqueeze(-1)
        R_inv = torch.inverse(R)
        cam_intrinsic = cam_params_list[:, :, 1, :3, :3].clone()
        cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 2.0
        if isTest:
            cam_intrinsic[:, :, :2, :3] = cam_intrinsic[:, :, :2, :3] / 4.0

        depth_start = cam_params_list[:, 0, 1, 3, 0]
        depth_interval = cam_params_list[:, 0, 1, 3, 1]
        num_depth = cam_params_list[0, 0, 1, 3, 2].long()

        depth_end = depth_start + (num_depth - 1) * depth_interval

        batch_size, num_view, img_channel, img_height, img_width = list(img_list.size())

        coarse_feature_maps = []
        for i in range(num_view):
            curr_img = img_list[:, i, :, :, :]
            curr_feature_map = self.coarse_img_conv(curr_img)["conv3"]
            coarse_feature_maps.append(curr_feature_map)

        feature_list = torch.stack(coarse_feature_maps, dim=1)

        feature_channels, feature_height, feature_width = list(curr_feature_map.size())[1:]

        depths = []
        for i in range(batch_size):
            depths.append(torch.linspace(depth_start[i], depth_end[i], num_depth, device=img_list.device) \
                          .view(1, 1, num_depth, 1))
        depths = torch.stack(depths, dim=0)  # (B, 1, 1, D, 1)

        feature_map_indices_grid = get_pixel_grids(feature_height, feature_width) \
            .view(1, 1, 3, -1).expand(batch_size, 1, 3, -1).to(img_list.device)

        ref_cam_intrinsic = cam_intrinsic[:, 0, :, :].clone()
        uv = torch.matmul(torch.inverse(ref_cam_intrinsic).unsqueeze(1), feature_map_indices_grid)  # (B, 1, 3, FH*FW)

        cam_points = (uv.unsqueeze(3) * depths).view(batch_size, 1, 3, -1)  # (B, 1, 3, D*FH*FW)
        world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2).contiguous() \
            .view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

        preds["world_points"] = world_points

        num_world_points = world_points.size(-1)
        assert num_world_points == feature_height * feature_width * num_depth

        point_features = self.feature_fetcher(feature_list, world_points, cam_intrinsic, cam_extrinsic)
        ref_feature = coarse_feature_maps[0]
        ref_feature = ref_feature.unsqueeze(2).expand(-1, -1, num_depth, -1, -1)\
                        .contiguous().view(batch_size,feature_channels,-1)
        point_features[:, 0, :, :] = ref_feature

        avg_point_features = torch.mean(point_features, dim=1)
        avg_point_features_2 = torch.mean(point_features ** 2, dim=1)

        point_features = avg_point_features_2 - (avg_point_features ** 2)

        cost_volume = point_features.view(batch_size, feature_channels, num_depth, feature_height, feature_width)

        filtered_cost_volume = self.coarse_vol_conv(cost_volume).squeeze(1)

        probability_volume = F.softmax(-filtered_cost_volume, dim=1)
        depth_volume = []
        for i in range(batch_size):
            depth_array = torch.linspace(depth_start[i], depth_end[i], num_depth, device=depth_start.device)
            depth_volume.append(depth_array)
        depth_volume = torch.stack(depth_volume, dim=0)  # (B, D)
        depth_volume = depth_volume.view(batch_size, num_depth, 1, 1).expand(probability_volume.shape)
        pred_depth_img = torch.sum(depth_volume * probability_volume, dim=1).unsqueeze(1)  # (B, 1, FH, FW)

        preds["coarse_depth_map"] = pred_depth_img

        prob_map = get_propability_map(probability_volume, pred_depth_img, depth_start, depth_interval)

        preds["coarse_prob_map"] = prob_map

        if isFlow:
            feature_pyramids = {}
            chosen_conv = ["conv1", "conv2", "conv3"]
            for conv in chosen_conv:
                feature_pyramids[conv] = []
            for i in range(num_view):
                curr_img = img_list[:, i, :, :, :]
                curr_feature_pyramid = self.flow_img_conv(curr_img)
                for conv in chosen_conv:
                    feature_pyramids[conv].append(curr_feature_pyramid[conv])

            for conv in chosen_conv:
                feature_pyramids[conv] = torch.stack(feature_pyramids[conv], dim=1)

            if isTest:
                for conv in chosen_conv:
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

                interval_list = [-2, -1, 0, 1, 2]
                for i in interval_list:
                    interval_depth_map = estimated_depth_map + interval.view(-1, 1, 1, 1) * i
                    interval_feature_collection = []
                    cam_points = (uv * interval_depth_map.view(batch_size, 1, 1, -1))
                    world_points = torch.matmul(R_inv[:, 0:1, :, :], cam_points - t[:, 0:1, :, :]).transpose(1, 2) \
                        .contiguous().view(batch_size, 3, -1)  # (B, 3, D*FH*FW)

                    for conv in chosen_conv:
                        curr_feature = feature_pyramids[conv]
                        c, h, w = list(curr_feature.size())[2:]
                        curr_feature = curr_feature.contiguous().view(-1, c, h, w)
                        curr_feature = F.interpolate(curr_feature, (flow_height, flow_width), mode="bilinear")
                        curr_feature = curr_feature.contiguous().view(batch_size, num_view, c, flow_height, flow_width)

                        point_features = self.feature_fetcher(curr_feature, world_points, cam_intrinsic, cam_extrinsic)
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

                        flow = self.flow_mlp(edge_features)
                        flow = flow.contiguous().view(batch_size, len(interval_list), sub_flow_height, sub_flow_width)
                        flow_prob = F.softmax(-flow, dim=1)

                        flow_length = torch.tensor(interval_list).float().to(img_list.device)
                        flow_length = flow_length.contiguous().view(1, -1, 1, 1) * interval.view(-1, 1, 1, 1)

                        flow = torch.sum(flow_prob * flow_length, dim=1, keepdim=True)

                        return flow, flow_prob

                    if img_scale in [0.125]:
                        flow, flow_prob = cal_sub_flow(xyz, feature, flow_height, flow_width)
                        preds["flow{}_prob".format(it+1)] = flow_prob
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
                                flow_ij, flow_prob_ij = cal_sub_flow(xyz_ij, feature_ij, sub_flow_height, sub_flow_width)
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
                        preds["flow{}_prob".format(it+1)] = flow_prob
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

                    flow = self.flow_mlp(edge_features)
                    flow = flow.contiguous().view(batch_size, len(interval_list), flow_height, flow_width)
                    flow_prob = F.softmax(-flow, dim=1)
                    preds["flow{}_prob".format(it+1)] = flow_prob

                    flow_length = torch.tensor(interval_list).float().to(img_list.device)
                    flow_length = flow_length.contiguous().view(1, -1, 1, 1) * interval.view(-1, 1, 1, 1)

                    flow = torch.sum(flow_prob * flow_length, dim=1, keepdim=True)

                    flow_result = estimated_depth_map + flow

                return flow_result

            for i, (img_scale, inter_scale) in enumerate(zip(img_scales, inter_scales)):
                if isTest:
                    pred_depth_img = torch.detach(pred_depth_img)
                    print("flow: {}".format(i))
                flow = point_flow(pred_depth_img, inter_scale* depth_interval, img_scale, i)
                preds["flow{}".format(i+1)] = flow
                pred_depth_img = flow

        return preds


class PointMVSNetLoss(nn.Module):
    def __init__(self, valid_threshold):
        super(PointMVSNetLoss, self).__init__()
        self.maeloss = MAELoss()
        self.valid_maeloss = Valid_MAELoss(valid_threshold)

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        coarse_loss = self.maeloss(coarse_depth_map, resize_gt_depth, depth_interval)

        losses = {}
        losses["coarse_loss"] = coarse_loss

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
            flow1_loss = self.maeloss(flow1, resize_gt_depth, 0.75 * depth_interval)
            losses["flow1_loss"] = flow1_loss

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
            flow2_loss = self.maeloss(flow2, resize_gt_depth, 0.375 * depth_interval)
            losses["flow2_loss"] = flow2_loss

        for k in losses.keys():
            losses[k] /= float(len(losses.keys()))

        return losses


def cal_less_percentage(pred_depth, gt_depth, depth_interval, threshold):
    shape = list(pred_depth.size())
    mask_valid = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    denom = torch.sum(mask_valid) + 1e-7
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


def cal_valid_less_percentage(pred_depth, gt_depth, before_depth, depth_interval, threshold, valid_threshold):
    shape = list(pred_depth.size())
    mask_true = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    interval_image = depth_interval.view(-1, 1, 1, 1).expand(shape)
    abs_diff_image = torch.abs(pred_depth - gt_depth) / interval_image

    if before_depth.size(2) != shape[2]:
        before_depth = F.interpolate(before_depth, (shape[2], shape[3]))

    diff = torch.abs(before_depth - gt_depth) / interval_image
    mask_valid = (diff < valid_threshold).type(torch.float)
    mask_valid = mask_valid * mask_true

    denom = torch.sum(mask_valid) + 1e-7
    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


class PointMVSNetMetric(nn.Module):
    def __init__(self, valid_threshold):
        super(PointMVSNetMetric, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, preds, labels, isFlow):
        gt_depth_img = labels["gt_depth_img"]
        depth_interval = labels["cam_params_list"][:, 0, 1, 3, 1]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))

        less_one_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 1.0)
        less_three_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, depth_interval, 3.0)

        metrics = {
            "<1_pct_cor": less_one_pct_coarse,
            "<3_pct_cor": less_three_pct_coarse,
        }

        if isFlow:
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))

            less_one_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                           0.75 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow1 = cal_valid_less_percentage(flow1, resize_gt_depth, coarse_depth_map,
                                                             0.75 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow1"] = less_one_pct_flow1
            metrics["<3_pct_flow1"] = less_three_pct_flow1

            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))

            less_one_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                           0.375 * depth_interval, 1.0, self.valid_threshold)
            less_three_pct_flow2 = cal_valid_less_percentage(flow2, resize_gt_depth, flow1,
                                                             0.375 * depth_interval, 3.0, self.valid_threshold)

            metrics["<1_pct_flow2"] = less_one_pct_flow2
            metrics["<3_pct_flow2"] = less_three_pct_flow2

        return metrics


def build_pointmvsnet(cfg):
    net = PointMVSNet(
        img_base_channels=cfg.MODEL.IMG_BASE_CHANNELS,
        vol_base_channels=cfg.MODEL.VOL_BASE_CHANNELS,
        flow_channels=cfg.MODEL.FLOW_CHANNELS,
    )

    loss_fn = PointMVSNetLoss(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    metric_fn = PointMVSNetMetric(
        valid_threshold=cfg.MODEL.VALID_THRESHOLD,
    )

    return net, loss_fn, metric_fn


if __name__ == "__main__":
    batch_size = 2
    in_channels = 3
    height = 512
    width = 640
    num_view = 3
    num_depth = 64
    images = torch.randn(batch_size, num_view, in_channels, height, width)
    cams = torch.randn(batch_size, num_view, 2, 4, 4)
    gt_depth_img = torch.randn(batch_size, 1, height // 4, width // 4)

    pmvs = PointMVSNet()
    loss_fn = PointMVSNetLoss()
    metric_fn = PointMVSNetMetric()

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
        print("Point-MVSNet: {}: {}".format(k, v.size()))

    loss = loss_fn(preds, data_batch)
    print("Point-MVSNet Loss: {}".format(loss))

    metric = metric_fn(preds, data_batch)
    for k, v in metric.items():
        print("Point-MVSNet Metric: {}: {}".format(k, v))
