import torch
import torch.nn as nn
import torch.nn.functional as F

from pointmvsnet.functions.gather_knn import gather_knn
from pointmvsnet.nn.conv import *


class EdgeConv(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(2 * out_channels)

    def forward(self, feature, knn_inds):
        batch_size, _, num_points = feature.shape
        k = knn_inds.shape[2]

        local_feature = self.conv1(feature)  # (batch_size, out_channels, num_points)
        edge_feature = self.conv2(feature)  # (batch_size, out_channels, num_points)
        channels = local_feature.shape[1]

        if feature.is_cuda:
            # custom improved gather
            neighbour_feature = gather_knn(edge_feature, knn_inds)
        else:
            # pytorch gather
            knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_points, k)
            edge_feature_expand = local_feature.unsqueeze(2).expand(batch_size, -1, num_points, num_points)
            neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)

        # (batch_size, out_channels, num_points, k)

        central_feature = local_feature.unsqueeze(-1).expand(-1, -1, -1, k)

        edge_feature = torch.cat([central_feature, neighbour_feature - central_feature], dim=1)

        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)
        edge_feature = torch.mean(edge_feature, dim=3)

        return edge_feature


class EdgeConvNoC(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(EdgeConvNoC, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, feature, knn_inds):
        batch_size, _, num_points = feature.shape
        k = knn_inds.shape[2]

        local_feature = self.conv1(feature)  # (batch_size, out_channels, num_points)
        edge_feature = self.conv2(feature)  # (batch_size, out_channels, num_points)
        channels = local_feature.shape[1]

        if feature.is_cuda:
            # custom improved gather
            neighbour_feature = gather_knn(edge_feature, knn_inds)
        else:
            # pytorch gather
            knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_points, k)
            edge_feature_expand = edge_feature.unsqueeze(2).expand(batch_size, -1, num_points, num_points)
            neighbour_feature = torch.gather(edge_feature_expand, 3, knn_inds_expand)

        # (batch_size, out_channels, num_points, k)
        central_feature = local_feature.unsqueeze(-1).expand(-1, -1, -1, k)

        edge_feature = neighbour_feature - central_feature
        edge_feature = self.bn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)
        edge_feature = torch.mean(edge_feature, dim=3)

        return edge_feature


class ImageConv(nn.Module):
    def __init__(self, base_channels):
        super(ImageConv, self).__init__()
        self.base_channels = base_channels
        self.out_channels = 8 * base_channels
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
            Conv2d(base_channels * 4, base_channels * 4, 3, 1, padding=1),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2),
            Conv2d(base_channels * 8, base_channels * 8, 3, 1, padding=1),
            nn.Conv2d(base_channels * 8, base_channels * 8, 3, padding=1, bias=False)
        )

    def forward(self, imgs):
        out_dict = {}

        conv0 = self.conv0(imgs)
        out_dict["conv0"] = conv0
        conv1 = self.conv1(conv0)
        out_dict["conv1"] = conv1
        conv2 = self.conv2(conv1)
        out_dict["conv2"] = conv2
        conv3 = self.conv3(conv2)
        out_dict["conv3"] = conv3

        return out_dict


class VolumeConv(nn.Module):
    def __init__(self, in_channels, base_channels):
        super(VolumeConv, self).__init__()
        self.in_channels = in_channels
        self.out_channels = base_channels * 8
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


class MAELoss(nn.Module):
    def forward(self, pred_depth_image, gt_depth_image, depth_interval):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        depth_interval = depth_interval.view(-1)
        mask_valid = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae


class Valid_MAELoss(nn.Module):
    def __init__(self, valid_threshold=2.0):
        super(Valid_MAELoss, self).__init__()
        self.valid_threshold = valid_threshold

    def forward(self, pred_depth_image, gt_depth_image, depth_interval, before_depth_image):
        """non zero mean absolute loss for one batch"""
        # shape = list(pred_depth_image)
        pred_height = pred_depth_image.size(2)
        pred_width = pred_depth_image.size(3)
        depth_interval = depth_interval.view(-1)
        mask_true = (~torch.eq(gt_depth_image, 0.0)).type(torch.float)
        before_hight = before_depth_image.size(2)
        if before_hight != pred_height:
            before_depth_image = F.interpolate(before_depth_image, (pred_height, pred_width))
        diff = torch.abs(gt_depth_image - before_depth_image) / depth_interval.view(-1, 1, 1, 1)
        mask_valid = (diff < self.valid_threshold).type(torch.float)
        mask_valid = mask_true * mask_valid
        denom = torch.sum(mask_valid, dim=(1, 2, 3)) + 1e-7
        masked_abs_error = mask_valid * torch.abs(pred_depth_image - gt_depth_image)
        masked_mae = torch.sum(masked_abs_error, dim=(1, 2, 3))
        masked_mae = torch.sum((masked_mae / depth_interval) / denom)

        return masked_mae
