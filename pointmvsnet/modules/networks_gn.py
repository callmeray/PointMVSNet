import torch

from pointmvsnet.functions.gather_knn import gather_knn
from pointmvsnet.nn_gn.conv import *


class EdgeConvGN(nn.Module):

    def __init__(self, in_channels, out_channels, num_group=4):
        super(EdgeConvGN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.gn = nn.GroupNorm(num_group, 2 * out_channels)

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

        edge_feature = self.gn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)
        edge_feature = torch.mean(edge_feature, dim=3)

        return edge_feature


class EdgeConvNoCGN(nn.Module):
    def __init__(self, in_channels, out_channels, num_group=4):
        super(EdgeConvNoCGN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, 1, bias=False)
        self.conv2 = nn.Conv1d(in_channels, out_channels, 1, bias=False)

        self.gn = nn.GroupNorm(num_group, out_channels)

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
        edge_feature = self.gn(edge_feature)
        edge_feature = F.relu(edge_feature, inplace=True)
        edge_feature = torch.mean(edge_feature, dim=3)

        return edge_feature


class ImageConvGN(nn.Module):
    def __init__(self, base_channels, num_group=4):
        super(ImageConvGN, self).__init__()
        self.base_channels = base_channels
        self.out_channels = 8 * base_channels
        self.conv0 = nn.Sequential(
            Conv2d(3, base_channels, 3, padding=1, num_group=num_group),
            Conv2d(base_channels, base_channels, 3, padding=1, num_group=num_group)
        )

        self.conv1 = nn.Sequential(
            Conv2d(base_channels, base_channels * 2, 5, stride=2, padding=2, num_group=num_group),
            Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, num_group=num_group),
            Conv2d(base_channels * 2, base_channels * 2, 3, padding=1, num_group=num_group),
        )

        self.conv2 = nn.Sequential(
            Conv2d(base_channels * 2, base_channels * 4, 5, stride=2, padding=2, num_group=num_group),
            Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, num_group=num_group),
            Conv2d(base_channels * 4, base_channels * 4, 3, padding=1, num_group=num_group),
        )

        self.conv3 = nn.Sequential(
            Conv2d(base_channels * 4, base_channels * 8, 5, stride=2, padding=2, num_group=num_group),
            Conv2d(base_channels * 8, base_channels * 8, 3, padding=1, num_group=num_group),
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


class VolumeConvGN(nn.Module):
    def __init__(self, in_channels, base_channels, num_group=4):
        super(VolumeConvGN, self).__init__()
        self.in_channels = in_channels
        self.out_channels = base_channels * 8
        self.base_channels = base_channels

        self.conv1_0 = Conv3d(in_channels, base_channels * 2, 3, stride=2, padding=1, num_group=num_group)
        self.conv2_0 = Conv3d(base_channels * 2, base_channels * 4, 3, stride=2, padding=1, num_group=num_group)
        self.conv3_0 = Conv3d(base_channels * 4, base_channels * 8, 3, stride=2, padding=1, num_group=num_group)

        self.conv0_1 = Conv3d(in_channels, base_channels, 3, stride=1, padding=1, num_group=num_group)

        self.conv1_1 = Conv3d(base_channels * 2, base_channels * 2, 3, stride=1, padding=1, num_group=num_group)
        self.conv2_1 = Conv3d(base_channels * 4, base_channels * 4, 3, stride=1, padding=1, num_group=num_group)

        self.conv3_1 = Conv3d(base_channels * 8, base_channels * 8, 3, stride=1, padding=1, num_group=num_group)

        self.conv4_0 = Deconv3d(base_channels * 8, base_channels * 4, 3, stride=2, num_group=num_group, padding=1,
                                output_padding=1, bias=False, relu=False)
        self.conv5_0 = Deconv3d(base_channels * 4, base_channels * 2, 3, stride=2, num_group=num_group, padding=1,
                                output_padding=1, bias=False, relu=False)
        self.conv6_0 = Deconv3d(base_channels * 2, base_channels, 3, stride=2, num_group=num_group, padding=1,
                                output_padding=1, bias=False, relu=False)

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


class ImageFeatureConvGN(nn.Module):
    def __init__(self, in_channels):
        super(ImageFeatureConvGN, self).__init__()
        self.in_channels = in_channels
        self.conv = nn.Sequential(
            Conv2d(in_channels, in_channels, (2, 1), padding=(1, 0), num_group=in_channels,
                   bias=False, groups=in_channels),
            Conv2d(in_channels, in_channels, (2, 1), padding=0, num_group=in_channels,
                   bias=False, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, (2, 1), padding=0, bias=False, groups=in_channels))

    def forward(self, fetched_img_features):
        """
        :param fetched_img_features: (B, V, C, N)
        :return:
            img_features: (B, V-1, C, N)
        """
        batch_size, num_view, channels, num_points = fetched_img_features.shape
        assert num_view > 1
        img_features = []
        for i in range(1, num_view):
            curr_img_features = torch.stack([fetched_img_features[:, 0, :, :], fetched_img_features[:, i, :, :]],
                                            dim=2)
            curr_img_features = self.conv(curr_img_features)
            img_features.append(curr_img_features)
        img_features = torch.cat(img_features, dim=2).permute([0, 2, 1, 3])

        return img_features
