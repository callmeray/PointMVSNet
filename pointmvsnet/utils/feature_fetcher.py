import torch
import torch.nn as nn
import torch.nn.functional as F
import random
import numpy as np


class FeatureFetcher(nn.Module):
    def __init__(self, mode="bilinear"):
        super(FeatureFetcher, self).__init__()
        self.mode = mode

    def forward(self, feature_maps, pts, cam_intrinsics, cam_extrinsics):
        """

        :param feature_maps: torch.tensor, [B, V, C, H, W]
        :param pts: torch.tensor, [B, 3, N]
        :param cam_intrinsics: torch.tensor, [B, V, 3, 3]
        :param cam_extrinsics: torch.tensor, [B, V, 3, 4], [R|t], p_cam = R*p_world + t
        :return:
            pts_feature: torch.tensor, [B, V, C, N]
        """
        batch_size, num_view, channels, height, width = list(feature_maps.size())
        feature_maps = feature_maps.view(batch_size * num_view, channels, height, width)

        curr_batch_size = batch_size * num_view
        cam_intrinsics = cam_intrinsics.view(curr_batch_size, 3, 3)

        with torch.no_grad():
            num_pts = pts.size(2)
            pts_expand = pts.unsqueeze(1).contiguous().expand(batch_size, num_view, 3, num_pts) \
                .contiguous().view(curr_batch_size, 3, num_pts)
            if cam_extrinsics is None:
                transformed_pts = pts_expand.type(torch.float).transpose(1, 2)
            else:
                cam_extrinsics = cam_extrinsics.view(curr_batch_size, 3, 4)
                R = torch.narrow(cam_extrinsics, 2, 0, 3)
                t = torch.narrow(cam_extrinsics, 2, 3, 1).expand(curr_batch_size, 3, num_pts)
                transformed_pts = torch.bmm(R, pts_expand) + t
                transformed_pts = transformed_pts.type(torch.float).transpose(1, 2)
            x = transformed_pts[..., 0]
            y = transformed_pts[..., 1]
            z = transformed_pts[..., 2]

            normal_uv = torch.cat(
                [torch.div(x, z).unsqueeze(-1), torch.div(y, z).unsqueeze(-1), torch.ones_like(x).unsqueeze(-1)],
                dim=-1)
            uv = torch.bmm(normal_uv, cam_intrinsics.transpose(1, 2))
            uv = uv[:, :, :2]

            grid = (uv - 0.5).view(curr_batch_size, num_pts, 1, 2)
            grid[..., 0] = (grid[..., 0] / float(width - 1)) * 2 - 1.0
            grid[..., 1] = (grid[..., 1] / float(height - 1)) * 2 - 1.0

        pts_feature = F.grid_sample(feature_maps, grid, mode=self.mode)
        pts_feature = pts_feature.squeeze(3)

        pts_feature = pts_feature.view(batch_size, num_view, channels, num_pts)

        return pts_feature


def test_feature_fetching():
    import numpy as np
    batch_size = 3
    num_view = 2
    channels = 16
    height = 240
    width = 320
    num_pts = 32

    cam_intrinsic = torch.tensor([[10, 0, 1], [0, 10, 1], [0, 0, 1]]).float() \
        .view(1, 1, 3, 3).expand(batch_size, num_view, 3, 3).cuda()
    cam_extrinsic = torch.rand(batch_size, num_view, 3, 4).cuda()

    feature_fetcher = FeatureFetcher().cuda()

    features = torch.rand(batch_size, num_view, channels, height, width).cuda()

    imgpt = torch.tensor([60.5, 80.5, 1.0]).view(1, 1, 3, 1).expand(batch_size, num_view, 3, num_pts).cuda()

    z = 200

    pt = torch.matmul(torch.inverse(cam_intrinsic), imgpt) * z

    pt = torch.matmul(torch.inverse(cam_extrinsic[:, :, :, :3]),
                      (pt - cam_extrinsic[:, :, :, 3].unsqueeze(-1)))  # Xc = [R|T] Xw

    gathered_feature = feature_fetcher(features, pt[:, 0, :, :], cam_intrinsic, cam_extrinsic)

    gathered_feature = gathered_feature[:, 0, :, 0]
    np.savetxt("gathered_feature.txt", gathered_feature.detach().cpu().numpy(), fmt="%.4f")

    groundtruth_feature = features[:, :, :, 80, 60][:, 0, :]
    np.savetxt("groundtruth_feature.txt", groundtruth_feature.detach().cpu().numpy(), fmt="%.4f")

    print(np.allclose(gathered_feature.detach().cpu().numpy(), groundtruth_feature.detach().cpu().numpy(), 1.e-2))


if __name__ == "__main__":
    test_feature_fetching()
