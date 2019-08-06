import random
import numpy as np

import torch
import torch.nn.functional as F

def set_random_seed(seed):
    if seed < 0:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_knn_3d(xyz, kernel_size=5, knn=20):
    """ Use 3D Conv to compute neighbour distance and find k nearest neighbour
          xyz: (B, 3, D, H, W)

      Returns:
        idx: (B, D*H*W, k)
    """
    batch_size, _, depth, height, width = list(xyz.size())
    assert (kernel_size % 2 == 1)
    hk = (kernel_size // 2)
    k2 = kernel_size ** 2
    k3 = kernel_size ** 3

    t = np.zeros((kernel_size, kernel_size, kernel_size, 1, kernel_size ** 3))
    ind = 0
    for i in range(kernel_size):
        for j in range(kernel_size):
            for k in range(kernel_size):
                t[i, j, k, 0, ind] -= 1.0
                t[hk, hk, hk, 0, ind] += 1.0
                ind += 1
    weight = np.zeros((kernel_size, kernel_size, kernel_size, 3, 3 * k3))
    weight[:, :, :, 0:1, :k3] = t
    weight[:, :, :, 1:2, k3:2 * k3] = t
    weight[:, :, :, 2:3, 2 * k3:3 * k3] = t
    weight = torch.tensor(weight).float()

    weights_torch = torch.Tensor(weight.permute((4, 3, 0, 1, 2))).to(xyz.device)
    dist = F.conv3d(xyz, weights_torch, padding=hk)

    dist_flat = dist.contiguous().view(batch_size, 3, k3, -1)
    dist2 = torch.sum(dist_flat ** 2, dim=1)

    _, nn_idx = torch.topk(-dist2, k=knn, dim=1)
    nn_idx = nn_idx.permute(0, 2, 1)
    d_offset = nn_idx // k2 - hk
    h_offset = (nn_idx % k2) // kernel_size - hk
    w_offset = nn_idx % kernel_size - hk

    idx = torch.arange(depth * height * width).to(xyz.device)
    idx = idx.view(1, -1, 1).expand(batch_size, -1, knn)
    idx = idx + (d_offset * height * width) + (h_offset * width) + w_offset

    idx = torch.clamp(idx, 0, depth * height * width - 1)

    return idx
