"""Helpers for DGCNN"""

import torch

from pointmvsnet.nn.functional import pdist
from .gather_knn import gather_knn


def get_knn_inds(pdist, k=20, remove=False):
    """Get k nearest neighbour index based on the pairwise_distance.

    Args:
        pdist (torch.Tensor): tensor (batch_size, num_nodes, num_nodes)
        k (int): the number of nearest neighbour
        remove (bool): whether to remove itself

    Returns:
        knn_inds (torch.Tensor): (batch_size, num_nodes, k)

    """
    if remove:
        _, knn_inds = torch.topk(pdist, k + 1, largest=False, sorted=False)
        return knn_inds[..., 1:]
    else:
        _, knn_inds = torch.topk(pdist, k, largest=False, sorted=False)
        return knn_inds


def construct_edge_feature_index(feature, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using advanced indexing

    Args:
        feature (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    batch_idx = torch.arange(batch_size).view(-1, 1, 1, 1)
    feature_idx = torch.arange(channels).view(1, -1, 1, 1)
    # (batch_size, channels, num_nodes, k)
    feature_neighbour = feature[batch_idx, feature_idx, knn_inds.unsqueeze(1)]
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def construct_edge_feature_gather(feature, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using torch.gather

    Args:
        feature (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    Notes:
        Pytorch Gather is 50x faster than advanced indexing, but needs 2x more memory.
        It is because it will allocate a tensor as large as expanded features during backward.

    """
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_expand = feature.unsqueeze(2).expand(batch_size, channels, num_nodes, num_nodes)
    knn_inds_expand = knn_inds.unsqueeze(1).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = torch.gather(feature_expand, 3, knn_inds_expand)
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def construct_edge_feature(feature, knn_inds):
    """Construct edge feature for each point (or regarded as a node)
    using gather_knn

    Args:
        feature (torch.Tensor): point features, (batch_size, channels, num_nodes),
        knn_inds (torch.Tensor): indices of k-nearest neighbour, (batch_size, num_nodes, k)

    Returns:
        edge_feature: (batch_size, 2*channels, num_nodes, k)

    """
    batch_size, channels, num_nodes = feature.shape
    k = knn_inds.size(-1)

    # CAUTION: torch.expand
    feature_central = feature.unsqueeze(3).expand(batch_size, channels, num_nodes, k)
    feature_neighbour = gather_knn(feature, knn_inds)
    # (batch_size, 2 * channels, num_nodes, k)
    edge_feature = torch.cat((feature_central, feature_neighbour - feature_central), 1)

    return edge_feature


def get_edge_feature(feature, k):
    """Get edge feature for point features

    Args:
        feature (torch.Tensor): (batch_size, channels, num_nodes)
        k (int): the number of nearest neighbours

    Returns:
        edge_feature (torch.Tensor): (batch_size, 2*num_dims, num_nodes, k)

    """
    with torch.no_grad():
        distance = pdist(feature)
        knn_inds = get_knn_inds(distance, k)

    edge_feature = construct_edge_feature(feature, knn_inds)

    return edge_feature


def get_pixel_grids(height, width):
    with torch.no_grad():
        # texture coordinate
        x_linspace = torch.linspace(0.5, width - 0.5, width).view(1, width).expand(height, width)
        y_linspace = torch.linspace(0.5, height - 0.5, height).view(height, 1).expand(height, width)
        # y_coordinates, x_coordinates = torch.meshgrid(y_linspace, x_linspace)
        x_coordinates = x_linspace.contiguous().view(-1)
        y_coordinates = y_linspace.contiguous().view(-1)
        ones = torch.ones(height * width)
        indices_grid = torch.stack([x_coordinates, y_coordinates, ones], dim=0)
    return indices_grid


def get_propability_map(cv, depth_map, depth_start, depth_interval):
    """get probability map from cost volume"""
    with torch.no_grad():
        batch_size, channels, height, width = list(depth_map.size())
        depth = cv.size(1)

        # byx coordinates, batched & flattened
        b_coordinates = torch.arange(batch_size, dtype=torch.int64)
        y_coordinates = torch.arange(height, dtype=torch.int64)
        x_coordinates = torch.arange(width, dtype=torch.int64)
        b_coordinates = b_coordinates.view(batch_size, 1, 1).expand(batch_size, height, width)
        y_coordinates = y_coordinates.view(1, height, 1).expand(batch_size, height, width)
        x_coordinates = x_coordinates.view(1, 1, width).expand(batch_size, height, width)

        b_coordinates = b_coordinates.contiguous().view(-1).type(torch.long)
        y_coordinates = y_coordinates.contiguous().view(-1).type(torch.long)
        x_coordinates = x_coordinates.contiguous().view(-1).type(torch.long)
        # b_coordinates = _repeat_(b_coordinates, batch_size)
        # y_coordinates = _repeat_(y_coordinates, batch_size)
        # x_coordinates = _repeat_(x_coordinates, batch_size)

        # d coordinates (floored and ceiled), batched & flattened
        d_coordinates = ((depth_map - depth_start.view(-1, 1, 1, 1)) / depth_interval.view(-1, 1, 1, 1)).view(-1)
        d_coordinates = torch.detach(d_coordinates)
        d_coordinates_left0 = torch.clamp(d_coordinates.floor(), 0, depth - 1).type(torch.long)
        d_coordinates_right0 = torch.clamp(d_coordinates.ceil(), 0, depth - 1).type(torch.long)

        # # get probability image by gathering
        prob_map_left0 = cv[b_coordinates, d_coordinates_left0, y_coordinates, x_coordinates]
        prob_map_right0 = cv[b_coordinates, d_coordinates_right0, y_coordinates, x_coordinates]

        prob_map = prob_map_left0 + prob_map_right0
        prob_map = prob_map.view(batch_size, 1, height, width)

    return prob_map
