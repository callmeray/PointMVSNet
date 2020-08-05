import torch
import torch.nn.functional as F


# -----------------------------------------------------------------------------
# Distance
# -----------------------------------------------------------------------------

def pdist(feature):
    """Compute pairwise distances of features.

    Args:
        feature (torch.Tensor): (batch_size, channels, num_features)

    Returns:
        distance (torch.Tensor): (batch_size, num_features, num_features)

    Notes:
        This method returns square distances, and is optimized for lower memory and faster speed.
        Sqaure sum is more efficient than gather diagonal from inner product.

    """
    square_sum = torch.sum(feature ** 2, 1, keepdim=True)
    square_sum = square_sum + square_sum.transpose(1, 2)
    distance = torch.baddbmm(square_sum, feature.transpose(1, 2), feature, alpha=-2.0)
    return distance


def pdist2(feature1, feature2):
    """Compute pairwise distances of two sets of features.

    Args:
        feature1 (torch.Tensor): (batch_size, channels, num_features1)
        feature2 (torch.Tensor): (batch_size, channels, num_features2)

    Returns:
        distance (torch.Tensor): (batch_size, num_features1, num_features2)

    Notes:
        This method returns square distances, and is optimized for lower memory and faster speed.
        Sqaure sum is more efficient than gather diagonal from inner product.

    """
    square_sum1 = torch.sum(feature1 ** 2, 1, keepdim=True)
    square_sum2 = torch.sum(feature2 ** 2, 1, keepdim=True)
    square_sum = square_sum1.transpose(1, 2) + square_sum2
    distance = torch.baddbmm(square_sum, feature1.transpose(1, 2), feature2, alpha=-2.0)
    return distance


# -----------------------------------------------------------------------------
# Losses
# -----------------------------------------------------------------------------

def encode_one_hot(target, num_classes):
    """Encode integer labels into one-hot vectors

    Args:
        target (torch.Tensor): (N,)
        num_classes (int): the number of classes

    Returns:
        torch.FloatTensor: (N, C)

    """
    one_hot = target.new_zeros(target.size(0), num_classes)
    one_hot = one_hot.scatter(1, target.unsqueeze(1), 1)
    return one_hot.float()


def smooth_cross_entropy(input, target, label_smoothing):
    """Cross entropy loss with label smoothing

    Args:
        input (torch.Tensor): (N, C)
        target (torch.Tensor): (N,)
        label_smoothing (float):

    Returns:
        loss (torch.Tensor): scalar

    """
    assert input.dim() == 2 and target.dim() == 1
    assert isinstance(label_smoothing, float)
    batch_size, num_classes = input.shape
    one_hot = torch.zeros_like(input).scatter(1, target.unsqueeze(1), 1)
    smooth_one_hot = one_hot * (1 - label_smoothing) + torch.ones_like(input) * (label_smoothing / num_classes)
    log_prob = F.log_softmax(input, dim=1)
    loss = (- smooth_one_hot * log_prob).sum(1).mean()
    return loss


