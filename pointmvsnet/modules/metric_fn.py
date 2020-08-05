import torch


def cal_less_percentage(pred_depth, gt_depth, depth_interval, threshold):
    mask_valid = (~torch.eq(gt_depth, 0.0)).type(torch.float)
    denom = torch.sum(mask_valid) + 1e-7
    abs_diff_image = torch.abs(pred_depth - gt_depth) / depth_interval

    pct = mask_valid * (abs_diff_image <= threshold).type(torch.float)

    pct = torch.sum(pct) / denom

    return pct


