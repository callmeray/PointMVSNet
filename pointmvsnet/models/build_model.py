import torch
import torch.nn as nn
import torch.nn.functional as F

from pointmvsnet.modules.metric_fn import cal_less_percentage
from pointmvsnet.models.max_avg_pointmvsnet import build_max_avg_pointmvsnet
from pointmvsnet.models.var_pointmvsnet import build_var_pointmvsnet
from pointmvsnet.models.var_pointmvsnet_reg import build_var_pointmvsnet_reg

from pointmvsnet.models.mvsnet import build_mvsnet
from pointmvsnet.models.rmvsnet import build_rmvsnet


def build_model(cfg):
    if cfg.MODEL.VIS_MODEL in ["var", "vis-var", "vis-var-gt"]:
        net = build_var_pointmvsnet(cfg)
    elif cfg.MODEL.VIS_MODEL in ["avg", "max", "vis-avg", "vis-max"]:
        net = build_max_avg_pointmvsnet(cfg)
    elif cfg.MODEL.VIS_MODEL in ["reg"]:
        net = build_var_pointmvsnet_reg(cfg)
    elif cfg.MODEL.VIS_MODEL in ["mvsnet"]:
        net = build_mvsnet(cfg)
    elif cfg.MODEL.VIS_MODEL in ["rmvsnet"]:
        net = build_rmvsnet(cfg)
    else:
        raise ValueError("Unknown visibility-aware model: {}".format(cfg.MODEL.VIS_MODEL))

    loss_fn = PointMVSNetLoss(cfg.MODEL.OCC_LOSS_WEIGHT)
    metric_fn = PointMVSNetMetric(cfg.MODEL.METRIC_DEPTH_INTERVAL, cfg.MODEL.METRIC_MASKED)

    return net, loss_fn, metric_fn


class PointMVSNetLoss(nn.Module):
    def __init__(self, occ_loss_weight=1.0):
        super(PointMVSNetLoss, self).__init__()
        self.occ_loss_weight = occ_loss_weight

    def forward(self, preds):
        losses = {}
        for k, v in preds.items():
            if "_loss" in k:
                if "occ" in k:
                    losses[k] = v.sum() * self.occ_loss_weight
                else:
                    losses[k] = v.sum()

        return losses


class PointMVSNetMetric(nn.Module):
    def __init__(self, metric_depth_interval, masked=False):
        super(PointMVSNetMetric, self).__init__()
        self.metric_depth_interval = metric_depth_interval
        self.masked = masked

    def forward(self, preds, labels):
        gt_depth_img = labels["gt_depth_img"]

        coarse_depth_map = preds["coarse_depth_map"]
        resize_gt_depth = F.interpolate(gt_depth_img, (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
        if self.masked:
            view_mask = F.interpolate(preds["2view_mask"], (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
            resize_gt_depth = resize_gt_depth * view_mask

        less_one_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, self.metric_depth_interval, 1.0)
        less_three_pct_coarse = cal_less_percentage(coarse_depth_map, resize_gt_depth, self.metric_depth_interval, 3.0)

        metrics = {
            "<1_pct_cor": less_one_pct_coarse,
            "<3_pct_cor": less_three_pct_coarse,
        }

        if "nofull_mask" in preds.keys():
            nofull_mask = F.interpolate(preds["nofull_mask"], (coarse_depth_map.shape[2], coarse_depth_map.shape[3]))
            nofull_gt_depth = resize_gt_depth * nofull_mask
            less_one_pct_coarse_nofull = cal_less_percentage(coarse_depth_map, nofull_gt_depth, self.metric_depth_interval, 1.0)
            less_three_pct_coarse_nofull = cal_less_percentage(coarse_depth_map, nofull_gt_depth, self.metric_depth_interval, 3.0)
            metrics["<1_pct_cor_nofull"] = less_one_pct_coarse_nofull
            metrics["<3_pct_cor_nofull"] = less_three_pct_coarse_nofull
            full_gt_depth = resize_gt_depth * (1.0 - nofull_mask)
            less_one_pct_coarse_full = cal_less_percentage(coarse_depth_map, full_gt_depth, self.metric_depth_interval, 1.0)
            less_three_pct_coarse_full = cal_less_percentage(coarse_depth_map, full_gt_depth, self.metric_depth_interval, 3.0)
            metrics["<1_pct_cor_full"] = less_one_pct_coarse_full
            metrics["<3_pct_cor_full"] = less_three_pct_coarse_full

        if "occ_pred" in preds.keys() and "occ_gt" in preds.keys():
            occ_gt = preds["occ_gt"]
            noocc_gt = preds["noocc_gt"]
            occ_pred = preds["occ_pred"]
            occ_gt = F.interpolate(occ_gt, (occ_pred.shape[2], occ_pred.shape[3], occ_pred.shape[4]),
                                   mode="trilinear")
            noocc_gt = F.interpolate(noocc_gt, (occ_pred.shape[2], occ_pred.shape[3], occ_pred.shape[4]),
                                     mode="trilinear")
            occ_pred_05 = (occ_pred > 0.5).float()
            noocc_pred_05 = (occ_pred < 0.5).float()
            occ_rec = (occ_pred_05 * occ_gt).sum() / torch.clamp(occ_gt.sum(), 1e-6)
            noocc_rec = (noocc_pred_05 * noocc_gt).sum() / torch.clamp(noocc_gt.sum(), 1e-6)
            metrics["occ_0.5_rec"] = occ_rec
            metrics["noocc_0.5_rec"] = noocc_rec

        if "flow1" in preds.keys():
            flow1 = preds["flow1"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow1.shape[2], flow1.shape[3]))
            if self.masked:
                view_mask = F.interpolate(preds["2view_mask"], (flow1.shape[2], flow1.shape[3]))
                resize_gt_depth = resize_gt_depth * view_mask

            less_one_pct_flow1 = cal_less_percentage(flow1, resize_gt_depth, self.metric_depth_interval, 1.0)
            less_three_pct_flow1 = cal_less_percentage(flow1, resize_gt_depth, self.metric_depth_interval, 3.0)

            metrics["<1_pct_flow1"] = less_one_pct_flow1
            metrics["<3_pct_flow1"] = less_three_pct_flow1

            if "nofull_mask" in preds.keys():
                nofull_mask = F.interpolate(preds["nofull_mask"], (flow1.shape[2], flow1.shape[3]))
                nofull_gt_depth = resize_gt_depth * nofull_mask
                less_one_pct_flow1_nofull = cal_less_percentage(flow1, nofull_gt_depth, self.metric_depth_interval,
                                                                1.0)
                less_three_pct_flow1_nofull = cal_less_percentage(flow1, nofull_gt_depth, self.metric_depth_interval,
                                                                  3.0)
                metrics["<1_pct_flow1_nofull"] = less_one_pct_flow1_nofull
                metrics["<3_pct_flow1_nofull"] = less_three_pct_flow1_nofull
                full_gt_depth = resize_gt_depth * (1.0 - nofull_mask)
                less_one_pct_flow1_full = cal_less_percentage(flow1, full_gt_depth, self.metric_depth_interval, 1.0)
                less_three_pct_flow1_full = cal_less_percentage(flow1, full_gt_depth, self.metric_depth_interval, 3.0)
                metrics["<1_pct_flow1_full"] = less_one_pct_flow1_full
                metrics["<3_pct_flow1_full"] = less_three_pct_flow1_full

        if "flow2" in preds.keys():
            flow2 = preds["flow2"]
            resize_gt_depth = F.interpolate(gt_depth_img, (flow2.shape[2], flow2.shape[3]))
            if self.masked:
                view_mask = F.interpolate(preds["2view_mask"], (flow2.shape[2], flow2.shape[3]))
                resize_gt_depth = resize_gt_depth * view_mask

            less_one_pct_flow2 = cal_less_percentage(flow2, resize_gt_depth, self.metric_depth_interval, 1.0)
            less_three_pct_flow2 = cal_less_percentage(flow2, resize_gt_depth, self.metric_depth_interval, 3.0)

            metrics["<1_pct_flow2"] = less_one_pct_flow2
            metrics["<3_pct_flow2"] = less_three_pct_flow2

            if "nofull_mask" in preds.keys():
                nofull_mask = F.interpolate(preds["nofull_mask"], (flow2.shape[2], flow2.shape[3]))
                nofull_gt_depth = resize_gt_depth * nofull_mask
                less_one_pct_flow2_nofull = cal_less_percentage(flow2, nofull_gt_depth, self.metric_depth_interval, 1.0)
                less_three_pct_flow2_nofull = cal_less_percentage(flow2, nofull_gt_depth, self.metric_depth_interval,
                                                                  3.0)
                metrics["<1_pct_flow2_nofull"] = less_one_pct_flow2_nofull
                metrics["<3_pct_flow2_nofull"] = less_three_pct_flow2_nofull
                full_gt_depth = resize_gt_depth * (1. - nofull_mask)
                less_one_pct_flow2_full = cal_less_percentage(flow2, full_gt_depth, self.metric_depth_interval, 1.0)
                less_three_pct_flow2_full = cal_less_percentage(flow2, full_gt_depth, self.metric_depth_interval, 3.0)
                metrics["<1_pct_flow2_full"] = less_one_pct_flow2_full
                metrics["<3_pct_flow2_full"] = less_three_pct_flow2_full

        return metrics
