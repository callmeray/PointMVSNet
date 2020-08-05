#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import git
import sys
from path import Path

sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from pointmvsnet.config import load_cfg_from_file
from pointmvsnet.utils.logger import setup_logger
from pointmvsnet.models import build_model
from pointmvsnet.utils.checkpoint import Checkpointer
from pointmvsnet.data_loader import build_data_loader
from pointmvsnet.utils.metric_logger import MetricLogger
from pointmvsnet.utils.eval_file_logger import eval_file_logger


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Point-MVSNet Evaluation")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "--cpu",
        action='store_true',
        default=False,
        help="whether to only use cpu for test",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def test_model(model,
               image_scales,
               inter_scales,
               data_loader,
               folder,
               isCPU=False,
               ):
    logger = logging.getLogger("pointmvsnet.test")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()
    path_list = []
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end
            curr_ref_img_path = data_batch["ref_img_path"][0]
            l = curr_ref_img_path.split("/")
            if "dtu" in curr_ref_img_path:
                eval_folder = "/".join(l[:-3])
                scene = l[-2]
                scene_folder = osp.join(eval_folder, folder, scene)
                out_index = int(l[-1][5:8]) - 1
            elif "tanks" in curr_ref_img_path:
                eval_folder = "/".join(l[:-3])
                scene = l[-3]
                scene_folder = osp.join(eval_folder, folder, scene)
                out_index = int(l[-1][0:8])
            # out_ref_image_path = scene_folder + ('/%08d.jpg' % out_index)
            # if osp.exists(out_ref_image_path):
            #     print("{} exists".format(out_ref_image_path))
            #     continue
            out_flow_path = scene_folder + ("/%08d_flow1.pfm" % out_index)
            out_flow_prob_path = scene_folder + ("/%08d_flow1_prob.pfm" % out_index)
            if osp.exists(out_flow_path) and osp.exists(out_flow_prob_path):
                print("{} exits".format(out_flow_path))
                continue

            path_list.extend(curr_ref_img_path)
            if not isCPU:
                data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if
                              isinstance(v, torch.Tensor)}
            preds = model(data_batch, image_scales, inter_scales, use_occ_pred=True, isFlow=True, isTest=True)

            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)
            logger.info(
                "{} finished.".format(curr_ref_img_path) + str(meters))
            eval_file_logger(data_batch, preds, curr_ref_img_path, folder)
            del data_batch
            del preds
            torch.cuda.empty_cache()


def test(cfg, output_dir, isCPU=False):
    logger = logging.getLogger("pointmvsnet.tester")
    # build model
    model, _, _ = build_model(cfg)
    if not isCPU:
        model = nn.DataParallel(model).cuda()

    # build checkpointer
    checkpointer = Checkpointer(model, save_dir=output_dir)

    if cfg.TEST.WEIGHT:
        weight_path = cfg.TEST.WEIGHT.replace("@", output_dir)
        checkpointer.load(weight_path, resume=False)
    else:
        checkpointer.load(None, resume=True)

    # build data loader
    test_data_loader = build_data_loader(cfg, mode="test")
    start_time = time.time()
    test_model(model,
               image_scales=cfg.MODEL.TEST.IMG_SCALES,
               inter_scales=cfg.MODEL.TEST.INTER_SCALES,
               data_loader=test_data_loader,
               folder=output_dir.split("/")[-1],
               isCPU=isCPU,
               )
    test_time = time.time() - start_time
    logger.info("Test forward time: {:.2f}s".format(test_time))


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    if len(args.opts) == 1:
        args.opts = args.opts[0].strip().split(" ")

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    assert cfg.TEST.BATCH_SIZE == 1

    isCPU = args.cpu

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        Path(output_dir).makedirs_p()

    logger = setup_logger("pointmvsnet", output_dir, prefix="test")
    try:
        repo = git.Repo(path=output_dir, search_parent_directories=True)
        sha = repo.head.object.hexsha
        logger.info("Git SHA: {}".format(sha))
    except:
        logger.info("No git info")

    if isCPU:
        logger.info("Using CPU")
    else:
        logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    test(cfg, output_dir, isCPU=isCPU)


if __name__ == "__main__":
    main()
