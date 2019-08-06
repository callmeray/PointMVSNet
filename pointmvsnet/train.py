#!/usr/bin/env python
import argparse
import os.path as osp
import logging
import time
import sys
sys.path.insert(0, osp.dirname(__file__) + '/..')

import torch
import torch.nn as nn

from pointmvsnet.config import load_cfg_from_file
from pointmvsnet.utils.io import mkdir
from pointmvsnet.utils.logger import setup_logger
from pointmvsnet.utils.torch_utils import set_random_seed
from pointmvsnet.model import build_pointmvsnet as build_model
from pointmvsnet.solver import build_optimizer, build_scheduler
from pointmvsnet.utils.checkpoint import Checkpointer
from pointmvsnet.dataset import build_data_loader
from pointmvsnet.utils.tensorboard_logger import TensorboardLogger
from pointmvsnet.utils.metric_logger import MetricLogger
from pointmvsnet.utils.file_logger import file_logger


def parse_args():
    parser = argparse.ArgumentParser(description="PyTorch Point-MVSNet Training")
    parser.add_argument(
        "--cfg",
        dest="config_file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()
    return args


def train_model(model,
                loss_fn,
                metric_fn,
                image_scales,
                inter_scales,
                isFlow,
                data_loader,
                optimizer,
                curr_epoch,
                tensorboard_logger,
                log_period=1,
                output_dir="",
                ):
    logger = logging.getLogger("pointmvsnet.train")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()
    path_list = []

    for iteration, data_batch in enumerate(data_loader):
        data_time = time.time() - end
        curr_ref_img_path = data_batch["ref_img_path"]
        path_list.extend(curr_ref_img_path)
        data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}

        preds = model(data_batch, image_scales, inter_scales, isFlow)
        optimizer.zero_grad()

        loss_dict = loss_fn(preds, data_batch, isFlow)
        metric_dict = metric_fn(preds, data_batch, isFlow)
        losses = sum(loss_dict.values())
        meters.update(loss=losses, **loss_dict, **metric_dict)

        losses.backward()

        optimizer.step()

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        if iteration % log_period == 0:
            logger.info(
                meters.delimiter.join(
                    [
                        "EPOCH: {epoch:2d}",
                        "iter: {iter:4d}",
                        "{meters}",
                        "lr: {lr:.2e}",
                        "max mem: {memory:.0f}",
                    ]
                ).format(
                    epoch=curr_epoch,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    memory=torch.cuda.max_memory_allocated() / (1024.0 ** 2),
                )
            )
            tensorboard_logger.add_scalars(loss_dict, curr_epoch * total_iteration + iteration, prefix="train")
            tensorboard_logger.add_scalars(metric_dict, curr_epoch * total_iteration + iteration, prefix="train")

        if iteration % (100 * log_period) == 0:
            file_logger(data_batch, preds, curr_epoch * total_iteration + iteration, output_dir, prefix="train")

    return meters


def validate_model(model,
                   loss_fn,
                   metric_fn,
                   image_scales,
                   inter_scales,
                   isFlow,
                   data_loader,
                   curr_epoch,
                   tensorboard_logger,
                   log_period=1,
                   output_dir="",
                   ):
    logger = logging.getLogger("pointmvsnet.validate")
    meters = MetricLogger(delimiter="  ")
    model.train()
    end = time.time()
    total_iteration = data_loader.__len__()
    with torch.no_grad():
        for iteration, data_batch in enumerate(data_loader):
            data_time = time.time() - end
            curr_ref_img_path = data_batch["ref_img_path"]

            data_batch = {k: v.cuda(non_blocking=True) for k, v in data_batch.items() if isinstance(v, torch.Tensor)}

            preds = model(data_batch, image_scales, inter_scales, isFlow)
            loss_dict = loss_fn(preds, data_batch, isFlow)
            metric_dict = metric_fn(preds, data_batch, isFlow)
            losses = sum(loss_dict.values())
            meters.update(loss=losses, **loss_dict, **metric_dict)
            batch_time = time.time() - end
            end = time.time()
            meters.update(time=batch_time, data=data_time)

            if iteration % log_period == 0:
                logger.info(
                    meters.delimiter.join(
                        [
                            "EPOCH: {epoch:2d}",
                            "iter: {iter:4d}",
                            "{meters}",
                        ]
                    ).format(
                        epoch=curr_epoch,
                        iter=iteration,
                        meters=str(meters),
                    )
                )
                tensorboard_logger.add_scalars(meters.meters, curr_epoch * total_iteration + iteration, prefix="valid")

            if iteration % (100 * log_period) == 0:
                file_logger(data_batch, preds, curr_epoch * total_iteration + iteration, output_dir, prefix="valid")

    return meters


def train(cfg, output_dir=""):
    logger = logging.getLogger("pointmvsnet.trainer")

    # build model
    set_random_seed(cfg.RNG_SEED)
    model, loss_fn, metric_fn = build_model(cfg)
    logger.info("Build model:\n{}".format(str(model)))
    model = nn.DataParallel(model).cuda()

    # build optimizer
    optimizer = build_optimizer(cfg, model)

    # build lr scheduler
    scheduler = build_scheduler(cfg, optimizer)

    # build checkpointer
    checkpointer = Checkpointer(model,
                                optimizer=optimizer,
                                scheduler=scheduler,
                                save_dir=output_dir,
                                logger=logger)

    checkpoint_data = checkpointer.load(cfg.MODEL.WEIGHT, resume=cfg.AUTO_RESUME)
    ckpt_period = cfg.TRAIN.CHECKPOINT_PERIOD

    # build data loader
    train_data_loader = build_data_loader(cfg, mode="train")
    val_period = cfg.TRAIN.VAL_PERIOD
    val_data_loader = build_data_loader(cfg, mode="val") if val_period > 0 else None

    # build tensorboard logger (optionally by comment)
    tensorboard_logger = TensorboardLogger(output_dir)

    # train
    max_epoch = cfg.SCHEDULER.MAX_EPOCH
    start_epoch = checkpoint_data.get("epoch", 0)
    best_metric_name = "best_{}".format(cfg.TRAIN.VAL_METRIC)
    best_metric = checkpoint_data.get(best_metric_name, None)
    logger.info("Start training from epoch {}".format(start_epoch))
    for epoch in range(start_epoch, max_epoch):
        cur_epoch = epoch + 1
        scheduler.step()
        start_time = time.time()
        train_meters = train_model(model,
                                   loss_fn,
                                   metric_fn,
                                   image_scales=cfg.MODEL.TRAIN.IMG_SCALES,
                                   inter_scales=cfg.MODEL.TRAIN.INTER_SCALES,
                                   isFlow=(cur_epoch > cfg.SCHEDULER.INIT_EPOCH),
                                   data_loader=train_data_loader,
                                   optimizer=optimizer,
                                   curr_epoch=epoch,
                                   tensorboard_logger=tensorboard_logger,
                                   log_period=cfg.TRAIN.LOG_PERIOD,
                                   output_dir=output_dir,
                                   )
        epoch_time = time.time() - start_time
        logger.info("Epoch[{}]-Train {}  total_time: {:.2f}s".format(
            cur_epoch, train_meters.summary_str, epoch_time))

        # checkpoint
        if cur_epoch % ckpt_period == 0 or cur_epoch == max_epoch:
            checkpoint_data["epoch"] = cur_epoch
            checkpoint_data[best_metric_name] = best_metric
            checkpointer.save("model_{:03d}".format(cur_epoch), **checkpoint_data)

        # validate
        if val_period < 1:
            continue
        if cur_epoch % val_period == 0 or cur_epoch == max_epoch:
            val_meters = validate_model(model,
                                        loss_fn,
                                        metric_fn,
                                        image_scales=cfg.MODEL.VAL.IMG_SCALES,
                                        inter_scales=cfg.MODEL.VAL.INTER_SCALES,
                                        isFlow=(cur_epoch > cfg.SCHEDULER.INIT_EPOCH),
                                        data_loader=val_data_loader,
                                        curr_epoch=epoch,
                                        tensorboard_logger=tensorboard_logger,
                                        log_period=cfg.TEST.LOG_PERIOD,
                                        output_dir=output_dir,
                                        )
            logger.info("Epoch[{}]-Val {}".format(cur_epoch, val_meters.summary_str))

            # best validation
            cur_metric = val_meters.meters[cfg.TRAIN.VAL_METRIC].global_avg
            if best_metric is None or cur_metric > best_metric:
                best_metric = cur_metric
                checkpoint_data["epoch"] = cur_epoch
                checkpoint_data[best_metric_name] = best_metric
                checkpointer.save("model_best", **checkpoint_data)

    logger.info("Best val-{} = {}".format(cfg.TRAIN.VAL_METRIC, best_metric))

    return model


def main():
    args = parse_args()
    num_gpus = torch.cuda.device_count()

    cfg = load_cfg_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        config_path = osp.splitext(args.config_file)[0]
        config_path = config_path.replace("configs", "outputs")
        output_dir = output_dir.replace('@', config_path)
        mkdir(output_dir)

    logger = setup_logger("pointmvsnet", output_dir, prefix="train")
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    logger.info("Running with config:\n{}".format(cfg))

    train(cfg, output_dir)


if __name__ == "__main__":
    main()
