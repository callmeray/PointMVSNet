from torch.utils.data import DataLoader

from pointmvsnet.data_loader.dtu import DTUTrainValSet, DTUTestSet
from pointmvsnet.data_loader.tanks_and_temples import TanksAndTemplesTestSet
from pointmvsnet.data_loader.blended_mvs import BlendedMVSTrainValSet


def build_data_loader(cfg, mode="train"):
    if cfg.DATA.DATASET == "DTU":
        if mode == "train":
            dataset = DTUTrainValSet(
                root_dir=cfg.DATA.TRAIN.ROOT_DIR,
                mode="train",
                num_view=cfg.DATA.TRAIN.NUM_VIEW,
                num_depth=cfg.DATA.TRAIN.NUM_DEPTH,
                interval_scale=cfg.DATA.TRAIN.INTERVAL_SCALE,
            )
        elif mode == "val":
            dataset = DTUTrainValSet(
                root_dir=cfg.DATA.VAL.ROOT_DIR,
                mode="val",
                num_view=cfg.DATA.VAL.NUM_VIEW,
                num_depth=cfg.DATA.TRAIN.NUM_DEPTH,
                interval_scale=cfg.DATA.TRAIN.INTERVAL_SCALE,
            )
        elif mode == "test":
            dataset = DTUTestSet(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                num_view=cfg.DATA.TEST.NUM_VIEW,
                num_depth=cfg.DATA.TEST.NUM_DEPTH,
                interval_scale=cfg.DATA.TEST.INTERVAL_SCALE,
                img_height=cfg.DATA.TEST.IMG_HEIGHT,
                img_width=cfg.DATA.TEST.IMG_WIDTH,
            )
        else:
            raise ValueError("Unknown mode: {}.".format(mode))
    elif cfg.DATA.DATASET == "Blended_MVS":
        if mode == "train":
            dataset = BlendedMVSTrainValSet(
                root_dir=cfg.DATA.TRAIN.ROOT_DIR,
                mode="train",
                num_view=cfg.DATA.TRAIN.NUM_VIEW,
                num_depth=cfg.DATA.TRAIN.NUM_DEPTH,
                interval_scale=cfg.DATA.TRAIN.INTERVAL_SCALE,
            )
        elif mode == "val":
            dataset = BlendedMVSTrainValSet(
                root_dir=cfg.DATA.VAL.ROOT_DIR,
                mode="val",
                num_view=cfg.DATA.VAL.NUM_VIEW,
                num_depth=cfg.DATA.TRAIN.NUM_DEPTH,
                interval_scale=cfg.DATA.TRAIN.INTERVAL_SCALE,
            )
        else:
            raise ValueError("Unknown mode: {}.".format(mode))
    elif cfg.DATA.DATASET == "TANK_TEMPLE":
        if mode == "test":
            dataset = TanksAndTemplesTestSet(
                root_dir=cfg.DATA.TEST.ROOT_DIR,
                num_view=cfg.DATA.TEST.NUM_VIEW,
                num_depth=cfg.DATA.TEST.NUM_DEPTH,
                interval_scale=cfg.DATA.TEST.INTERVAL_SCALE,
                img_height=cfg.DATA.TEST.IMG_HEIGHT,
                img_width=cfg.DATA.TEST.IMG_WIDTH,
                depth_in_folder=cfg.DATA.TEST.DEPTH_FOLDER,
                depth_in_name=cfg.DATA.TEST.DEPTH_IN_NAME,
                prob_in_name=cfg.DATA.TEST.PROB_IN_NAME,
            )
    if mode == "train":
        batch_size = cfg.TRAIN.BATCH_SIZE
    else:
        batch_size = cfg.TEST.BATCH_SIZE

    data_loader = DataLoader(
        dataset,
        batch_size,
        shuffle=(mode == "train"),
        num_workers=cfg.DATA.NUM_WORKERS,
        pin_memory=True,
    )

    return data_loader
