"""
Build optimizers and schedulers

Notes:
    Default optimizer will optimize all parameters.
    Custom optimizer should be implemented and registered in '_OPTIMIZER_BUILDERS'

"""
import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm
from torch.nn.modules.conv import _ConvNd

_OPTIMIZER_BUILDERS = {}


def build_optimizer(cfg, model):
    name = cfg.SOLVER.TYPE
    if hasattr(torch.optim, name):
        def builder(cfg, model):
            return getattr(torch.optim, name)(
                group_weight(model, cfg.SOLVER.WEIGHT_DECAY),
                lr=cfg.SOLVER.BASE_LR,
                **cfg.SOLVER[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, model)


def group_weight(module, weight_decay):
    group_decay = []
    group_no_decay = []
    keywords = [".bn."]

    for m in list(module.named_parameters()):
        exclude = False
        for k in keywords:
            if k in m[0]:
                print("Weight decay exclude: "+m[0])
                group_no_decay.append(m[1])
                exclude = True
                break
        if not exclude:
            print("Weight decay include: " + m[0])
            group_decay.append(m[1])

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay, weight_decay=weight_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups


def register_optimizer_builder(name, builder):
    if name in _OPTIMIZER_BUILDERS:
        raise KeyError(
            "Duplicate keys for {:s} with {} and {}."
            "Solve key conflicts first!".format(name, _OPTIMIZER_BUILDERS[name], builder))
    _OPTIMIZER_BUILDERS[name] = builder


def build_scheduler(cfg, optimizer):
    name = cfg.SCHEDULER.TYPE
    if hasattr(torch.optim.lr_scheduler, name):
        def builder(cfg, optimizer):
            return getattr(torch.optim.lr_scheduler, name)(
                optimizer,
                **cfg.SCHEDULER[name],
            )
    elif name in _OPTIMIZER_BUILDERS:
        builder = _OPTIMIZER_BUILDERS[name]
    else:
        raise ValueError("Unsupported type of optimizer.")

    return builder(cfg, optimizer)
