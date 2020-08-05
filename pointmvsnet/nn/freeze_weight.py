"""Helpers for operating weights/params"""
import re
import torch.nn as nn

__all__ = ['freeze_bn', 'freeze_params', 'freeze_modules', 'freeze_by_patterns']


def freeze_bn(module, bn_eval, bn_frozen):
    """Freeze Batch Normalization in Module

    Args:
        module (torch.nn.Module):
        bn_eval (bool): flag to using global stats
        bn_frozen (bool): flag to freeze bn params

    Returns:

    """
    for module_name, m in module.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if bn_eval:
                # Notice the difference between the behaviors of
                # BatchNorm.eval() and BatchNorm(track_running_stats=False)
                m.eval()
                # print('BN: %s in eval mode.' % module_name)
            if bn_frozen:
                for param_name, params in m.named_parameters():
                    params.requires_grad = False
                    # print('BN: %s is frozen.' % (module_name + '.' + param_name))


def freeze_params(module, frozen_params):
    """Freeze params and/or convert them into eval mode

    Args:
        module (torch.nn.Module):
        frozen_params: a list/tuple of strings,
            which define all the patterns of interests

    Returns:

    """
    for name, params in module.named_parameters():
        print(name)
        for pattern in frozen_params:
            assert isinstance(pattern, str)
            if re.search(pattern, name):
                params.requires_grad = False
                print('Params %s is frozen.' % name)


def freeze_modules(module, frozen_modules, prefix=''):
    """Set module's eval mode and freeze its params

    Args:
        module (torch.nn.Module):
        frozen_modules (list[str]):

    Returns:

    """
    for name, m in module._modules.items():
        for pattern in frozen_modules:
            assert isinstance(pattern, str)
            full_name = prefix + ('.' if prefix else '') + name
            if re.search(pattern, full_name):
                m.eval()
                _freeze_all_params(m)
                print('Module %s is frozen.' % full_name)
            else:
                freeze_modules(m, frozen_modules, prefix=full_name)


def freeze_by_patterns(module, patterns):
    """Freeze Module by matching patterns"""
    frozen_params = []
    frozen_modules = []
    for pattern in patterns:
        if pattern.startswith('module:'):
            frozen_modules.append(pattern[7:])
        else:
            frozen_params.append(pattern)
    freeze_params(module, frozen_params)
    freeze_modules(module, frozen_modules)


def _freeze_all_params(module):
    """Freeze all params in a module"""
    for name, params in module.named_parameters():
        params.requires_grad = False


def unfreeze_params(module, frozen_params):
    """Unfreeze params and/or convert them into eval mode

    Args:
        module (torch.nn.Module):
        frozen_params: a list/tuple of strings,
            which define all the patterns of interests

    Returns:

    """
    for name, params in module.named_parameters():
        print(name)
        for pattern in frozen_params:
            assert isinstance(pattern, str)
            if re.search(pattern, name):
                params.requires_grad = True
                print('Params %s is unfrozen.' % name)


def unfreeze_modules(module, frozen_modules, prefix=''):
    """Set module's eval mode and freeze its params

    Args:
        module (torch.nn.Module):
        frozen_modules (list[str]):

    Returns:

    """
    for name, m in module._modules.items():
        for pattern in frozen_modules:
            assert isinstance(pattern, str)
            full_name = prefix + ('.' if prefix else '') + name
            if re.search(pattern, full_name):
                m.eval()
                _unfreeze_all_params(m)
                print('Module %s is unfrozen.' % full_name)
            else:
                unfreeze_modules(m, frozen_modules, prefix=full_name)


def unfreeze_by_patterns(module, patterns):
    """Defrost Module by matching patterns"""
    unfreeze_params = []
    unfreeze_modules = []
    for pattern in patterns:
        if pattern.startswith('module:'):
            unfreeze_modules.append(pattern[7:])
        else:
            unfreeze_params.append(pattern)


def _unfreeze_all_params(module):
    """Freeze all params in a module"""
    for name, params in module.named_parameters():
        params.requires_grad = True
        print('Params %s is unfrozen.' % name)
