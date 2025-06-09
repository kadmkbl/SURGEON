import torch
import numpy as np
import random
import copy
import argparse


# ref: https://github.com/Oldpan/Pytorch-Memory-Utils/blob/master/gpu_mem_track.py
dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/6,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}


def set_seed(seed, cuda):

    # Make as reproducible as possible.
    # Please note that pytorch does not let us make things completely reproducible across machines.
    # See https://pytorch.org/docs/stable/notes/randomness.html
    print('setting seed', seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def unwrap_module(wrapped_state_dict):
    is_wrapped_module = False
    for k in wrapped_state_dict:
        if k.startswith('module.'):
            is_wrapped_module = True
            break
    if is_wrapped_module:
        state_dict = {k[len('module.'):]: v for k, v in wrapped_state_dict.items()}
    else:
        state_dict = copy.copy(wrapped_state_dict)
    return state_dict


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() == 'true':
        return True
    elif v.lower() == 'false':
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')
