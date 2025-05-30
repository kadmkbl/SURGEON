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

def clip_tensor(input_tensor, clip_ratio, mode='random'):
    """
    Args:
        input_tensor (torch.Tensor), size: [n, w, h]
        clip_ratio (float): range: [0, 1]
        mode (str): 'min_abs'
    Returns:
        torch.Tensor
    """
    input_tensor = input_tensor.clone()

    n, c, w, h = input_tensor.shape

    num_to_clip = int(clip_ratio * n * c * w * h)

    if mode == 'min_abs':
        abs_tensor = torch.abs(input_tensor)
        min_abs_indices = abs_tensor.view(-1).argsort()[:num_to_clip]
        zero_indices = min_abs_indices

    # Create a mask tensor with ones everywhere except at zero_indices
    mask_tensor = torch.ones_like(input_tensor)
    mask_tensor.view(-1)[zero_indices] = 0

    return mask_tensor

class Clipper(object):
    def __init__(self, clip_ratio, mode):
        self.clip_ratio = clip_ratio
        self.mode = mode
        self.clip = getattr(self, f"clip_{mode}")
        self.reshape = getattr(self, f"reshape_{mode}")

    def clip_min_abs(self, x: torch.Tensor, ctx=None):
        ctx.x_shape = x.shape
        numel = x.numel()
        x = x.view(-1)
        idxs = x.abs().topk(int(numel * (1 - self.clip_ratio)), sorted=False)[1]
        x = x[idxs]
        ctx.idxs = idxs.to(torch.int32)
        ctx.numel = numel
        return x

    def reshape_min_abs(self, x, ctx=None):
        idxs = ctx.idxs.to(torch.int64)
        del ctx.idxs
        return torch.zeros(
            ctx.numel, device=x.device, dtype=x.dtype
        ).scatter_(0, idxs, x)
