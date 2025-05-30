"""AutoFreeze Conv"""
import torch
from torch import nn, Tensor
from torch.nn import Conv2d
from typing import Optional, Any
from collections import defaultdict

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
import numpy as np
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from .utils import dtype_memory_size_dict, clip_tensor, Clipper

import collections
from itertools import repeat

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")

class AutoFreezeConv2d(Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros',
                 accum_mode='exp', name='conv', num=0, BN_only=False):
        super(AutoFreezeConv2d, self).__init__(in_channels, out_channels, kernel_size, stride=stride,
        padding=padding, dilation=dilation, groups=groups, bias=True if bias is not None else False, padding_mode=padding_mode)
        self.name = name
        self.num = num
        self.full_matched = False
        self.accum_mode = accum_mode
        self.clip_ratio = 0 # 0-without pruning, 1-pruning all activations
        self.sparsity_signal = False
        self.back_cache_size = 0
        self.activation_size = 0
        self.forward_compute = 0
        self.backward_compute = 0
        self.BN_only = BN_only # False-updating all layers, True-updating BN layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return AutoFreezeConv2dFunction.apply(self, True, x, self.weight, self.bias)
    
    def forward_autofreeze(self, input, weight, bias) -> torch.Tensor:
        if self.padding_mode != 'zeros':
            return F.conv2d(F.pad(input, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                            weight, bias, self.stride,
                            _pair(0), self.dilation, self.groups)
        return F.conv2d(input, weight, bias, self.stride,
                        self.padding, self.dilation, self.groups)

def forward_compute(x, kernel_size, out_channels, stride, padding, dilation):
    # Compute the computational cost of Conv forward
    N, in_channels, H, W = x.shape
    kernel_h, kernel_w = kernel_size
    H_out = (H + 2 * padding[0] - dilation[0] * (kernel_h - 1) - 1) // stride[0] + 1
    W_out = (W + 2 * padding[0] - dilation[0] * (kernel_w - 1) - 1) // stride[0] + 1
    return N * out_channels * H_out * W_out * (kernel_h * kernel_w * in_channels * 2)

def backward_compute(x, kernel_size, out_channels, stride, padding, dilation):
    # Compute the computational cost of gradients for W and x
    N, in_channels, H, W = x.shape
    dL_dW_flops = forward_compute(x, kernel_size, out_channels, stride, padding, dilation)
    dL_dX_flops = dL_dW_flops * in_channels / out_channels
    return dL_dW_flops + dL_dX_flops
    # return dL_dX_flops

class AutoFreezeConv2dFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, autofreeze_conv: AutoFreezeConv2d, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        # print(f"### x req grad: {x.requires_grad}")
        ctx.autofreeze_conv = autofreeze_conv
        ctx.preserve_rng_state = preserve_rng_state
        # Accommodates the (remote) possibility that autocast is enabled for cpu AND gpu.
        ctx.gpu_autocast_kwargs = {"enabled": torch.is_autocast_enabled(),
                                   "dtype": torch.get_autocast_gpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        ctx.cpu_autocast_kwargs = {"enabled": torch.is_autocast_cpu_enabled(),
                                   "dtype": torch.get_autocast_cpu_dtype(),
                                   "cache_enabled": torch.is_autocast_cache_enabled()}
        if preserve_rng_state:
            ctx.fwd_cpu_state = torch.get_rng_state()
            # Don't eagerly initialize the cuda context by accident.
            # (If the user intends that the context is initialized later, within their
            # run_function, we SHOULD actually stash the cuda state here.  Unfortunately,
            # we have no way to anticipate this will happen before we run the function.)
            ctx.had_cuda_in_fwd = False
            if torch.cuda._initialized:
                ctx.had_cuda_in_fwd = True
                ctx.fwd_gpu_devices, ctx.fwd_gpu_states = get_device_states(x)

        with torch.no_grad():
            y = autofreeze_conv.forward_autofreeze(x, weight, bias)

            # Computation costs
            autofreeze_conv.forward_compute = forward_compute(x, autofreeze_conv.kernel_size, autofreeze_conv.out_channels, autofreeze_conv.stride, autofreeze_conv.padding, autofreeze_conv.dilation)
            autofreeze_conv.backward_compute = backward_compute(x, autofreeze_conv.kernel_size, autofreeze_conv.out_channels, autofreeze_conv.stride, autofreeze_conv.padding, autofreeze_conv.dilation)

            mode = "min_abs"
            # Dynamic Activation Sparsity
            if autofreeze_conv.sparsity_signal: # DAS
                clip_strategy = Clipper(autofreeze_conv.clip_ratio, mode)
                autofreeze_conv.back_cache_size = int(x.numel() * (1 - autofreeze_conv.clip_ratio))
            else:
                clip_strategy = Clipper(0, mode)
                autofreeze_conv.activation_size = int(x.numel())
            ctx.clip_strategy = clip_strategy
            clipped_x = clip_strategy.clip(x, ctx)

            ctx.save_for_backward(clipped_x)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        autofreeze_conv = ctx.autofreeze_conv
        clip_strategy = ctx.clip_strategy
        x = ctx.saved_tensors
        # sparsity_signal = autofreeze_conv.sparsity_signal
        BN_only = autofreeze_conv.BN_only

        # Stash the surrounding rng state, and mimic the state that was present at this time during forward. Restore the surrounding state when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            # grad computing
            clipped_x = clip_strategy.reshape(x[0], ctx).view(ctx.x_shape)
            # grad_b = torch.sum(grad_out, (0,2,3)) if autofreeze_conv.bias is not None else None
            grad_x, grad_w_clipped = torch.ops.aten.convolution_backward(
                grad_out, clipped_x, autofreeze_conv.weight, None,
                autofreeze_conv.stride, autofreeze_conv.padding, autofreeze_conv.dilation,
                False, [0], autofreeze_conv.groups, (True, True, False)
            )[:2]

            del clip_strategy, autofreeze_conv

        if BN_only: # BN-only mode
            return None, None, grad_x, None, None
        else:
            return None, None, grad_x, grad_w_clipped, None