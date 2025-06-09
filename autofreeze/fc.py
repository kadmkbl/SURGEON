"""AutoFreeze FC"""
import torch
from torch import nn, Tensor
from torch.nn import Linear
from typing import Optional, Any
from collections import defaultdict

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
import numpy as np
from copy import deepcopy
import torch
from torch import nn
import torch.nn.functional as F
from .utils import dtype_memory_size_dict

import collections
from itertools import repeat

from .utils import dtype_memory_size_dict, clip_tensor, Clipper

def _ntuple(n, name="parse"):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return tuple(x)
        return tuple(repeat(x, n))

    parse.__name__ = name
    return parse

_pair = _ntuple(2, "_pair")

class AutoFreezeFC(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, name=None, num=0, BN_only=False):
        super(AutoFreezeFC, self).__init__(in_features, out_features, True if bias is not None else False)
        self.name = name
        self.num = num
        self.clip_ratio = 0 # 0-without pruning, 1-pruning all activations
        self.sparsity_signal = False
        self.back_cache_size = 0
        self.activation_size = 0
        self.forward_compute = 0
        self.backward_compute = 0
        self.BN_only = BN_only # False-updating all layers, True-updating BN layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return AutoFreezeFCFunction.apply(self, True, x, self.weight, self.bias)
    
    def forward_autofreeze(self, input, weight, bias) -> torch.Tensor:
        return F.linear(input, weight, bias)

def forward_compute(in_features, out_features):
    # Compute the computational cost of FC forward
    return in_features * out_features * 2

def backward_compute(in_features, out_features):
    # Compute the computational cost of gradients for W and x
    dL_dW_flops = in_features * out_features * 2
    dL_dX_flops = out_features * in_features * 2
    return dL_dW_flops + dL_dX_flops
    # return dL_dX_flops

class AutoFreezeFCFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, autofreeze_fc: AutoFreezeFC, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        # print(f"### x req grad: {x.requires_grad}")
        ctx.autofreeze_fc = autofreeze_fc
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
            y = autofreeze_fc.forward_autofreeze(x, weight, bias)

            # Computation costs
            autofreeze_fc.forward_compute = forward_compute(autofreeze_fc.in_features, autofreeze_fc.out_features)
            autofreeze_fc.backward_compute = forward_compute(autofreeze_fc.in_features, autofreeze_fc.out_features)

            mode = "min_abs"
            # Dynamic Activation Sparsity
            if autofreeze_fc.sparsity_signal: # DAS
                clip_strategy = Clipper(autofreeze_fc.clip_ratio, mode)
                autofreeze_fc.back_cache_size = int(x.numel() * (1 - autofreeze_fc.clip_ratio))
            else:
                clip_strategy = Clipper(0, mode)
                autofreeze_fc.activation_size = int(x.numel())
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

        autofreeze_fc = ctx.autofreeze_fc
        clip_strategy = ctx.clip_strategy
        x = ctx.saved_tensors
        # sparsity_signal = autofreeze_fc.sparsity_signal
        BN_only = autofreeze_fc.BN_only

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
            # grad_b = torch.sum(grad_out, list(range(grad_out.dim()-1))) if autofreeze_fc.bias is not None else None
            ic, oc = autofreeze_fc.weight.shape
            clipped_x = clip_strategy.reshape(x[0], ctx)
            grad_w = grad_out.view(-1,ic).T.mm(clipped_x.view(-1,oc))
            grad_x = torch.matmul(grad_out, autofreeze_fc.weight, out=clipped_x.view(ctx.x_shape))

            del clip_strategy, autofreeze_fc

        if BN_only: # BN-only mode
            return None, None, grad_x, None, None
        else:
            return None, None, grad_x, grad_w, None