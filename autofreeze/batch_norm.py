"""AutoFreeze batch-norm"""
import torch
from torch import nn, Tensor
from torch.nn import BatchNorm2d
from typing import Optional, Any
from collections import defaultdict

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.functional as F
from .utils import dtype_memory_size_dict, clip_tensor, Clipper

def simple_divergence(mean1, var1, mean2, var2, eps):
    return (mean1 - mean2) ** 2 / (var2 + eps)

def gauss_kl_divergence(mean1, var1, mean2, var2, eps):
    # /// v1: relative to distribution 2 ///
    d1 = (torch.log(var2 + eps) - torch.log(var1 + eps))/2. + \
        (var1 + eps + (mean1 - mean2)**2) / 2. / (var2 + eps) - 0.5
    return d1

# @torch.jit.script
def gauss_symm_kl_divergence(mean1, var1, mean2, var2, eps):
    # >>> out-place ops
    dif_mean = (mean1 - mean2) ** 2
    d1 = var1 + eps + dif_mean
    d1.div_(var2 + eps)
    d2 = (var2 + eps + dif_mean)
    d2.div_(var1 + eps)
    d1.add_(d2)
    d1.div_(2.).sub_(1.)
    # d1 = (var1 + eps + dif_mean) / (var2 + eps) + (var2 + eps + dif_mean) / (var1 + eps)
    return d1

class AutoFreezeNorm2d(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True, track_running_stats=True,
                 accum_mode='exp', name='bn', num=0, beta=1, beta_thre=0, BN_only=False):
        super(AutoFreezeNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        self.name = name
        self.num = num
        self.full_matched = False
        self.accum_mode = accum_mode
        self.beta = beta
        self.beta_thre = beta_thre
        self.clip_ratio = 0 # 0-without pruning, 1-pruning all activations
        self.sparsity_signal = False
        self.back_cache_size = 0
        self.activation_size = 0
        self.forward_compute = 0
        self.backward_compute = 0
        self.BN_only = BN_only # False-updating all layers, True-updating BN layers

        self.bn_dist_scale = 1.
        self.update_dist_metric("skl")  # simple | kl | skl

    def update_dist_metric(self, dist_metric):
        if dist_metric == 'kl':
            self.dist_metric = gauss_kl_divergence
        elif dist_metric == 'skl':
            self.dist_metric = gauss_symm_kl_divergence
        elif dist_metric == 'simple':
            self.dist_metric = simple_divergence
        else:
            raise RuntimeError(f"Unknown distance: {dist_metric}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return AutoFreezeNorm2dFunction.apply(self, True, x, self.weight, self.bias)

    def forward_autofreeze(self, input, weight, bias) -> torch.Tensor:
        with torch.no_grad():
            batch_var, batch_mean = torch.var_mean(input, dim=(0,2,3), unbiased=False)
            running_mean, running_var = self.running_mean.clone(), self.running_var.clone()

        y = F.batch_norm(
                    input,
                    # If buffers are not to be tracked, ensure that they won't be updated
                    self.running_mean,
                    self.running_var,
                    weight,
                    bias,
                    True,
                    0.1,
                    self.eps,
                )

        return y, batch_mean, batch_var, running_mean, running_var

    def bn_forward(self, input, weight, bias, track_running_stats, training):
        """Reimplement the standard BN forward with customizable args."""
        self._check_input_dim(input)

        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if training and track_running_stats:
            # TODO: if statement only here to tell the jit to skip emitting this when it is None
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / \
                        float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum

        r"""
        Decide whether the mini-batch stats should be used for normalization rather than the buffers.
        Mini-batch stats are used in training mode, and in eval mode when buffers are None.
        """
        if training:
            bn_training = True
        else:
            bn_training = (self.running_mean is None) and (
                self.running_var is None)

        r"""
        Buffers are only updated if they are to be tracked and we are in training mode. Thus they only need to be
        passed when the update should occur (i.e. in training mode when they are tracked), or when buffer stats are
        used for normalization (i.e. in eval mode when buffers are not None).
        """
        return F.batch_norm(
            input,
            # If buffers are not to be tracked, ensure that they won't be updated
            self.running_mean if not training or track_running_stats else None,
            self.running_var if not training or track_running_stats else None,
            weight,
            bias,
            bn_training,
            exponential_average_factor,
            self.eps,
        )

    def forward_for_backward(self, x: torch.Tensor, weight, bias, running_mean, running_var, batch_mean, batch_var) -> torch.Tensor:
        self._check_input_dim(x)

        if self.full_matched:
            weight = weight.detach()
            bias = bias.detach()

        # >>>>> Fastest.
        with torch.no_grad():

            inv_r_std = torch.sqrt(running_var + self.eps)
            weight_hat = torch.sqrt(batch_var + self.eps) / inv_r_std
            bias_hat = (batch_mean - running_mean) / inv_r_std

        weight_hat = weight * weight_hat
        bias_hat = weight * bias_hat + bias

        y = F.batch_norm(x, None, None, weight_hat, bias_hat,
                             training=True, momentum=0., eps=self.eps)

        return y

def forward_compute(x):
    # Compute the computational cost of BN forward, including mean/variance calculation, normalization, scaling, and shifting
    N, C, H, W = x.shape
    mean_flops = C * H * W * N
    var_flops = C * H * W * N
    norm_flops = C * H * W * N
    scale_shift_flops = 2 * C * H * W * N
    return mean_flops + var_flops + norm_flops + scale_shift_flops
    
def backward_compute(x):
    # Compute the computational cost of gradients for gamma and x
    N, C, H, W = x.shape
    dx_flops = 2 * C * H * W * N
    dgamma_flops = C * H * W * N
    return dx_flops + dgamma_flops

class AutoFreezeNorm2dFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, autofreeze_norm: AutoFreezeNorm2d, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        # print(f"### x req grad: {x.requires_grad}")
        ctx.autofreeze_norm = autofreeze_norm
        ctx.preserve_rng_state = preserve_rng_state
        ctx.x = x.clone()
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
            y, batch_mean, batch_var, running_mean, running_var = autofreeze_norm.forward_autofreeze(x, weight, bias)

            # Computation costs
            autofreeze_norm.forward_compute = forward_compute(x)
            autofreeze_norm.backward_compute = backward_compute(x)

            mode = "min_abs"
            # Dynamic Activation Sparsity
            if autofreeze_norm.sparsity_signal: # DAS
                clip_strategy = Clipper(autofreeze_norm.clip_ratio, mode)
                autofreeze_norm.back_cache_size = int(x.numel() * (1 - autofreeze_norm.clip_ratio))
            else:
                clip_strategy = Clipper(0, mode)
                autofreeze_norm.activation_size = int(x.numel())
            ctx.clip_strategy = clip_strategy
            clipped_x = clip_strategy.clip(x, ctx)
            clipped_x.requires_grad = x.requires_grad

            ctx.save_for_backward(clipped_x, batch_mean, batch_var, running_mean, running_var)

        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        autofreeze_norm = ctx.autofreeze_norm
        clip_strategy = ctx.clip_strategy
        c_x, batch_mean, batch_var, running_mean, running_var = ctx.saved_tensors
        # sparsity_signal = autofreeze_norm.sparsity_signal
        # BN_only = autofreeze_norm.BN_only

        # Stash the surrounding rng state, and mimic the state that was present at this time during forward. Restore the surrounding state when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)

            # detach variables
            detached_x = ctx.x.detach()
            if autofreeze_norm.affine:
                weight, bias = autofreeze_norm.weight.detach(), autofreeze_norm.bias.detach()
                weight_c, bias_c = autofreeze_norm.weight.detach(), autofreeze_norm.bias.detach()
            else:
                weight, bias = None, None
                weight_c, bias_c = None, None
            if batch_mean is not None and batch_var is not None:
                batch_mean, batch_var = batch_mean.detach(), batch_var.detach()
            detached_x.requires_grad = c_x.requires_grad
            weight.requires_grad, bias.requires_grad = autofreeze_norm.weight.requires_grad, autofreeze_norm.bias.requires_grad
            weight_c.requires_grad, bias_c.requires_grad = autofreeze_norm.weight.requires_grad, autofreeze_norm.bias.requires_grad

            # grad computing
            clipped_x = clip_strategy.reshape(c_x, ctx).view(ctx.x_shape)
            clipped_x.requires_grad = c_x.requires_grad
            with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                # NOTE running_mean, running_var, batch_mean, batch_var are only needed for exp-mode of accum_bn
                y = autofreeze_norm.forward_for_backward(
                    detached_x, weight, bias, running_mean, running_var, batch_mean, batch_var)
                c_y = autofreeze_norm.forward_for_backward(
                    clipped_x, weight_c, bias_c, running_mean, running_var, batch_mean, batch_var)

            with torch.no_grad():
                if torch.is_tensor(y) and y.requires_grad:
                    torch.autograd.backward([y], [grad_out])
                if torch.is_tensor(c_y) and c_y.requires_grad:
                    torch.autograd.backward([c_y], [grad_out])

                grad_x = detached_x.grad
                grad_w = weight_c.grad if autofreeze_norm.affine else None
                # grad_b = bias_c.grad if autofreeze_norm.affine else None

            del clip_strategy, autofreeze_norm

            return None, None, grad_x, grad_w, None