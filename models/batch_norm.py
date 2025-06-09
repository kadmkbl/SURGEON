"""MECTA batch-norm"""
import torch
from torch import nn, Tensor
from torch.nn import BatchNorm2d
from typing import Optional, Any

from torch.utils.checkpoint import check_backward_validity, get_device_states, set_device_states, detach_variable, checkpoint
import numpy as np
from copy import deepcopy
import torch
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm2d
import torch.nn.functional as F
from util.utils import dtype_memory_size_dict

class SlowBatchNorm2d(BatchNorm2d):
    """Reimplement batch norm without C++ speedup."""
    def forward(self, x):
        batch_mean = torch.mean(x, dim=(0, 2, 3))
        batch_var = torch.var(x, dim=(0, 2, 3), unbiased=False)
        return batch_norm(x, batch_mean, batch_var, self.weight, self.bias, self.eps)

class MectaNorm2d(BatchNorm2d):
    def __init__(self, num_features, eps=1e-5, momentum=0.1, affine=True,
                 track_running_stats=True,
                 accum_mode='exp', beta=0.1, use_forget_gate=False,
                 init_beta=None,
                 verbose=False, dist_metric='skl', bn_dist_scale=1.,
                 beta_thre=0., prune_q=0., name='bn',
                 ):
        super(MectaNorm2d, self).__init__(
            num_features, eps=eps, momentum=momentum, affine=affine,
            track_running_stats=track_running_stats)
        # self.train(False)
        self.name = name
        # self.accum_mode = 'none' if not use_forget_gate and beta == 1. else accum_mode
        self.accum_mode = accum_mode
        self.beta = beta
        self.init_beta = self.beta if init_beta is None else init_beta
        self.past_size = 0
        self.use_forget_gate = use_forget_gate

        self.verbose = verbose

        self.bn_dist_scale = bn_dist_scale
        self.update_dist_metric(dist_metric)  # simple | kl | skl

        self.beta_thre = beta_thre
        self.full_matched = False
        self.prune_q = prune_q

        self.forward_cache_size = None

    def update_dist_metric(self, dist_metric):
        if dist_metric == 'kl':
            self.dist_metric = gauss_kl_divergence
        elif dist_metric == 'skl':
            self.dist_metric = gauss_symm_kl_divergence
        elif dist_metric == 'simple':
            self.dist_metric = simple_divergence
        else:
            raise RuntimeError(f"Unknown distance: {dist_metric}")

    def reset(self):
        # if not keep_stats:
        #     self.reset_running_stats()
        self.past_size = 0
        self.forward_cache_size = None
        self.full_matched = False

    def cache_size(self):
        assert self.forward_cache_size is not None, "Please call this after forward!"
        forward_cache = self.forward_cache_size
        backward_cache = 0.
        if self.weight.requires_grad and not self.full_matched:
            backward_cache = np.prod(forward_cache)
            backward_cache *= dtype_memory_size_dict[torch.float]
        forward_cache = np.prod(forward_cache) * \
            dtype_memory_size_dict[torch.float]
        # else:
        return forward_cache, backward_cache

    def set_accum_mode(self, accum_mode):
        self.accum_mode = accum_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # if x.requires_grad or self.weight.requires_grad or self.bias.requires_grad:
        self.forward_cache_size = list(x.shape)
        self.forward_cache_size[1] = int(
            (1.-self.prune_q) * self.forward_cache_size[1])
        return StochCacheFunction.apply(self, True, x, self.weight, self.bias)
    
    def forward_w_update_stats(self, x: torch.Tensor, weight, bias, disable_PreG=False, return_batch_stats=False) -> torch.Tensor:
        if self.accum_mode == 'exp':  # suitable for batch streaming
            x_size = len(x)

            with torch.no_grad():
                batch_var, batch_mean = torch.var_mean(x, dim=(0,2,3), unbiased=False)

                if self.use_forget_gate:
                    if self.running_mean is None:
                        raise RuntimeError(f"Not found running_mean.")
                    if self.running_var is None:
                        raise RuntimeError(f"Not found running_var.")

                    dist = self.dist_metric(batch_mean, batch_var,
                                            self.running_mean, self.running_var,
                                            eps=1e-3)  # self.eps) Small eps can reduce the sensitivity to unstable small variance.
                    beta = 1. - torch.exp(- self.bn_dist_scale * dist.mean())

                    self.beta = beta.item() # if hasattr(beta, 'item') else beta
                else:  # use constant beta
                    beta = self.beta if self.past_size > 0 else self.init_beta

                if beta < 1.:  # accumulate
                    self.running_mean.mul_((1-beta)).add_(batch_mean.mul(beta))
                    self.running_var.mul_((1-beta)).add_(batch_var.mul(beta))
                else:
                    self.running_mean.data.copy_(batch_mean)
                    self.running_var.data.copy_(batch_var)

                if self.use_forget_gate:
                    if beta > self.beta_thre:
                    # if beta > 0:
                        self.full_matched = False
                        if weight is not None and bias is not None:
                            weight.requires_grad, bias.requires_grad = True, True
                    else:  # stop grad
                        self.full_matched = True
                        # detach to avoid cache and grad.
                        if weight is not None and bias is not None:
                            weight.requires_grad, bias.requires_grad = False, False
                            # FIXME not necessary?
                            weight, bias = weight.detach(), bias.detach()

            if not disable_PreG:
                # >>>>> Fastest.
                with torch.no_grad():
                    # reparameterize weight,bias
                    inv_r_std = torch.sqrt(self.running_var + self.eps)
                    weight_hat = torch.sqrt(batch_var + self.eps) / inv_r_std
                    bias_hat = (batch_mean - self.running_mean) / inv_r_std

                weight_hat = weight * weight_hat
                bias_hat = weight * bias_hat + bias

                y = F.batch_norm(x, None, None, weight_hat, bias_hat,
                                training=True, momentum=0., eps=self.eps)
            else:
                y = F.batch_norm(x, self.running_mean, self.running_var, weight,
                                 bias, training=False, momentum=0., eps=self.eps)

            # update stat
            self.past_size += x_size
        elif self.accum_mode == 'none':  # standard BN
            y = self.bn_forward(x, self.weight, self.bias, self.track_running_stats, self.training)
            batch_mean, batch_var = None, None
        else:
            raise NotImplementedError(f"accum_mode: {self.accum_mode}")

        if return_batch_stats:
            return y, batch_mean, batch_var
        else:
            return y

    def forward_wo_update_stats(self, x: torch.Tensor, weight, bias, running_mean, running_var, batch_mean, batch_var) -> torch.Tensor:
        if self.accum_mode == 'exp':  # suitable for batch streaming
            # batch_mean, batch_var = self.stats
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

        elif self.accum_mode == 'none':  # standard BN
            # NOTE the results make no difference, if eval mode is never used. For training, always only the batch_mean/var are used.
            assert self.training
            y = self.bn_forward(x, weight, bias,
                                track_running_stats=False, training=True)
        else:
            raise NotImplementedError(f"accum_mode: {self.accum_mode}")

        return y

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

class StochCacheFunction(torch.autograd.Function):
    """Will stochastically cache data for backwarding."""

    @staticmethod
    def forward(ctx, mecta_norm: MectaNorm2d, preserve_rng_state, x, weight, bias):
        check_backward_validity([x, weight, bias])
        # print(f"### x req grad: {x.requires_grad}")
        ctx.mecta_norm = mecta_norm
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
            # outputs, req_cache = run_function(x)
            y, batch_mean, batch_var = mecta_norm.forward_w_update_stats(x, weight, bias, disable_PreG=True, return_batch_stats=True)
            req_cache = not mecta_norm.full_matched

            ctx.req_cache = req_cache
            if req_cache:  # require cache for norm grad or weight.
                # x = tensor_inputs[0]
                n_channels = x.size(1)
                n_rm = int(n_channels * mecta_norm.prune_q)
                ctx.n_rm = n_rm
                if n_rm > 0:
                    idxs = torch.randperm(n_channels, device=x.device)

                    ctx.remained_idxs = idxs[n_rm:]
                    ctx.removed_idxs = idxs[:n_rm]
                    ctx.n_channels = n_channels

                    x_slice = x.index_select(1, ctx.remained_idxs)
                    x_slice.requires_grad = x.requires_grad
                    x = x_slice

                    if batch_mean is not None and batch_var is not None:
                        batch_mean = batch_mean.index_select(0, ctx.remained_idxs)
                        batch_var = batch_var.index_select(0, ctx.remained_idxs)
                ctx.save_for_backward(x, batch_mean, batch_var)
        # else will not save tensor.
        return y

    @staticmethod
    def backward(ctx, grad_out):
        if not torch.autograd._is_checkpoint_valid():
            raise RuntimeError(
                "Checkpointing is not compatible with .grad() or when an `inputs` parameter"
                " is passed to .backward(). Please use .backward() and do not pass its `inputs`"
                " argument.")

        if not ctx.req_cache:  # no intermediate grad
            return None, None, grad_out, None, None

        mecta_norm = ctx.mecta_norm  # type: MectaNorm2d
        x, batch_mean, batch_var = ctx.saved_tensors
        # x, batch_mean, batch_var = 0, 0, 0

        # Stash the surrounding rng state, and mimic the state that was
        # present at this time during forward.  Restore the surrounding state
        # when we're done.
        rng_devices = []
        if ctx.preserve_rng_state and ctx.had_cuda_in_fwd:
            rng_devices = ctx.fwd_gpu_devices
        with torch.random.fork_rng(devices=rng_devices, enabled=ctx.preserve_rng_state):
            if ctx.preserve_rng_state:
                torch.set_rng_state(ctx.fwd_cpu_state)
                if ctx.had_cuda_in_fwd:
                    set_device_states(ctx.fwd_gpu_devices, ctx.fwd_gpu_states)
            detached_x = x.detach()
            if mecta_norm.affine:
                weight, bias = mecta_norm.weight.detach(), mecta_norm.bias.detach()
            else:
                weight, bias = None, None
            if mecta_norm.accum_mode == 'exp':
                running_mean, running_var = mecta_norm.running_mean.detach(), mecta_norm.running_var.detach()
            else:
                running_mean, running_var = None, None
            if batch_mean is not None and batch_var is not None:
                batch_mean, batch_var = batch_mean.detach(), batch_var.detach()
            if ctx.n_rm > 0:
                # Get remained sliced params
                with torch.no_grad():
                    weight, bias = mecta_norm.weight[ctx.remained_idxs], mecta_norm.bias[ctx.remained_idxs]
                    if mecta_norm.accum_mode == 'exp':
                        running_mean, running_var = running_mean[ctx.remained_idxs], mecta_norm.running_var[ctx.remained_idxs]
            detached_x.requires_grad = x.requires_grad
            weight.requires_grad, bias.requires_grad = mecta_norm.weight.requires_grad, mecta_norm.bias.requires_grad

            with torch.enable_grad(), \
                torch.cuda.amp.autocast(**ctx.gpu_autocast_kwargs), \
                torch.cpu.amp.autocast(**ctx.cpu_autocast_kwargs):
                # NOTE running_mean, running_var, batch_mean, batch_var are only needed for exp-mode of accum_bn
                y = mecta_norm.forward_wo_update_stats(
                    detached_x, weight, bias, running_mean, running_var, batch_mean, batch_var)

            with torch.no_grad():
                if ctx.n_rm > 0:
                    remained_idxs = ctx.remained_idxs
                    removed_idxs = ctx.removed_idxs

                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out[:, remained_idxs, :, :]])

                    grad_x, grad_w, grad_b = grad_refill(grad_out, removed_idxs, remained_idxs, mecta_norm.weight, mecta_norm.bias, 
                                                         mecta_norm.running_var, mecta_norm.eps, weight.grad, bias.grad, detached_x.grad)
                else:
                    # run backward() with only tensor that requires grad
                    if torch.is_tensor(y) and y.requires_grad:
                        torch.autograd.backward([y], [grad_out])

                    # retrive input grad
                    grad_x = detached_x.grad
                    grad_w = weight.grad if mecta_norm.affine else None
                    grad_b = bias.grad if mecta_norm.affine else None

        return None, None, grad_x, grad_w, grad_b

# @torch.jit.script
def grad_refill(grad_out, removed_idxs, remained_idxs, full_weight, full_bias, running_var, eps,
                weight_grad, bias_grad, detached_x_grad):
    """Refill the dropped gradients with constants."""
    grad_rm = grad_out[:, removed_idxs, :, :] # # size[batch_size, n_rm, dim, dim2]
    grad_rm_sum = torch.sum(grad_rm, dim=(0, 2, 3), keepdim=True) # size[1, n_rm, 1, 1]

    if detached_x_grad is None:
        grad_x = None
    else:
        grad_x = torch.zeros_like(grad_out) # 0

        # fill in missing grad
        ch_n = grad_out.shape[0] * grad_out.shape[2] * grad_out.shape[3]
        grad_x_factor = full_weight[removed_idxs] / torch.sqrt(running_var[removed_idxs] + eps) # ∂grad_y/∂grad_x, size[n_rm]
        # grad_x[:, removed_idxs, :, :] = - grad_rm_sum / ch_n * grad_x_factor.view((1,-1,1,1)) + grad_rm * grad_x_factor.view((1,-1,1,1))
        # cal_1 = - grad_rm_sum / ch_n * grad_x_factor.view((1, -1, 1, 1))
        # cal_2 = grad_rm * grad_x_factor.view((1, -1, 1, 1))
        # cal = cal_1 + cal_2
        grad_x.index_copy_(
            1, removed_idxs,
            - grad_rm_sum / ch_n *
            grad_x_factor.view((1, -1, 1, 1)) + grad_rm *
            grad_x_factor.view((1, -1, 1, 1))
        ) # c * ∂grad_y/∂grad_x + ∂grad_l/∂grad_y * ∂grad_y/∂grad_x

        # grad_x[:, remained_idxs, :] = detached_x_grad
        grad_x.index_copy_(1, remained_idxs, detached_x_grad)

    if full_weight is not None and weight_grad is not None:
        grad_w = torch.zeros_like(full_weight)
        # grad_w[remained_idxs] = weight_grad
        grad_w.index_copy_(0, remained_idxs, weight_grad)
    else:
        grad_w = None
    if full_bias is not None and bias_grad is not None:
        grad_b = torch.zeros_like(full_bias)
        # grad_b[remained_idxs] = bias_grad
        # grad_b[removed_idxs] = grad_rm_sum.squeeze()
        grad_b.index_copy_(0, remained_idxs, bias_grad)
        grad_b.index_copy_(0, removed_idxs, grad_rm_sum.squeeze())
    else:
        grad_b = None

    return grad_x, grad_w, grad_b

class standard_bn_cxt:
    """Make AccumBN into standard BN within context.
    Use this to disable adaptation.
    1. Set accum mode to none.
    2. Set prune_q=0.
    """

    def __init__(self, model: nn.Module):
        self.model = model
        self.accum_modes = {}
        self.prune_q = {}

    def __enter__(self):
        self.accum_modes = {}
        for name, m in self.model.named_modules():
            if isinstance(m, MectaNorm2d):
                self.accum_modes[name] = m.accum_mode
                self.prune_q[name] = m.prune_q
                m.set_accum_mode('none')
                m.prune_q = 0.

    def __exit__(self, exc_type, exc_val, exc_tb):
        for name, m in self.model.named_modules():
            if isinstance(m, MectaNorm2d):
                m.set_accum_mode(self.accum_modes[name])
                m.prune_q = self.prune_q[name]
        self.accum_modes = {}
        self.prune_q = {}

def get_bn_cache_size(model: nn.Module, return_dict=False):
    max_forward_cs = 0.
    total_backward_cs = 0.
    n_accum_bn = 0
    module_cache_size = {}
    for m in model.modules():
        if isinstance(m, MectaNorm2d):
            forward_cs, backward_cs = m.cache_size()
            # print(f" {m.name}: {backward_cs/1e6:.3f}")
            if forward_cs > max_forward_cs:
                max_forward_cs = forward_cs
            total_backward_cs += backward_cs
            module_cache_size[m.name] = (forward_cs, backward_cs)
            n_accum_bn += 1
    if n_accum_bn == 0:
        max_forward_cs, total_backward_cs = None, None
    if return_dict:
        return max_forward_cs, total_backward_cs, module_cache_size
    else:
        return max_forward_cs, total_backward_cs

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

def mmd_divergence(mean1, var1, mean2, var2):
    d1 = torch.sqrt((var1 - var2) ** 2 + (mean1 - mean2) ** 2)
    return d1

def batch_norm(x, current_mean, current_var, weight, bias, eps):
    """BN enabling BP using given stats."""
    eps = torch.tensor([eps], dtype=current_var.dtype, device=current_var.device)
    _var = torch.sqrt(torch.maximum(current_var, eps)).view((1, -1, 1, 1)).detach()
    _mean = current_mean.view((1, -1, 1, 1))
    x_norm = (x - _mean) / _var
    # x_norm = (x - current_mean.view((1, -1, 1, 1))) / torch.sqrt(current_var + eps).view((1, -1, 1, 1))

    # re-norm refer to 'Online Normalization for Training Neural Networks'
    # x_norm = x_norm / torch.sqrt(torch.mean(x_norm**2, dim=(0,2,3), keepdim=True))
    # print(f"$$$ self.weight {self.weight.shape} self.bias {self.bias.shape}")
    if weight is not None and bias is not None:
        y = x_norm * weight.view((1, -1, 1, 1)) + bias.view((1, -1, 1, 1))
    else:
        y = x_norm
    return y

def get_last_beta(model: nn.Module):
    all_betas = []
    for m in model.modules():
        if isinstance(m, MectaNorm2d):
            all_betas.append(m.beta)
    return all_betas

def has_accum_bn_grad(model):
    """Return True, if at least one param has grad."""
    all_mached = True
    has_acc_bn = False
    for m in model.modules():
        if isinstance(m, MectaNorm2d):
            has_acc_bn = True
            if not m.full_matched:
                all_mached = False
                break
    if has_acc_bn and all_mached:
        return False
    return True
