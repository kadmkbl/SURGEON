import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.jit
import logging

logger = logging.getLogger(__name__)

from methods.base import TTAMethod
from augmentations.transforms_cotta import get_tta_transforms
from collections import defaultdict

from autofreeze.batch_norm import AutoFreezeNorm2d
from autofreeze.conv import AutoFreezeConv2d
from autofreeze.fc import AutoFreezeFC

import numpy as np

import math

def softmax(weights_dict):
    weights = np.array(list(weights_dict.values()))
    exp_weights = np.exp(weights)
    softmax_weights = exp_weights / np.sum(exp_weights)
    softmax_dict = {k: v for k, v in zip(weights_dict.keys(), softmax_weights)}
    return softmax_dict

def log_norm(weights_dict, epsilon=1e-8):
    weights = np.array(list(weights_dict.values()))
    log_weights = np.log(weights + epsilon)
    min_log = np.min(log_weights)
    max_log = np.max(log_weights)
    log_norm_weights = (log_weights - min_log) / (max_log - min_log)
    log_norm_dict = {k: v for k, v in zip(weights_dict.keys(), log_norm_weights)}
    return log_norm_dict

class DAS(TTAMethod):
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)
        self.base_lr = self.optimizer.param_groups[0]['lr']
        self.betas = self.optimizer.param_groups[0]['betas']
        self.weight_decay = self.optimizer.param_groups[0]['weight_decay']
        self.transforms = get_tta_transforms(self.dataset_name)
        self.eps = 1e-8        
        self.grad_weight = defaultdict(lambda: 0.0)
        self.trainable_dict = {k: v for k, v in self.model.named_parameters() if v.requires_grad}
        self.tau = cfg.LAW.TAU
        if cfg.CORRUPTION.DATASET == 'cifar10_c':
            self.high_margin = math.log(10) * 0.40
        elif cfg.CORRUPTION.DATASET == 'cifar100_c':
            self.high_margin = math.log(100) * 0.40

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """
        1. Get layer-wise Gradient Importance and Memory Importance via an additional forward-backward process.
        2. Calculate layer-wise activation pruning ratios.
        3. Forward and adapt models via dynamic activation sparsity.
        """

        # 1-Setting the state of the additional forward-backward process
        self.Activation_Sparsity_Deactivate(self.model)
        # 1-An additional forward-backward process with random sampling
        imgs = x[0]
        random_indices = torch.randperm(imgs.size()[0])[:10] # Randomly select 10 samples for gradient importance calculation
        logits = self.model(imgs[random_indices])
        loss = softmax_entropy(logits)
        loss.backward(retain_graph=True)
        # 1-Get layer-wise Gradient Importance and Memory Importance
        layer_names = [n for n, param in self.model.named_parameters() if param.grad is not None]
        grad_xent = [param.grad for param in self.model.parameters() if param.grad is not None]
        param_xent = [param for param in self.model.parameters() if param.grad is not None]
        memories = self.memory_layer_cal(self.model)
        metrics = defaultdict(list)
        average_metrics = defaultdict(float)
        xent_grads = []
        xent_grads.append([g.detach() for g in grad_xent])
        for xent_grad in xent_grads:
            xent_grad_metrics = get_tgi(param_xent, xent_grad, layer_names, memories) # Combination of Importance Metrics
            for k, v in xent_grad_metrics.items():
                metrics[k].append(v)
        for k, v in metrics.items():
            average_metrics[k] = np.array(v).mean(0)

        # 2-Calculate layer-wise activation pruning ratios
        weights = average_metrics
        lr_weights_standard = defaultdict(type(weights.default_factory))
        lr_weights_standard.update(weights)
        max_weight = max(lr_weights_standard.values())
        for k, v in weights.items():
            lr_weights_standard[k] = v / max_weight
        del layer_names, grad_xent, param_xent, metrics, average_metrics, xent_grads, weights

        # 3-Setting the state of the normal forward-adapt process
        self.Activation_Sparsity_Activate(self.model, lr_weights_standard)
        # 3-Forward propagation
        logits = self.model(imgs)
        logits_aug = self.model(self.transforms(imgs))
        self.optimizer.zero_grad()
        # 3-Loss calculation
        # Original loss
        # loss = softmax_entropy(logits).mean(0)
        entropys = softmax_entropy_sample(logits)
        filter_ids_1 = torch.where(entropys < self.high_margin)
        entropys = entropys[filter_ids_1]
        coeff = 1 / (torch.exp(entropys.clone().detach() - self.high_margin))
        entropys = entropys.mul(coeff)
        # Loss with Certainty-based Sample Selection (CSS) and Consistency Regularization (CR)
        loss = entropys.mean(0) + 0.01*logits.shape[1]*consistency(logits, logits_aug)
        # 3-Model Adaptation
        loss.backward()
        self.optimizer.step()

        return logits

    def memory_layer_cal(self, net):
        layer_memories = defaultdict(list)
        memory_sum = 0
        for mod_name, target_mod in net.named_modules():
            mod_name = mod_name + ".weight"
            if isinstance(target_mod, nn.BatchNorm2d) or isinstance(target_mod, nn.Conv2d) or isinstance(target_mod, nn.Linear):
                layer_memories[mod_name] = target_mod.activation_size
                memory_sum = memory_sum + target_mod.activation_size
        return [layer_memories, memory_sum]

    def Activation_Sparsity_Activate(self, net, lr_weights_standard):
        """Setting the state of the normal forward-adapt process."""
        for mod_name, target_mod in net.named_modules():
            mod_name = mod_name + ".weight"
            if isinstance(target_mod, nn.BatchNorm2d): # Setting BN
                target_mod.sparsity_signal = True
                if mod_name in lr_weights_standard:
                    target_mod.clip_ratio = 1 - lr_weights_standard[mod_name]
                else:
                    target_mod.clip_ratio = 1
            elif isinstance(target_mod, nn.Conv2d): # Setting Conv
                target_mod.sparsity_signal = True
                if mod_name in lr_weights_standard:
                    target_mod.clip_ratio = 1 - lr_weights_standard[mod_name]
                else:
                    target_mod.clip_ratio = 1
            elif isinstance(target_mod, nn.Linear): # Setting FC
                target_mod.sparsity_signal = True
                if mod_name in lr_weights_standard:
                    target_mod.clip_ratio = 1 - lr_weights_standard[mod_name]
                else:
                    target_mod.clip_ratio = 1

    def Activation_Sparsity_Deactivate(self, net):
        """Setting the state of the additional forward-backward process."""
        for _, target_mod in net.named_modules():
            if isinstance(target_mod, nn.BatchNorm2d):  # Setting BN
                target_mod.sparsity_signal = False
            elif isinstance(target_mod, nn.Conv2d):  # Setting Conv
                target_mod.sparsity_signal = False
            elif isinstance(target_mod, nn.Linear):  # Setting FC
                target_mod.sparsity_signal = False

    def collect_params(self):
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model for use with das. load → (train_conf → optimizer)."""
        self.model = self.Auto_freeze_module(self.model, self.cfg)
        self.model = self.model.to("cuda" if torch.cuda.is_available() else "cpu")
        self.model.train()
        self.model.requires_grad_(True)
        self.params_origin = self.origin_params_collect(self.model)

    def Auto_freeze_module(self, net, cfg):
        """Load Autofreeze modules."""
        # Replace BN as AutoFreezeBN
        n_replaced = self.replace_bn(net, cfg.MODEL.ARCH, cfg.BN_ONLY)
        n_bn = self.count_bn(net)
        if n_replaced != n_bn:
            print(f"Replaced {n_replaced} BNs but actually have {n_bn}. Need to update `Auto_freeze_module`.")
        m_cnt = 0
        for m in net.modules():
            if isinstance(m, AutoFreezeNorm2d):
                m_cnt += 1
        assert n_replaced == m_cnt, f"Replaced {n_replaced} BNs but actually inserted {m_cnt} Custom_BN."

        # Replace Conv as AutoFreezeConv
        c_replaced = self.replace_conv(net, cfg.MODEL.ARCH, cfg.BN_ONLY)
        c_conv = self.count_conv(net)
        if c_replaced != c_conv:
            print(f"Replaced {c_replaced} Convs but actually have {c_conv}. Need to update `Auto_freeze_module`.")
        mc_cnt = 0
        for m in net.modules():
            if isinstance(m, AutoFreezeConv2d):
                mc_cnt += 1
        assert c_replaced == mc_cnt, f"Replaced {c_replaced} Convs but actually inserted {mc_cnt} AutoFreezeConv."

        # Replace FC as AutoFC
        f_replaced = self.replace_fc(net, cfg.MODEL.ARCH, cfg.BN_ONLY)
        f_fc = self.count_fc(net)
        if f_replaced != f_fc:
            print(f"Replaced {f_replaced} FCs but actually have {f_fc}. Need to update `Auto_freeze_module`.")
        mf_cnt = 0
        for m in net.modules():
            if isinstance(m, AutoFreezeFC):
                mf_cnt += 1
        assert f_replaced == mf_cnt, f"Replaced {f_replaced} FCs but actually inserted {mf_cnt} AutoFreezeFC."

        print("Successfully insert %d AutoFreeze_Conv layers, %d AutoFreeze_BN layers, %d AutoFreeze_FC layers,", c_conv, n_bn, f_fc)

        return net

    def replace_bn(self, model, name, BN_only, n_replaced=0, **abn_kwargs):
        copy_keys = ['eps', 'momentum', 'affine', 'track_running_stats']

        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.BatchNorm2d):
                print(f" Insert AutoFreeze-BN to ", name + '.' + mod_name)
                n_replaced += 1

                new_mod = AutoFreezeNorm2d(
                    target_mod.num_features,
                    **{k: getattr(target_mod, k) for k in copy_keys},
                    **abn_kwargs,
                    name=f'{name}.{mod_name}',
                    num=n_replaced,
                    beta_thre=0,
                    BN_only=BN_only)
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                n_replaced = self.replace_bn(
                    target_mod, name + '.' + mod_name, BN_only, n_replaced=n_replaced, **abn_kwargs)
        return n_replaced

    def replace_conv(self, model, name, BN_only, c_replaced=0):
        copy_keys = ['stride', 'padding', 'dilation', 'groups', 'bias', 'padding_mode']
        
        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.Conv2d):
                print(f" Insert AutoFreeze-Conv to ", name + '.' + mod_name)
                c_replaced += 1

                new_mod = AutoFreezeConv2d(
                    target_mod.in_channels,
                    target_mod.out_channels,
                    target_mod.kernel_size,
                    **{k: getattr(target_mod, k) for k in copy_keys},
                    name=f'{name}.{mod_name}',
                    num=c_replaced,
                    BN_only=BN_only)
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                c_replaced = self.replace_conv(
                    target_mod, name + '.' + mod_name, BN_only, c_replaced=c_replaced)
        return c_replaced

    def replace_fc(self, model, name, BN_only, f_replaced=0):
        copy_keys = []

        for mod_name, target_mod in model.named_children():
            if isinstance(target_mod, nn.Linear):
                print(f" Insert AutoFreeze-FC to ", name + '.' + mod_name)
                f_replaced += 1

                new_mod = AutoFreezeFC(
                    target_mod.in_features,
                    target_mod.out_features,
                    target_mod.bias,
                    **{k: getattr(target_mod, k) for k in copy_keys},
                    name=f'{name}.{mod_name}',
                    num=f_replaced,
                    BN_only=BN_only)
                new_mod.load_state_dict(target_mod.state_dict())
                setattr(model, mod_name, new_mod)
            else:
                f_replaced = self.replace_fc(
                    target_mod, name + '.' + mod_name, BN_only, f_replaced=f_replaced)
        return f_replaced

    def count_bn(self, model: nn.Module):
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                cnt += 1
        return cnt

    def count_conv(self, model: nn.Module):
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                cnt += 1
        return cnt

    def count_fc(self, model: nn.Module):
        cnt = 0
        for n, m in model.named_modules():
            if isinstance(m, nn.Linear):
                cnt += 1
        return cnt

    @staticmethod
    def check_model(model):
        """Check model for compatability with law."""
        is_training = model.training
        assert is_training, "law needs train mode: call model.train()"

    def origin_params_collect(self, model):
        params_origin = {}
        for name, param in model.named_parameters():
            if 'weight' in name:
                params_origin[name] = param.data.clone().detach()
        return params_origin

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1).mean()

@torch.jit.script
def softmax_entropy_sample(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)

@torch.jit.script
def consistency(x: torch.Tensor, y:torch.Tensor) -> torch.Tensor:
    """Consistency loss between two softmax distributions."""
    return -(x.softmax(1) * y.log_softmax(1)).sum(1).mean()

def get_grad_norms(params, grads, layer_names):
    _metrics = defaultdict(list)
    for (layer_name, param, grad) in zip(layer_names, params, grads):
        _metrics[layer_name] = torch.norm(grad).item() / torch.norm(param).item()
    return _metrics

def get_tgi(params, grads, layer_names, memories):
    """Combination of Gradient Importance and Memory Importance."""
    layer_memories = memories[0].values()
    memory_sum = memories[1]
    _metrics = defaultdict(list)
    for (layer_name, param, grad, memory) in zip(layer_names, params, grads, layer_memories):
        grad_norm = torch.norm(grad).item()
        _metrics[layer_name] = grad_norm / grad.numel()**0.5 * math.log(memory_sum / memory)
    return _metrics
