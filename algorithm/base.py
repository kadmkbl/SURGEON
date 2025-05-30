import torch
import torch.nn as nn
from collections import defaultdict

from models.batch_norm import MectaNorm2d

class AdaptableModule(nn.Module):
    """Module that can adapt model at test time."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError()

    def reset(self):
        raise NotImplementedError()

    def reset_all(self):
        raise NotImplementedError()

    def reset_bn(self):
        for m in self.model.modules():
            if isinstance(m, MectaNorm2d):
                m.reset()

    @staticmethod
    def collect_params(model):
        """Collect parameters to update."""
        raise NotImplementedError()

    @staticmethod
    def configure_model(model):
        """Configure model, e.g., training status, gradient requirements."""
        raise NotImplementedError()

def configure_model(model, local_bn=True, filter=None):
    """Configure model for use with eata."""
    filter = parse_filter(model, filter)
    # train mode, because eata optimizes the model to minimize entropy
    model.train()
    # disable grad, to (re-)enable only what eata updates
    model.requires_grad_(False)
    # configure norm for eata updates: enable grad + force batch statisics
    for nm, m in model.named_modules():
        if isinstance(m, nn.BatchNorm2d):
            if filter is not None and not filter(nm):
                continue
            # print(f" # require grad for {nm}")
            m.requires_grad_(True)
            if local_bn:
                # force use of batch stats in train and eval modes
                m.track_running_stats = False
                m.running_mean = None
                m.running_var = None
    return model

def collect_bn_params(model, filter=None):
    filter = parse_filter(model, filter)
    params = defaultdict(list)
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm2d,)):
            if filter is not None and not filter(nm):
                continue
            for np, p in m.named_parameters():
                if np in ['weight', 'bias']:  # weight is scale, bias is shift
                    params['affine'].append(p)
                    names.append(f"{nm}.{np}")
            print(f' train module: {nm}')
    return params, names

def parse_filter(model, filter):
    if filter is not None:
        if isinstance(filter, str):
            from models.eata_resnet import ResNet
            assert filter.startswith('layer')
            assert isinstance(
                model, ResNet), f"Unsupported model arch: {type(model)}"
            start, end = filter[len('layer'):].split('-')
            start, end = int(start), int(end)
            def filter(nm): return start <= int(nm[len('layer'):].split(
                '.')[0]) <= end if nm.startswith('layer') else start == 0
        elif isinstance(filter, int):
            n_layer = filter
            cnt = 0
            layer_names = []
            for n, m in model.named_modules():
                if 'downsample' in n:
                    continue
                if isinstance(m, nn.BatchNorm2d):
                    cnt += 1
                    layer_names.append(n)
            if n_layer > 0:
                layer_names = layer_names[-n_layer:]
            else:
                layer_names = layer_names[:-n_layer]
            assert len(layer_names) == abs(n_layer)
            def filter(n): return n in layer_names
        else:
            raise RuntimeError(f"Unknown filter: {filter}")
    return filter
