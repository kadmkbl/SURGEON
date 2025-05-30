"""
Builds upon: https://github.com/DequanWang/tent
Corresponding paper: https://arxiv.org/abs/2006.10726
"""

import torch.nn as nn
import torch.jit

from methods.base import TTAMethod

class Tent(TTAMethod):
    """Tent adapts a model by entropy minimization during testing."""
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt model on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        """
        imgs_test = x[0]
        outputs = self.model(imgs_test)
        loss = softmax_entropy(outputs).mean(0)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return outputs

    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms."""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
                for np, p in m.named_parameters():
                    if np in ['weight', 'bias']:
                        params.append(p)
                        names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.train()

        self.model.requires_grad_(False)

        for name, m in self.model.named_modules():
            if isinstance(m, nn.BatchNorm2d):
                m.requires_grad_(True)
                m.eps = 1e-5
                m.momentum = 0.1

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
