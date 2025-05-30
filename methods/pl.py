import torch.nn as nn
import torch.jit

from methods.base import TTAMethod

class PL(TTAMethod):
    """Psuedo labels (PL) adapt a model by entropy minimization during testing."""
    def __init__(self, cfg, model, num_classes):
        super().__init__(cfg, model, num_classes)

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        """Forward and adapt models on batch of data.
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
        """All parameters are updated in PL."""
        params = []
        names = []
        for nm, m in self.model.named_modules():
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
        return params, names

    def configure_model(self):
        """Configure model."""
        self.model.train()
        self.model.requires_grad_(True)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)
