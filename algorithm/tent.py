"""
Copyright to Tent Authors ICLR 2021 Spotlight
"""

from argparse import ArgumentDefaultsHelpFormatter
from copy import deepcopy

import torch
import torch.nn as nn
import torch.jit
from collections import defaultdict

from models.batch_norm import has_accum_bn_grad
from .base import AdaptableModule, collect_bn_params, configure_model
from utils.gpu_mem_track import MemTracker

class Tent(AdaptableModule):
    """Tent adapts a model by entropy minimization during testing.
    Once tented, a model adapts itself by updating on every forward.
    """
    def __init__(self, model, optimizer, fishers=None, training_avg=None, training_var=None, steps=1, episodic=False):
        super().__init__()
        self.model = model
        self.optimizer = optimizer
        self.steps = steps
        assert steps > 0, "tent requires >= 1 step(s) to forward and update"
        self.episodic = episodic

        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.model_state, self.optimizer_state = \
            copy_model_and_optimizer(self.model, self.optimizer)

    def forward(self, x):
        if self.episodic:
            self.reset()
        if self.steps > 0:
            for _ in range(self.steps):
                outputs = forward_and_adapt(x, self.model, self.optimizer)
        else:
            self.model.eval()
            with torch.no_grad():
                outputs = self.model(x)
        return outputs

    def reset(self):
        if self.model_state is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        load_model_and_optimizer(self.model, self.optimizer,
                                 self.model_state, self.optimizer_state)

    def reset_steps(self, new_steps):
        self.steps = new_steps

    def reset_all(self):
        self.reset()
        self.reset_steps(1)
        self.reset_bn()

    @staticmethod
    def collect_params(model, filter=None):
        """Collect the affine scale + shift parameters from batch norms.
        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.
        Note: other choices of parameterization are possible!
        """
        return collect_bn_params(model, filter=filter)

    @staticmethod
    def configure_model(model, local_bn=True, filter=None):
        """Configure model for use with tent."""
        return configure_model(model, local_bn, filter)

@torch.jit.script
def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    temprature = 1
    x = x/ temprature
    x = -(x.softmax(1) * x.log_softmax(1)).sum(1)
    return x

@torch.jit.script
def energy(x: torch.Tensor) -> torch.Tensor:
    """Energy calculation from logits."""
    temprature = 1.
    x = -(temprature*torch.logsumexp(x / temprature, dim=1))
    # if torch.rand(1) > 0.95:
    print('## energy ', x.mean(0).item())
    return x

@torch.enable_grad()  # ensure grads in possible no grad context for testing
def forward_and_adapt(x, model, optimizer):
    """Forward and adapt model on batch of data.
    Measure entropy of the model prediction, take gradients, and update params.
    """
    MemTracker.track('Before inference')
    # forward
    outputs = model(x)
    MemTracker.track('Do inference')
    # adapt
    loss = softmax_entropy(outputs).mean(0)
    if has_accum_bn_grad(model):
        loss.backward()
        MemTracker.track('Do backward')
        optimizer.step()
        optimizer.zero_grad()
        MemTracker.track('After optimizer.step')
    return outputs

def copy_model_and_optimizer(model, optimizer):
    """Copy the model and optimizer states for resetting after adaptation."""
    model_state = deepcopy(model.state_dict())
    optimizer_state = deepcopy(optimizer.state_dict())
    return model_state, optimizer_state

def load_model_and_optimizer(model, optimizer, model_state, optimizer_state):
    """Restore the model and optimizer states from copies."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)

def check_model(model):
    """Check model for compatability with tent."""
    is_training = model.training
    assert is_training, "tent needs train mode: call model.train()"
    param_grads = [p.requires_grad for p in model.parameters()]
    has_any_params = any(param_grads)
    has_all_params = all(param_grads)
    assert has_any_params, "tent needs params to update: " \
                           "check which require grad"
    assert not has_all_params, "tent should not update all params: " \
                               "check which require grad"
    has_bn = any([isinstance(m, nn.BatchNorm2d) for m in model.modules()])
    assert has_bn, "tent needs normalization for its optimization"
