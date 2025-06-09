"""Source: https://pastebin.com/V3wATa7w
Discussion: https://dev-discuss.pytorch.org/t/the-ideal-pytorch-flop-counter-with-torch-dispatch/505

Require pytorch-nightly (Last check: Nov. 14, 2022)
Not working on Pytorch 1.13.0 or lower versions.
"""
import torch
# import numpy as np


import torch
# import torch.nn as nn
from torch.utils._pytree import tree_map, tree_flatten
from typing import List, Any, Callable, Union, Optional
import typing
from numbers import Number
from collections import defaultdict
from torch.utils._python_dispatch import TorchDispatchMode


Handle = Callable[[List[Any], List[Any]], Union[typing.Counter[str], Number]]
aten = torch.ops.aten
 
def get_shape(i):
    return i.shape


# def get_shape(val: Any) -> Optional[List[int]]:
#     """
#     Get the shapes from a jit value object.
#     Args:
#         val (torch._C.Value): jit value object.
#     Returns:
#         list(int): return a list of ints.
#     """
#     if val.isCompleteTensor():
#         return val.type().sizes()
#     else:
#         return None
 
def prod(x):
    res = 1
    for i in x:
        res *= i
    return res

def void_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    return 0.

def elem_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """Count flops for element-wise operations."""
    return outputs[0].numel()


def norm_flop_counter(affine_arg_index: int) -> Handle:
    """
    Args:
        affine_arg_index: index of the affine argument in inputs
    """

    def norm_flop_jit(inputs: List[Any], outputs: List[Any]) -> Number:
        """
        Count flops for norm layers.
        """
        # Inputs[0] contains the shape of the input.
        input_shape = get_shape(inputs[0])
        has_affine = get_shape(inputs[affine_arg_index]) is not None
        assert 2 <= len(input_shape) <= 5, input_shape
        # 5 is just a rough estimate
        flop = prod(input_shape) * (6 if has_affine else 4)
        return flop

    return norm_flop_jit


def batchnorm_flop_fwd(inputs: List[Any], outputs: List[Any]) -> Number:
    training = inputs[5]#.toIValue()
    assert isinstance(
        training, bool), "Signature of aten::batch_norm has changed!"
    if training:
        return norm_flop_counter(1)(inputs, outputs)  # pyre-ignore
    has_affine = get_shape(inputs[1]) is not None
    input_shape = prod(get_shape(inputs[0]))
    return input_shape * (4 if has_affine else 2)


def batchnorm_flop_bwd(inputs: List[Any], outputs: List[Any]) -> Number:
    # Inputs[0] contains the shape of the input.
    input_shape = inputs[1].shape
    has_affine = inputs[4] is not None
    assert 2 <= len(input_shape) <= 5, input_shape
    # 5 is just a rough estimate
    flop = prod(input_shape) * (6 if has_affine else 4)
    return flop


def matmul_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for matmul.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two matrices.
    input_shapes = [get_shape(v) for v in inputs]
    assert len(input_shapes) == 2, input_shapes
    assert input_shapes[0][-1] == input_shapes[1][-2], input_shapes
    flop = prod(input_shapes[0]) * input_shapes[-1][-1]
    return flop
 
def addmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for fully connected layers.
    """
    # Count flop for nn.Linear
    # inputs is a list of length 3.
    input_shapes = [get_shape(v) for v in inputs[1:3]]
    # input_shapes[0]: [batch size, input feature dimension]
    # input_shapes[1]: [batch size, output feature dimension]
    assert len(input_shapes[0]) == 2, input_shapes[0]
    assert len(input_shapes[1]) == 2, input_shapes[1]
    batch_size, input_dim = input_shapes[0]
    output_dim = input_shapes[1][1]
    flops = batch_size * input_dim * output_dim
    return flops
 
def bmm_flop(inputs: List[Any], outputs: List[Any]) -> Number:
    """
    Count flops for the bmm operation.
    """
    # Inputs should be a list of length 2.
    # Inputs contains the shapes of two tensor.
    assert len(inputs) == 2, len(inputs)
    input_shapes = [get_shape(v) for v in inputs]
    n, c, t = input_shapes[0]
    d = input_shapes[-1][-1]
    flop = n * c * t * d
    return flop
 
def conv_flop_count(
    x_shape: List[int],
    w_shape: List[int],
    out_shape: List[int],
    transposed: bool = False,
) -> Number:
    """
    Count flops for convolution. Note only multiplication is
    counted. Computation for addition and bias is ignored.
    Flops for a transposed convolution are calculated as
    flops = (x_shape[2:] * prod(w_shape) * batch_size).
    Args:
        x_shape (list(int)): The input shape before convolution.
        w_shape (list(int)): The filter shape.
        out_shape (list(int)): The output shape after convolution.
        transposed (bool): is the convolution transposed
    Returns:
        int: the number of flops
    """
    batch_size = x_shape[0]
    conv_shape = (x_shape if transposed else out_shape)[2:]
    flop = batch_size * prod(w_shape) * prod(conv_shape)
    return flop
 
def conv_flop(inputs: List[Any], outputs: List[Any]):
    """
    Count flops for convolution.
    """
    x, w = inputs[:2]
    x_shape, w_shape, out_shape = (get_shape(x), get_shape(w), get_shape(outputs[0]))
    transposed = inputs[6]
 
    return conv_flop_count(x_shape, w_shape, out_shape, transposed=transposed)
 
def transpose_shape(shape):
    return [shape[1], shape[0]] + list(shape[2:])
 
def conv_backward_flop(inputs: List[Any], outputs: List[Any]):
    grad_out_shape, x_shape, w_shape = [get_shape(i) for i in inputs[:3]]
    output_mask = inputs[-1]
    fwd_transposed = inputs[7]
    flop_count = 0
 
    if output_mask[0]:
        grad_input_shape = get_shape(outputs[0])
        flop_count += conv_flop_count(grad_out_shape, w_shape, grad_input_shape, not fwd_transposed)
    if output_mask[1]:
        grad_weight_shape = get_shape(outputs[1])
        flop_count += conv_flop_count(transpose_shape(x_shape), grad_out_shape, grad_weight_shape, fwd_transposed)
 
    return flop_count

def index_flop(inputs: List[Any], outputs: List[Any]):
    return prod(inputs[3].shape)


def index_select_flop(inputs: List[Any], outputs: List[Any]):
    return prod(outputs[0].shape)
 
flop_mapping = {
    aten.mm: matmul_flop,
    aten.matmul: matmul_flop,
    aten.addmm: addmm_flop,
    aten.bmm: bmm_flop,
    # built-in funcs
    aten.convolution: conv_flop,
    aten._convolution: conv_flop,
    aten.convolution_backward: conv_backward_flop,
    aten.cudnn_batch_norm: batchnorm_flop_fwd,
    aten.cudnn_batch_norm_backward: batchnorm_flop_bwd,
    aten.detach: void_flop,
    aten.clone: void_flop,
    aten.view: void_flop,
    aten.slice: void_flop,  # only create view
    aten.index: void_flop,  # only create view
    # aten.slice: void_flop,  # only create view
    # element-wise operations
    aten.pow: elem_flop,
    aten.exp: elem_flop,
    aten.lt: elem_flop,
    aten.gt: elem_flop,
    aten.add_: elem_flop,
    aten.add: elem_flop,
    aten.sub_: elem_flop,
    aten.sub: elem_flop,
    aten.rsub: elem_flop,
    aten.mul_: elem_flop,
    aten.mul: elem_flop,
    aten.div_: elem_flop,
    aten.div: elem_flop,
    aten.sqrt: elem_flop,
    aten.zero_: elem_flop,
    aten.copy: elem_flop,
    aten.copy_: elem_flop,
    # aggregation ops
    aten.sum: elem_flop,
    aten.mean: elem_flop,
    aten.var_mean: elem_flop,
    # index ops
    # aten.slice: None,
    # aten.index: None,
    aten.index_copy_: index_flop,
    aten.index_select: index_select_flop,
    # aten.index: index_flop,
    # make new tensor,
    aten.zeros_like: elem_flop,
    aten.zeros: elem_flop,
}
 
def normalize_tuple(x):
    if not isinstance(x, tuple):
        return (x,)
    return x
 
class FlopCounterMode(TorchDispatchMode):
    def __init__(self, module = None, verbose_on_unknown_func=False):
        self.flop_counts = defaultdict(lambda: defaultdict(int))
        self.parents = ['Global']
        if module is not None:
            for name, module in dict(module.named_children()).items():
                module.register_forward_pre_hook(self.enter_module(name))
                module.register_forward_hook(self.exit_module(name))
        self.verbose_on_unknown_func = verbose_on_unknown_func
    
    @property
    def total_flops(self):
        return sum(self.flop_counts['Global'].values())
 
    def enter_module(self, name):
        def f(module, inputs):
            self.parents.append(name)
            inputs = normalize_tuple(inputs)
            out = self.create_backwards_pop(name)(*inputs)
            return out
 
        return f
 
    def exit_module(self, name):
        def f(module, inputs, outputs):
            assert(self.parents[-1] == name)
            self.parents.pop()
            outputs = normalize_tuple(outputs)
            return self.create_backwards_push(name)(*outputs)
        return f
 
    def create_backwards_push(self, name):
        class PushState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args
 
            @staticmethod
            def backward(ctx, *grad_outs):
                self.parents.append(name)
                return grad_outs
 
        return PushState.apply
 
    def create_backwards_pop(self, name):
        class PopState(torch.autograd.Function):
            @staticmethod
            def forward(ctx, *args):
                args = tree_map(lambda x: x.clone() if isinstance(x, torch.Tensor) else x, args)
                if len(args) == 1:
                    return args[0]
                return args
 
            @staticmethod
            def backward(ctx, *grad_outs):
                assert(self.parents[-1] == name)
                self.parents.pop()
                return grad_outs
 
        return PopState.apply
 
 
    def __enter__(self):
        self.flop_counts.clear()
        super().__enter__()
 
    def __exit__(self, *args):
        print(f"Total: {self.total_flops/1e9:.5f} GFLOPS")
        # for mod in self.flop_counts.keys():
        #     print(f" Module: ", mod)
        #     for k,v in self.flop_counts[mod].items():
        #         print(f"  {k}: {v/1e9:.3f} GFLOPS")
        #     print()
        super().__exit__(*args)
 
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs if kwargs else {}
 
        out = func(*args, **kwargs)
        func_packet = func._overloadpacket
        if func_packet in flop_mapping:
            flop_count = flop_mapping[func_packet](args, normalize_tuple(out))
            for par in self.parents:
                self.flop_counts[par][func_packet] += flop_count
        elif self.verbose_on_unknown_func:
            print(f" - Unknown func: {func_packet}")
 
        return out
 

if __name__ == '__main__':
    import torchvision.models as models
    from torch.nn import CrossEntropyLoss

    print(f"Count FLOPs on resnet18. Batch size=8.")
    
    inp = torch.randn(8, 3, 224, 224, device='cuda')
    model = models.resnet18().cuda()
    flop_counter = FlopCounterMode(model)
    with flop_counter:
        # mod(inp).sum().backward()
        loss = CrossEntropyLoss()(model(inp), torch.randint(0, 10, (8,), device='cuda'))
        fwd_flops = flop_counter.total_flops/1e9
        print(f"Forward: {fwd_flops:.3f} GFLOPs")
        loss.backward()
        print(
            f"Backward: {flop_counter.total_flops/1e9 - fwd_flops:.3f} GFLOPs")

# print(flop_counter.flop_counts)
# print(f"Total FLOPs: {np.sum([v for k, v in flop_counter.flop_counts.items()])/1e9:.3f} GFLOPs")

# with flop_counter:
#     mod(inp).sum().backward()

# print(flop_counter.flop_counts)
# exit(0)
 
# from torch.fx.experimental.symbolic_shapes import ShapeEnv
# from torch._subclasses import FakeTensorMode
# shape_env = ShapeEnv()
# fake_mode = FakeTensorMode(shape_env=shape_env)
 
# with fake_mode:
#     inp = fake_mode.from_tensor(inp)
#     assert inp.shape[0] == 1
#     mod = models.resnet18()
#     flop_counter = FlopCounterMode(mod)
#     with flop_counter:
#         with torch.no_grad():
#             mod(inp)
