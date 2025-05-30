"""Configuration file for defining paths to data."""
import os

def make_if_not_exist(p):
    if not os.path.exists(p):
        os.makedirs(p)

hostname = os.uname()[1]  # type: str
# Update your paths here.
CHECKPOINT_ROOT = './checkpoint'
data_root = '/your/data/path'
hub_root = os.path.join(data_root, 'cache/torch/hub')  # for torch.hub
make_if_not_exist(data_root)
make_if_not_exist(CHECKPOINT_ROOT)
make_if_not_exist(hub_root)

DATA_PATHS = {}

DATA_PATHS = {
    "Cifar10": '/your/data/path' + "/cifar10",
    "Cifar100": '/your/data/path' + "/cifar100",
    "CIFAR-10_root": '/your/data/path' + "/cifar10/",
    "IN": '/your/data/path' + "/image-net-all/ILSVRC2012/",
    "IN-C": '/your/data/path' + "/imagenet-c/",
}
MODEL_PATHS = {
    'RobustBench_root': data_root + "/robustbench_models",
}
cifar10_pretrained_fp = f'repos/pytorch-cifar/checkpoint/resnet18_lr0.1.pth'


def set_torch_hub():
    import torch
    torch.hub.set_dir(hub_root)
