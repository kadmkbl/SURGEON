import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from torchvision import transforms as trns
from torchvision.datasets import ImageFolder
from .robustbench_data import load_cifar10c, load_cifar100c
from util.config import DATA_PATHS, data_root
from torchvision.datasets import CIFAR10, CIFAR100

IN_C_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
                    'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                    'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']

# NOTE this is more than those in robustbench but included in Hendryc's dataset.
CIFAR10_CORRUPTIONS = ('saturate', 'glass_blur', 'fog', 'brightness', 'snow', 'contrast',
                       'defocus_blur', 'zoom_blur', 'jpeg_compression', 'elastic_transform',
                       'spatter', 'frost', 'gaussian_blur', 'impulse_noise', 'gaussian_noise',
                       'motion_blur', 'speckle_noise', 'pixelate', 'shot_noise')


class LabeledDataset(Dataset):
    def __init__(self, data, targets, transform=None):
        super(LabeledDataset, self).__init__()
        assert data.size(0) == targets.size(0)
        self.data = data
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, y

    def __len__(self):
        return len(self.targets)


# //////// Prepare data loaders //////////
def prepare_imagenet_test_data(corruption, level, batch_size,
                               subset_size=None, workers=1, seed=None,
                               num_classes=None):

    rng = np.random.RandomState(seed) if seed is not None else np.random
    normalize = trns.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    if corruption == 'original':
        te_transforms = trns.Compose([trns.Resize(256), trns.CenterCrop(224), trns.ToTensor(),
                                      normalize])
        print('Test on the original test set')
        val_root = os.path.join(DATA_PATHS['IN'], 'val')
        test_set = ImageFolder(val_root, te_transforms)
    elif corruption in IN_C_corruptions:
        te_transforms_imageC = trns.Compose([trns.CenterCrop(224),
                                             trns.ToTensor(),
                                             normalize])
        print('Test on %s level %d' % (corruption, level))
        val_root = os.path.join(DATA_PATHS['IN-C'], corruption, str(level))
        test_set = ImageFolder(val_root, te_transforms_imageC)
    else:
        raise Exception(f'Corruption {corruption} not found!')

    if num_classes is not None:
        idxs = np.nonzero(np.array(test_set.targets) < num_classes)[0]
        test_set = Subset(test_set, indices=idxs)

    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True)
    return test_set, loader

def prepare_cifar10_test_data(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random

    normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR10(DATA_PATHS['Cifar10'], train=False, transform=trans, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar10c(10_000, level, DATA_PATHS['Cifar10'], False, [corruption])
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        test_set = LabeledDataset(x_test, y_test, transform=normalize)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")
    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True)
    return test_set, loader


def prepare_cifar100_test_data(corruption, level, batch_size,
                              subset_size=None, workers=1, seed=None):
    rng = np.random.RandomState(seed) if seed is not None else np.random

    normalize = trns.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    trans = trns.Compose([trns.ToTensor(), normalize])
    if corruption == 'original':
        test_set = CIFAR100(DATA_PATHS['Cifar100'], train=False, transform=trans, download=True)
    elif corruption in CIFAR10_CORRUPTIONS:
        x_test, y_test = load_cifar100c(10_000, level, DATA_PATHS['Cifar100'], False, [corruption])
        # x_test, y_test = x_test.to(device), y_test.to(device)  # NOTE this will cause CUDA init error
        test_set = LabeledDataset(x_test, y_test, transform=normalize)
    else:
        raise RuntimeError(f"Not supported corruption: {corruption}")
    if subset_size is not None:
        idxs = np.arange(len(test_set))
        idxs = rng.permutation(idxs)
        idxs = idxs[:subset_size]
        test_set = Subset(test_set, idxs)

    loader = DataLoader(test_set, batch_size=batch_size, shuffle=True,
                        num_workers=workers, pin_memory=True)
    return test_set, loader
