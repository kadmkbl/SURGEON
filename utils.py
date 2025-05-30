import torch
import logging
import numpy as np
from datasets.imagenet_subsets import IMAGENET_D_MAPPING
import torch.nn as nn

logger = logging.getLogger(__name__)

import matplotlib.pyplot as plt

from fvcore.nn import FlopCountAnalysis

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = None
        self.sum = 0
        self.count = 0
        self.values = []
        self.update_cnt = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.values.append(val)
        self.update_cnt += 1

    @property
    def avg(self):
        return self.sum / self.count if self.count > 0 else None

    @property
    def max(self):
        return np.max(self.values) if self.count > 0 else None

    @property
    def step_avg(self):
        return np.mean(self.values)

    @property
    def step_std(self):
        return np.std(self.values)

    def __str__(self):
        if self.count > 0:
            fmtstr = '{name} {val' + self.fmt + '} (avg={avg' + self.fmt + '})'
            return fmtstr.format(name=self.name, val=self.val, avg=self.avg)
        else:
            return f'{self.name}: N/A'

def split_results_by_domain(domain_dict, data, predictions):
    """
    Separate the labels and predictions by domain
    :param domain_dict: dictionary, where the keys are the domain names and the values are lists with pairs [[label1, prediction1], ...]
    :param data: list containing [images, labels, domains, ...]
    :param predictions: tensor containing the predictions of the model
    :return: updated result dict
    """

    labels, domains = data[1], data[2]
    assert predictions.shape[0] == labels.shape[0], "The batch size of predictions and labels does not match!"

    for i in range(labels.shape[0]):
        if domains[i] in domain_dict.keys():
            domain_dict[domains[i]].append([labels[i].item(), predictions[i].item()])
        else:
            domain_dict[domains[i]] = [[labels[i].item(), predictions[i].item()]]

    return domain_dict

def eval_domain_dict(domain_dict, domain_seq=None):
    """
    Print detailed results for each domain. This is useful for settings where the domains are mixed
    :param domain_dict: dictionary containing the labels and predictions for each domain
    :param domain_seq: if specified and the domains are contained in the domain dict, the results will be printed in this order
    """
    correct = []
    num_samples = []
    avg_error_domains = []
    dom_names = domain_seq if all([dname in domain_seq for dname in domain_dict.keys()]) else domain_dict.keys()
    logger.info(f"Splitting up the results by domain...")
    for key in dom_names:
        content = np.array(domain_dict[key])
        correct.append((content[:, 0] == content[:, 1]).sum())
        num_samples.append(content.shape[0])
        accuracy = correct[-1] / num_samples[-1]
        error = 1 - accuracy
        avg_error_domains.append(error)
        logger.info(f"{key:<20} error: {error:.2%}")
    logger.info(f"Average error across all domains: {sum(avg_error_domains) / len(avg_error_domains):.2%}")
    # The error across all samples differs if each domain contains different amounts of samples
    logger.info(f"Error over all samples: {1 - sum(correct) / sum(num_samples):.2%}")

def get_accuracy(model: torch.nn.Module,
                 data_loader: torch.utils.data.DataLoader,
                 dataset_name: str,
                 domain_name: str,
                 setting: str,
                 domain_dict: dict,
                 device: torch.device = None):

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    correct = 0.
    cache_mt = AverageMeter('Cache', ':6.3f')
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            imgs, labels = data[0], data[1]
            output = model([img.to(device) for img in imgs]) if isinstance(imgs, list) else model(imgs.to(device))
            predictions = output.argmax(1)

            if dataset_name == "imagenet_d" and domain_name != "none":
                mapping_vector = list(IMAGENET_D_MAPPING.values())
                predictions = torch.tensor([mapping_vector[pred] for pred in predictions], device=device)

            correct += (predictions == labels.to(device)).float().sum()
            mem = get_mem(model)
            cache_mt.update(mem)

            if "mixed_domains" in setting and len(data) >= 3:
                domain_dict = split_results_by_domain(domain_dict, data, predictions)

    accuracy = correct.item() / len(data_loader.dataset)
    return accuracy, domain_dict, cache_mt

def get_mem(model: torch.nn.Module):
    """Get cache memory costs of each layer."""
    BN_cache_size = 0
    Conv_cache_size = 0
    FC_cache_size = 0
    for mod_name, target_mod in model.named_modules():
        # Cache size of BN layers
        if isinstance(target_mod, nn.BatchNorm2d):
            BN_cache_size = BN_cache_size + target_mod.back_cache_size
        # Cache size of Conv layers
        elif isinstance(target_mod, nn.Conv2d):
            Conv_cache_size = Conv_cache_size + target_mod.back_cache_size
        # Cache size of FC layers
        elif isinstance(target_mod, nn.Linear):
            FC_cache_size = FC_cache_size + target_mod.back_cache_size

    # return (BN_cache_size + Conv_cache_size + FC_cache_size) * 4 / (2 ** 20) # Total backward cache size
    return (BN_cache_size + Conv_cache_size + FC_cache_size) * 4 / (2 ** 20) * 2 # Total backward cache size
    # Note!: If the loss includes Consistency Regularization (CR), the total backward cache size doubles
