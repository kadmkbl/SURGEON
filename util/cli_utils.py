"""cli_utils from EATA"""
import os
import shutil
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch, print_fh=print):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters if meter.count > 0]
        print_fh(' '.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def save_checkpoint(state, is_best, save_dir=None):
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth.tar')
    torch.save(state, checkpoint_path)
    if is_best:
        best_checkpoint_path = os.path.join(save_dir, 'model_best.pth.tar')
        shutil.copyfile(checkpoint_path, best_checkpoint_path)


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = args.lr * (0.1 ** (epoch // 5))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res
