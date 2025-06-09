import gc
import datetime
import inspect
import sys

import torch
import numpy as np
from util.config import make_if_not_exist

sys.stdout.close = lambda: None

dtype_memory_size_dict = {
    torch.float64: 64/8,
    torch.double: 64/8,
    torch.float32: 32/8,
    torch.float: 32/8,
    torch.float16: 16/8,
    torch.half: 16/8,
    torch.int64: 64/8,
    torch.long: 64/8,
    torch.int32: 32/8,
    torch.int: 32/8,
    torch.int16: 16/8,
    torch.short: 16/8,
    torch.uint8: 8/8,
    torch.int8: 8/8,
}
# compatibility of torch1.0
if getattr(torch, "bfloat16", None) is not None:
    dtype_memory_size_dict[torch.bfloat16] = 16/8
if getattr(torch, "bool", None) is not None:
    dtype_memory_size_dict[torch.bool] = 8/8 # pytorch use 1 byte for a bool, see https://github.com/pytorch/pytorch/issues/41571

def get_mem_space(x):
    try:
        ret = dtype_memory_size_dict[x]
    except KeyError:
        print(f"dtype {x} is not supported!")
    return ret

class MemTracker(object):
    """
    Class used to track pytorch memory usage
    Arguments:
        detail(bool, default True): whether the function shows the detail gpu memory usage
        path(str): where to save log file
        verbose(bool, default False): whether show the trivial exception
        device(int): GPU number, default is 0
    """
    global_tracker = None

    @classmethod
    def init_tracker(cls, **kwargs):
        cls.global_tracker = MemTracker(**kwargs)

    @classmethod
    def track(cls, line_info='', **kwargs):
        if cls.global_tracker is not None:
            cls.global_tracker.do_track(line_info=line_info, **kwargs)

    def __init__(self, detail=True, path='mem_track/', verbose=False, device=0, filename=None):
        make_if_not_exist(path)
        self.print_detail = detail
        self.last_tensor_sizes = set()
        self.gpu_profile_fn = path
        if filename is None:
            self.gpu_profile_fn += f'{datetime.datetime.now():%d-%b-%y-%H:%M:%S}-gpu_mem_track.txt'
        else:
            self.gpu_profile_fn += filename
        self.verbose = verbose
        self.begin = True
        self.device = device
        self.last_cache_size = 0.

    def get_tensors(self):
        for obj in gc.get_objects():
            try:
                if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                    tensor = obj
                else:
                    continue
                if tensor.is_cuda:
                    yield tensor
            except Exception as e:
                if self.verbose:
                    print('A trivial exception occured: {}'.format(e))

    def get_tensor_usage(self):
        sizes = [np.prod(np.array(tensor.size())) * get_mem_space(tensor.dtype) for tensor in self.get_tensors()]
        return np.sum(sizes) / 1024**2

    def get_allocate_usage(self):
        return torch.cuda.memory_allocated() / 1024**2

    def get_cache_usage(self):
        return torch.cuda.memory_cached() / 1024**2

    def clear_cache(self):
        gc.collect()
        torch.cuda.empty_cache()

    def print_all_gpu_tensor(self, file=None):
        for x in self.get_tensors():
            print(x.size(), x.dtype, np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2, file=file)

    def do_track(self, line_info='', show_full_tensor=False):
        """
        Track the GPU memory usage
        """
        frameinfo = inspect.stack()[1]
        where_str = frameinfo.filename + ' line ' + str(frameinfo.lineno) + ': ' + frameinfo.function

        # with open(self.gpu_profile_fn, 'a+') as f:
        with sys.stdout as f:

            if self.begin:
                f.write(f"GPU Memory Track | {datetime.datetime.now():%d-%b-%y-%H:%M:%S} |"
                        f" Total Tensor Used Memory:{self.get_tensor_usage():<7.1f}Mb"
                        f" Total Allocated Memory:{self.get_allocate_usage():<7.1f}Mb\n\n")
                self.begin = False

            f.write("="*40 + f"\nTask: {line_info}\n")

            if self.print_detail is True:
                ts_list = [(tensor.size(), tensor.dtype) for tensor in self.get_tensors()]
                new_tensor_sizes = {(type(x),
                                    tuple(x.size()),
                                    ts_list.count((x.size(), x.dtype)),
                                    np.prod(np.array(x.size()))*get_mem_space(x.dtype)/1024**2,
                                    x.dtype) for x in self.get_tensors()}
                for t, s, n, m, data_type in new_tensor_sizes - self.last_tensor_sizes:
                    f.write(f'+ | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')
                for t, s, n, m, data_type in self.last_tensor_sizes - new_tensor_sizes:
                    f.write(f'- | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')
                if show_full_tensor:
                    for t, s, n, m, data_type in self.last_tensor_sizes.intersection(new_tensor_sizes):
                        f.write(f'o | {str(n)} * Size:{str(s):<20} | Memory: {str(m*n)[:6]} M | {str(t):<20} | {data_type}\n')

                self.last_tensor_sizes = new_tensor_sizes
            new_cache_size = self.get_cache_usage()

            f.write(f"\n@ {where_str:<50}\n"
                    f" Total Tensor Used Memory: {self.get_tensor_usage():<7.1f}Mb\n"
                    f" Total Allocated Memory  : {self.get_allocate_usage():<7.1f}Mb\n"
                    f" Total Cached Memory     : {new_cache_size:<7.1f}Mb\n"
                    f" Cache increase          : {new_cache_size-self.last_cache_size:<7.1f}Mb\n\n"
                    )
            self.last_cache_size = new_cache_size