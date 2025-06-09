import os
import logging
import numpy as np
from models.model import get_model
from utils import get_accuracy, eval_domain_dict
from datasets.data_loading import get_test_loader
from conf import cfg, load_cfg_from_args, get_num_classes, get_domain_sequence, adaptation_method_lookup
from methods.das import DAS

logger = logging.getLogger(__name__)

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

def evaluate(description):
    load_cfg_from_args(description)
    valid_settings = ["reset_each_shift",           # reset the model state after the adaptation to a domain
                      "continual",                  # train on sequence of domain shifts without knowing when a shift occurs
                      "gradual",                    # sequence of gradually increasing / decreasing domain shifts
                      "correlated",                 # sorted by class label
                      "gradual_correlated",         # gradual domain shifts + sorted by class label
                      "reset_each_shift_correlated"
                      ]
    assert cfg.SETTING in valid_settings, f"The setting '{cfg.SETTING}' is not supported! Choose from: {valid_settings}"

    """Setting the model."""
    num_classes = get_num_classes(dataset_name=cfg.CORRUPTION.DATASET)
    base_model = get_model(cfg, num_classes, device=cfg.DEVICE)

    """Setting the TTA methods and optimizers."""
    model = eval(f'{adaptation_method_lookup(cfg.MODEL.ADAPTATION)}')(cfg=cfg, model=base_model, num_classes=num_classes)
    logger.info(f"Successfully prepared test-time adaptation method: {cfg.MODEL.ADAPTATION.upper()}")

    """Setting the dataset."""
    if cfg.CORRUPTION.DATASET in {"domainnet126"}:
        # extract the domain sequence for a specific checkpoint.
        dom_names_all = get_domain_sequence(ckpt_path=cfg.CKPT_PATH)
    elif cfg.CORRUPTION.DATASET in {"imagenet_d", "imagenet_d109"} and not cfg.CORRUPTION.TYPE[0]:
        # dom_names_all = ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"]
        dom_names_all = ["clipart", "infograph", "painting", "real", "sketch"]
    else:
        dom_names_all = cfg.CORRUPTION.TYPE
    logger.info(f"Using the following domain sequence: {dom_names_all}")

    dom_names_loop = dom_names_all
    if "gradual" in cfg.SETTING and cfg.CORRUPTION.DATASET in {"cifar10_c", "cifar100_c", "imagenet_c"} and len(cfg.CORRUPTION.SEVERITY) == 1:
        severities = [1, 2, 3, 4, 5, 4, 3, 2, 1]
        logger.info(f"Using the following severity sequence for each domain: {severities}")
    else:
        severities = cfg.CORRUPTION.SEVERITY

    """TTA"""
    errs = []
    errs_5 = []
    domain_dict = {}
    cache_mt_all = AverageMeter('Cache', ':6.3f')
    for i_dom, domain_name in enumerate(dom_names_loop):
        if i_dom == 0 or "reset_each_shift" in cfg.SETTING:
            try:
                model.reset()
                logger.info("resetting model")
            except:
                logger.warning("not resetting model")
        else:
            logger.warning("not resetting model")

        for severity in severities:
            test_data_loader = get_test_loader(setting=cfg.SETTING,
                                            adaptation=cfg.MODEL.ADAPTATION,
                                            dataset_name=cfg.CORRUPTION.DATASET,
                                            root_dir=cfg.DATA_DIR,
                                            domain_name=domain_name,
                                            severity=severity,
                                            num_examples=cfg.CORRUPTION.NUM_EX,
                                            rng_seed=cfg.RNG_SEED,
                                            domain_names_all=dom_names_all,
                                            alpha_dirichlet=cfg.TEST.ALPHA_DIRICHLET,
                                            batch_size=cfg.TEST.BATCH_SIZE,
                                            shuffle=False,
                                            workers=min(cfg.TEST.NUM_WORKERS, os.cpu_count()))

            acc, domain_dict, cache_mt = get_accuracy(model,
                                            data_loader=test_data_loader,
                                            dataset_name=cfg.CORRUPTION.DATASET,
                                            domain_name=domain_name,
                                            setting=cfg.SETTING,
                                            domain_dict=domain_dict,
                                            device=cfg.DEVICE)

            cache_mt_all.update(cache_mt.avg)
    
            # Logger information
            err = 1. - acc
            errs.append(err)
            if severity == 5 and domain_name != "none":
                errs_5.append(err)
            logger.info(f"{cfg.CORRUPTION.DATASET} error % [{domain_name}{severity}][#samples={len(test_data_loader.dataset)}]: {err:.2%}, mem_avg: {cache_mt.avg:.2f}MB, mem_avg_all: {cache_mt_all.avg:.2f}MB")

    if len(errs_5) > 0:
        logger.info(f"mean error: {np.mean(errs):.2%}, mean error at 5: {np.mean(errs_5):.2%}")
    else:
        logger.info(f"mean error: {np.mean(errs):.2%}")

if __name__ == '__main__':
    evaluate('"Evaluation.')
