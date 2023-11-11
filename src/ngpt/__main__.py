# -*- coding: utf-8 -*-
"""
llm/__main__.py

Contains main entry point for training.
"""
from __future__ import (
    absolute_import,
    annotations,
    division,
    print_function,
    unicode_literals
)
import os
# import sys

import hydra
from pathlib import Path
from omegaconf import OmegaConf
from enrich import get_logger
# from ezpz import get_logger
# from ngpt import get_logger

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.dist import setup, setup_wandb
from ngpt.configs import ExperimentConfig, PROJECT_ROOT
from ngpt.trainer import Trainer

log = get_logger(__name__, "DEBUG")


def include_file(f) -> bool:
    fp = Path(f)
    exclude_ = (
        'venv/' not in fp.as_posix()
        and 'old/' not in fp.as_posix()
        and 'outputs/' not in fp.as_posix()
        and 'wandb/' not in fp.as_posix()
        and 'data/' not in fp.as_posix()
        and 'cache/' not in fp.as_posix()
        and fp.suffix not in ['.pt', '.pth']
    )
    include_ = fp.suffix in ['.py', '.log', '.yaml']
    # return (
    #     exclude_ and include_
    #     # 'venv' not in fp.as_posix()
    #     # and fp.suffix in ['.py', '.log', '.yaml']
    # )
    return (exclude_ and include_)


def build_trainer(cfg: DictConfig) -> Trainer:
    rank = setup(
        framework=cfg.train.framework,
        backend=cfg.train.backend,
        seed=cfg.train.seed,
        # framework=config.train.framework,
        # backend=config.train.backend,
        # seed=config.train.seed
    )
    config: ExperimentConfig = instantiate(cfg)
    if rank != 0:
        log.setLevel("CRITICAL")
    # if rank == 0:
    else:
        log.setLevel("DEBUG")
        from rich import print_json
        if config.train.use_wandb:
            setup_wandb(
                project_name=config.train.wandb_project,
                config=cfg,
            )
            if wandb.run is not None:
                wandb.run.config['tokens_per_iter'] = config.tokens_per_iter
                wandb.run.config['samples_per_iter'] = config.samples_per_iter
        log.critical(OmegaConf.to_yaml(cfg))
        print_json(config.to_json())
    log.warning(f'Output dir: {os.getcwd()}')
    return Trainer(config)


def train(cfg: DictConfig) -> Trainer:
    trainer = build_trainer(cfg)
    trainer.train()
    if wandb.run is not None:
        wandb.run.log_code(PROJECT_ROOT, include_fn=include_file)
        # raw_module = trainer.model.module
        # assert isinstance(raw_module, torch.nn.Module)
        trainer.save_ckpt(add_to_wandb=True)
    # if rank == 0 and config.backend.lower() in ['ds', 'dspeed', 'deepspeed']:
    #     git_ds_info()
    return trainer



@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> Trainer:
    return train(cfg)


if __name__ == '__main__':
    import wandb
    rank = main()
    if wandb.run is not None:
        wandb.finish(0)
    # sys.exit(0)
