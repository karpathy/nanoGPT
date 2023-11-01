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
import sys

import hydra
from pathlib import Path
from omegaconf import OmegaConf
from enrich import get_logger

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.dist import setup, setup_wandb
from ngpt.configs import ExperimentConfig, PROJECT_ROOT
from ngpt.trainer import Trainer

log = get_logger(__name__, level="INFO")


def include_file(f):
    fp = Path(f)
    return (
        'venv' not in fp.as_posix()
        and fp.suffix in ['.py', '.log', '.yaml']
    )


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> int:
    config: ExperimentConfig = instantiate(cfg)
    rank = setup(
        framework=config.train.framework,
        backend=config.train.backend,
        seed=config.train.seed
    )
    if rank != 0:
        log.setLevel("CRITICAL")
    else:
        from rich import print_json
        if config.train.use_wandb:
            setup_wandb(
                project_name=config.train.wandb_project,
                config=cfg,
            )
        if wandb.run is not None:
            wandb.run.log_code(PROJECT_ROOT, include_fn=include_file)
            wandb.run.config['tokens_per_iter'] = config.tokens_per_iter
            wandb.run.config['samples_per_iter'] = config.samples_per_iter
        log.info(OmegaConf.to_yaml(cfg))
        print_json(config.to_json())
    log.info(f'Output dir: {os.getcwd()}')
    trainer = Trainer(config)
    trainer.train()
    # if rank == 0 and config.backend.lower() in ['ds', 'dspeed', 'deepspeed']:
    #     git_ds_info()
    return rank


if __name__ == '__main__':
    import wandb
    rank = main()
    if wandb.run is not None:
        wandb.finish(0)
    sys.exit(0)
