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
from enrich import get_logger
# import logging

from hydra.utils import instantiate
from omegaconf.dictconfig import DictConfig

from ezpz.dist import setup, setup_wandb
from configs import TrainConfig
# from ezpz.configs import TrainConfig, git_ds_info
from trainer import Trainer

# log = logging.getLogger(__name__)
log = get_logger(__name__, level="INFO")


@hydra.main(version_base=None, config_path='./conf', config_name='config')
def main(cfg: DictConfig) -> int:
    config = instantiate(cfg)
    assert isinstance(config, TrainConfig)
    rank = setup(
        framework=config.framework,
        backend=config.backend,
        seed=config.seed
    )
    if rank != 0:
        log.setLevel("CRITICAL")
    else:
        from rich import print_json
        print_json(config.to_json())
        if config.use_wandb:
            setup_wandb(
                project_name=config.wandb_project,
                config=cfg,
            )
    log.info(f'Output dir: {os.getcwd()}')
    trainer = Trainer(config)
    trainer.train()
    # if rank == 0 and config.backend.lower() in ['ds', 'dspeed', 'deepspeed']:
    #     git_ds_info()
    return rank


if __name__ == '__main__':
    # import wandb
    # wandb.require(experiment='service')
    rank = main()
    # if wandb.run is not None:
    #     wandb.finish(0)
    sys.exit(0)
