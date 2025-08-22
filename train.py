import sys
import torch
import importlib
from config import load_config
from trainers.default_trainer import DefaultTrainer
from data_pipelines.char_pipeline import CharDataPipeline

def get_class(module_name, class_name):
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def main(config_file='configs/base_config.yaml'):
    config = load_config(config_file)

    # --- Tokenizer ---
    tokenizer_module = config['tokenizer'].pop('module')
    tokenizer_class = get_class(tokenizer_module, config['tokenizer']['class'])
    tokenizer = tokenizer_class()

    # --- Data Pipeline ---
    data_pipeline_module = config['data'].pop('module')
    data_pipeline_class = get_class(data_pipeline_module, config['data']['class'])
    data_pipeline = data_pipeline_class(config['data']['data_dir'], config['trainer']['batch_size'], config['model']['block_size'])

    # --- Model ---
    model_module = config['model'].pop('module')
    model_config_class = get_class(model_module, config['model'].pop('config_class'))
    model_class_name = config['model'].pop('class')
    model_config = model_config_class(**config['model'])
    model_class = get_class(model_module, model_class_name)
    model = model_class(model_config)
    model.to(config['trainer']['device'])

    # --- Trainer ---
    trainer_module = config['trainer'].pop('module')
    trainer_class = get_class(trainer_module, config['trainer']['class'])
    trainer = trainer_class(config, model, data_pipeline.get_train_loader(), data_pipeline.get_val_loader())

    trainer.train()

if __name__ == '__main__':
    config_file = sys.argv[1] if len(sys.argv) > 1 else 'configs/base_config.yaml'
    main(config_file)
