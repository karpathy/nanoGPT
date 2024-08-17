from torchinfo import summary
from rich import print

def print_summary(model):
    summary(model)

def print_module_structure(module, prefix=''):
    for name, submodule in module.named_children():
        print(f'{prefix}{name}: {submodule}')
        print_module_structure(submodule, prefix + '    ')

    # Printing parameters if any
    for name, param in module.named_parameters(recurse=False):
        print(f'{prefix}{name}: {param.size()}')

def print_model_blocks(model, block_range=1):
    for idx, block in enumerate(model.transformer.h):
        print(f"Summary for Block {idx + 1}:")
        summary(block)
        if (idx + 1) == block_range:
            break
