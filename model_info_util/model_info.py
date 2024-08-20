import torch
from torchinfo import summary
from rich import print
from rich.console import Console
from rich.text import Text
import io

console = Console()

def print_summary(model):
    block_header = Text(f"High Level Parameters:", style="bold underline purple")
    console.print(block_header)
    summary(model)

def print_model_blocks(model, block_range=1):
    for idx, block in enumerate(model.transformer.h):
        block_header = Text(f"Summary for Block {idx + 1}:", style="bold underline green")
        console.print(block_header)
        summary(block)
        if (idx + 1) == block_range:
            break

def print_module_structure(module):
    console.print("-" * 50, style="dim")
    for name, submodule in module.named_children():
        console.print(f'{name}: {submodule}', style="yellow")
    console.print("-" * 50, style="dim")

def print_model_tree(model, indent="", print_params=False):
    for name, module in model.named_children():
        print(indent + name + ": " + str(module.__class__.__name__))
        if isinstance(module, torch.nn.Module):
            # Print parameters for the next level only
            if print_params:
                for param_name, _ in module.named_parameters():
                    print(indent + "  " + param_name)
            else:  # Recursively print submodules without parameters
                print_model_tree(module, indent + "  ")
