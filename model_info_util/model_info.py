from torchinfo import summary
from rich import print
from rich.console import Console
from rich.text import Text
import io

console = Console()

def colorize_summary(summary_str):
    # Split the summary into lines
    lines = summary_str.splitlines()
    colored_lines = []

    for line in lines:
        if "Trainable params" in line or "Non-trainable params" in line:
            colored_lines.append(f"[bold green]{line}[/bold green]")
        elif "Total params" in line:
            colored_lines.append(f"[bold red]{line}[/bold red]")
        else:
            colored_lines.append(f"[yellow]{line}[/yellow]")

    return "\n".join(colored_lines)

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

