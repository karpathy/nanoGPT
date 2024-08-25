import os
import csv
from contextlib import nullcontext
import numpy as np
import time
import torch
import plotly.graph_objects as go
from model import GPT
from rich.console import Console
from rich.table import Table

# Load configuration
from gpt_conf import GPTConfig

# -----------------------------------------------------------------------------
batch_size = 1
bias = False
real_data = True
seed = 1337
device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1', etc.
dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32' or 'bfloat16' or 'float16'
use_gradient_checkpointing = True
profile = True

# List of softmax variants to test
softmax_variants = ["relumax", "softmax", "softplus"]

# Set of block sizes to sweep (including one warmup that we will drop later)
block_sizes = [16, 32, 64, 128, 256, 512, 768, 1024, 1548, 2048, 3072, 4096]

# Number of runs for averaging
num_runs = 5

# To store timing results
ln1_ln2_timing_results = {variant: {} for variant in softmax_variants}
forward_pass_timing_results = {variant: {} for variant in softmax_variants}

# Console for rich output
console = Console()

for variant in softmax_variants:
    for block_size in block_sizes:
        console.print(f"[bold yellow]Testing variant: {variant} with block size: {block_size}[/bold yellow]")

        ln1_ln2_times = []
        forward_pass_times = []

        for run in range(num_runs):
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
            torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
            device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
            ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
            ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

            # data loading init
            if real_data:
                dataset = 'shakespeare'
                data_dir = os.path.join('data', dataset)
                train_data = np.memmap(os.path.join(data_dir, 'train.bin'), dtype=np.uint16, mode='r')
                def get_batch(split):
                    data = train_data # note ignore split in benchmarking script
                    ix = torch.randint(len(data) - block_size, (batch_size,))
                    x = torch.stack([torch.from_numpy((data[i:i+block_size]).astype(np.int64)) for i in ix])
                    y = torch.stack([torch.from_numpy((data[i+1:i+1+block_size]).astype(np.int64)) for i in ix])
                    x, y = x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
                    return x, y
            else:
                # alternatively, if fixed data is desired to not care about data loading
                x = torch.randint(50304, (batch_size, block_size), device=device)
                y = torch.randint(50304, (batch_size, block_size), device=device)
                get_batch = lambda split: (x, y)

            # model init
            gptconf = GPTConfig(
                block_size = block_size, # how far back does the model look? i.e. context size
                n_layer = 6, n_head = 12, n_embd = 768, # size of the model
                softmax_variant_attn = variant,
                strongermax_use_xmax = True,
                strongermax_sum_to_1 = True,
                dropout = 0, # for determinism
                bias = bias,
            )
            model = GPT(gptconf)
            model.to(device)

            optimizer = model.configure_optimizers(weight_decay=1e-2, learning_rate=1e-4, betas=(0.9, 0.95), device_type=device_type)

            # Profiling section
            if profile:
                for _ in range(num_runs):
                    X, Y = get_batch('train')

                    # Timing from the start of the forward pass to the end (logits retrieval)
                    forward_start = torch.cuda.Event(enable_timing=True)
                    forward_end = torch.cuda.Event(enable_timing=True)

                    # Timing between ln_1 and ln_2 within a block
                    ln1_start = torch.cuda.Event(enable_timing=True)
                    ln2_end = torch.cuda.Event(enable_timing=True)

                    forward_start.record()
                    with ctx:
                        tok_emb = model.transformer.wte(X)  # Starting point
                        x = model.transformer.drop(tok_emb)

                        # Assuming we're focusing on the first block for ln_1 to ln_2 timing
                        block = model.transformer.h[0]

                        ln1_start.record()
                        x_ln1 = block.ln_1(x)  # Timing starts after ln_1
                        x_attn = block.attn(x_ln1)  # Attention
                        x = x + x_attn  # Residual connection
                        x_ln2 = block.ln_2(x)  # Timing ends after ln_2
                        ln2_end.record()

                        # Continue the forward pass
                        for b in model.transformer.h[1:]:
                            x = b(x)
                        logits = model.lm_head(x)  # Final logits
                    forward_end.record()

                    torch.cuda.synchronize()

                    ln1_ln2_time = ln1_start.elapsed_time(ln2_end)
                    forward_pass_time = forward_start.elapsed_time(forward_end)

                    ln1_ln2_times.append(ln1_ln2_time)
                    forward_pass_times.append(forward_pass_time)

                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), Y.view(-1), ignore_index=-1)
                    loss.backward()
                    optimizer.step()

        # Calculate average times
        avg_ln1_ln2_time = sum(ln1_ln2_times) / len(ln1_ln2_times)
        avg_forward_pass_time = sum(forward_pass_times) / len(forward_pass_times)

        # Store the results
        ln1_ln2_timing_results[variant][block_size] = avg_ln1_ln2_time
        forward_pass_timing_results[variant][block_size] = avg_forward_pass_time

# Remove the first block size from the results (warmup)
block_sizes = block_sizes[1:]

# Create Rich tables to display the results
ln1_ln2_table = Table(title="Avg LN1 to LN2 Time (ms)", caption="Variants on Rows, Block Sizes on Columns")
forward_pass_table = Table(title="Avg Forward Pass Time (ms)", caption="Variants on Rows, Block Sizes on Columns")

# Define columns for both tables
ln1_ln2_table.add_column("Softmax Variant", justify="center", style="cyan", no_wrap=True)
forward_pass_table.add_column("Softmax Variant", justify="center", style="cyan", no_wrap=True)

for block_size in block_sizes:
    ln1_ln2_table.add_column(f"Block Size {block_size}", justify="center", style="green")
    forward_pass_table.add_column(f"Block Size {block_size}", justify="center", style="green")

# Populate the tables with timing results
ln1_ln2_csv_data = []
forward_pass_csv_data = []

for variant in softmax_variants:
    ln1_ln2_row = [variant]
    forward_pass_row = [variant]
    for block_size in block_sizes:
        ln1_ln2_row.append(f"{ln1_ln2_timing_results[variant][block_size]:.4f}")
        forward_pass_row.append(f"{forward_pass_timing_results[variant][block_size]:.4f}")
    
    ln1_ln2_table.add_row(*ln1_ln2_row)
    forward_pass_table.add_row(*forward_pass_row)

    # Prepare CSV data
    ln1_ln2_csv_data.append(ln1_ln2_row)
    forward_pass_csv_data.append(forward_pass_row)

# Display the tables
console.print(ln1_ln2_table)
console.print(forward_pass_table)

# Save CSV
ln1_ln2_csv_file = 'ln1_ln2_timing_results.csv'
forward_pass_csv_file = 'forward_pass_timing_results.csv'

with open(ln1_ln2_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Softmax Variant'] + [f"Block Size {block_size}" for block_size in block_sizes])
    writer.writerows(ln1_ln2_csv_data)

with open(forward_pass_csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['Softmax Variant'] + [f"Block Size {block_size}" for block_size in block_sizes])
    writer.writerows(forward_pass_csv_data)

# Create Plotly plots
# Plot for Avg LN1 to LN2 Time
ln1_ln2_fig = go.Figure()
for variant in softmax_variants:
    ln1_ln2_fig.add_trace(go.Scatter(
        x=block_sizes,
        y=[ln1_ln2_timing_results[variant][block_size] for block_size in block_sizes],
        mode='lines+markers',
        name=variant
    ))

ln1_ln2_fig.update_layout(
    title="Avg LN1 to LN2 Time Across Block Sizes and Variants",
    xaxis_title="Block Size",
    yaxis_title="Avg LN1 to LN2 Time (ms)",
    legend_title="Softmax Variant",
)

# Save the figure as a PNG file
ln1_ln2_fig.write_image("ln1_ln2_timing_plot.png")

# Plot for Avg Forward Pass Time
forward_pass_fig = go.Figure()
for variant in softmax_variants:
    forward_pass_fig.add_trace(go.Scatter(
        x=block_sizes,
        y=[forward_pass_timing_results[variant][block_size] for block_size in block_sizes],
        mode='lines+markers',
        name=variant
    ))

forward_pass_fig.update_layout(
    title="Avg Forward Pass Time Across Block Sizes and Variants",
    xaxis_title="Block Size",
    yaxis_title="Avg Forward Pass Time (ms)",
    legend_title="Softmax Variant",
)

# Save the figure as a PNG file
forward_pass_fig.write_image("forward_pass_timing_plot.png")

