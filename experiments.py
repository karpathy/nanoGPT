# Verify GPU
import torch
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'PyTorch: {torch.__version__}')
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

import subprocess

runs = [
    # ('adamw',       'out-adamw'),
    # ('sgd',         'out-sgd'),
    # ('muon',        'out-muon'),
    #('lowrankmuon', 'out-lowrankmuon'),
    ('infrequentmuon', 'out-infrequentmuon'),
]

for optimizer_name, out_dir in runs:
    print(f"\n{'='*50}\nRunning {optimizer_name}\n{'='*50}")
    subprocess.run([
        'python', 'train.py', 'config/train_shakespeare_char.py',
        f'--optimizer_name={optimizer_name}',
        f'--out_dir={out_dir}',
        '--device=cuda',
        '--compile=False',
        '--max_iters=2000',
        '--lr_decay_iters=2000', 
        '--eval_interval=200'
    ], check=True)

# Load CSVs
optimizers = {'adamw': 'out-adamw', 'sgd': 'out-sgd', 'muon': 'out-muon', 'lowrankmuon': 'out-lowrankmuon', 'infrequentmuon': 'out-infrequentmuon'}
colors = {'adamw': '#1f77b4', 'sgd': '#ff7f0e', 'muon': '#2ca02c', 'lowrankmuon': '#d62728', 'infrequentmuon': '#9467bd'}
labels = {'adamw': 'AdamW', 'sgd': 'SGD', 'muon': 'Muon', 'lowrankmuon': 'LowRankMuon', 'infrequentmuon': 'InfrequentMuon'}

dfs = {}
for name, d in optimizers.items():
    path = os.path.join(d, 'log.csv')
    if os.path.exists(path):
        dfs[name] = pd.read_csv(path)
        print(f'Loaded {name}: {len(dfs[name])} rows')
    else:
        print(f'WARNING: {path} not found, skipping {name}')

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Plot 1: Val loss vs iteration
ax = axes[0]
for name, df in dfs.items():
    val = df[df['val_loss'].notna()]
    ax.plot(val['iter'], val['val_loss'].astype(float),
            marker='o', markersize=3, label=labels[name], color=colors[name])
ax.set_xlabel('Iteration')
ax.set_ylabel('Validation Loss')
ax.set_title('Val Loss vs Iteration')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 2: Train loss vs iteration
ax = axes[1]
for name, df in dfs.items():
    train = df[df['step_time_ms'].notna()]
    ax.plot(train['iter'], train['train_loss'].astype(float),
            alpha=0.7, label=labels[name], color=colors[name], linewidth=0.8)
ax.set_xlabel('Iteration')
ax.set_ylabel('Training Loss')
ax.set_title('Train Loss vs Iteration')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Val loss vs wall-clock time
ax = axes[2]
for name, df in dfs.items():
    train_rows = df[df['step_time_ms'].notna()].copy()
    train_rows['step_time_ms'] = train_rows['step_time_ms'].astype(float)
    train_rows['cumtime_s'] = train_rows['step_time_ms'].cumsum() / 1000.0
    time_at_iter = dict(zip(train_rows['iter'], train_rows['cumtime_s']))
    val = df[df['val_loss'].notna()].copy()
    val['wall_s'] = val['iter'].map(time_at_iter)
    val = val.dropna(subset=['wall_s'])
    if len(val) > 0:
        ax.plot(val['wall_s'], val['val_loss'].astype(float),
                marker='o', markersize=3, label=labels[name], color=colors[name])
ax.set_xlabel('Wall-clock Time (s)')
ax.set_ylabel('Validation Loss')
ax.set_title('Val Loss vs Time (Efficiency)')
ax.legend()
ax.grid(True, alpha=0.3)

fig.suptitle('Optimizer Comparison - Shakespeare char-level (2000 iters, GPU)', fontsize=14)
plt.tight_layout()
plt.savefig('results_comparison.png', dpi=150)
#plt.show()
# Summary table
print(f"{'Optimizer':<10} {'Final Val Loss':<16} {'Best Val Loss':<16} {'Avg Step (ms)':<16}")
print('-' * 58)
for name, df in dfs.items():
    val = df[df['val_loss'].notna()]
    train = df[df['step_time_ms'].notna()]
    final_val = float(val['val_loss'].iloc[-1]) if len(val) > 0 else float('nan')
    best_val = float(val['val_loss'].astype(float).min()) if len(val) > 0 else float('nan')
    avg_step = float(train['step_time_ms'].astype(float).mean()) if len(train) > 0 else float('nan')
    print(f'{labels[name]:<10} {final_val:<16.4f} {best_val:<16.4f} {avg_step:<16.2f}')
