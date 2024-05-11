import torch
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
import datetime
import numpy as np
from scipy.stats import kurtosis

timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

# Argument parser
parser = argparse.ArgumentParser(description='Plot Transformer Weights')
parser.add_argument('--weight', type=str, choices=[
    'normalization_variant.gain', 'transformer.wte.weight', 'transformer.wpe.weight', 
    'transformer.h.0.ln_1.gain', 'transformer.h.0.ln_2.gain', 
    'transformer.h.0.attn.c_attn_q.weight', 'transformer.h.0.attn.c_attn_k.weight', 'transformer.h.0.attn.c_attn_v.weight', 
    'transformer.h.0.attn.c_proj.weight', 'transformer.h.0.attn.rotary_emb_q.angles', 'transformer.h.0.attn.rotary_emb_k.angles', 
    'transformer.h.0.mlp.c_fc.weight', 'transformer.h.0.mlp.c_proj.weight', 
    'transformer.h.1.ln_1.gain', 'transformer.h.1.ln_2.gain', 
    'transformer.h.1.attn.c_attn_q.weight', 'transformer.h.1.attn.c_attn_k.weight', 'transformer.h.1.attn.c_attn_v.weight', 
    'transformer.h.1.attn.c_proj.weight', 'transformer.h.1.attn.rotary_emb_q.angles', 'transformer.h.1.attn.rotary_emb_k.angles', 
    'transformer.h.1.mlp.c_fc.weight', 'transformer.h.1.mlp.c_proj.weight', 
    'transformer.h.2.ln_1.gain', 'transformer.h.1.ln_2.gain', 
    'transformer.h.2.attn.c_attn_q.weight', 'transformer.h.2.attn.c_attn_k.weight', 'transformer.h.2.attn.c_attn_v.weight', 
    'transformer.h.2.attn.c_proj.weight', 'transformer.h.2.attn.rotary_emb_q.angles', 'transformer.h.2.attn.rotary_emb_k.angles', 
    'transformer.h.2.mlp.c_fc.weight', 'transformer.h.2.mlp.c_proj.weight', 
    'transformer.h.3.ln_1.gain', 'transformer.h.3.ln_2.gain', 
    'transformer.h.3.attn.c_attn_q.weight', 'transformer.h.3.attn.c_attn_k.weight', 'transformer.h.3.attn.c_attn_v.weight', 
    'transformer.h.3.attn.c_proj.weight', 'transformer.h.3.attn.rotary_emb_q.angles', 'transformer.h.3.attn.rotary_emb_k.angles', 
    'transformer.h.3.mlp.c_fc.weight', 'transformer.h.3.mlp.c_proj.weight', 
    'transformer.ln_f.gain', 'lm_head.weight'], 
    default='transformer.h.0.attn.c_attn_q.weight',
    help='Choose which weight to display')
parser.add_argument('--graph', type=str, choices=['histogram', 'matrix'], default='matrix',
                    help='Choose which graph to use: histogram or matrix')
args = parser.parse_args()

# Load the weights
checkpoint = torch.load('out/ckpt.pt')
weights = checkpoint['model']

# Select weights
weight_matrix = weights[args.weight]

# Create a directory for images if it doesn't exist
image_dir = 'images'
os.makedirs(image_dir, exist_ok=True)

# Plotting
if weight_matrix.is_cuda:
    weight_matrix = weight_matrix.cpu()
if args.graph == 'matrix':
    plt.figure(figsize=(10, 8))
    sns.heatmap(weight_matrix, cmap='viridis', annot=True)
    plt.title(f'{args.weight} Matrix')
    plt.xlabel('Columns')
    plt.ylabel('Rows')
    image_path = os.path.join(image_dir, f'{args.weight}_matrix_{timestamp}.png')
elif args.graph == 'histogram':
    flat_weights = weight_matrix.flatten().numpy()
    mean = np.mean(flat_weights)
    std_dev = np.std(flat_weights)
    kurt = kurtosis(flat_weights, fisher=True)
    plt.hist(flat_weights, bins=50)
    plt.title(f'{args.weight} Histogram\nMean: {mean:.4f}, Std Dev: {std_dev:.4f}, Kurtosis: {kurt:.4f}')
    plt.xlabel('Weight Value')
    plt.ylabel('Frequency')
    image_path = os.path.join(image_dir, f'{args.weight}_histogram_{timestamp}.png')

# Save the image
plt.savefig(image_path)
print(f'Saved image to {image_path}')
