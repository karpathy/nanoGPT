#!/usr/bin/env python3

import pandas as pd
import numpy as np

# Load and clean data
df = pd.read_csv('out-gradient-rank-analysis/gradient_rank_summary.csv')
df['layer_type'] = df['layer_type'].replace('unknown', 'embedding')

# Calculate compression ratio and percentage
tau = 0.1
df['compression_ratio'] = df[f'threshold_rank_tau_{tau}'] / df['max_rank']
df['compression_percentage'] = df['compression_ratio'] * 100

print("Checking subplot 3 calculations:")
print("=" * 60)

# Replicate the exact calculation from the code
avg_compression = df.groupby('layer_type')['compression_percentage'].agg(['mean', 'std']).reset_index()
avg_ranks = df.groupby('layer_type')[f'threshold_rank_tau_{tau}'].agg(['mean']).reset_index()

print("Before merge:")
print("avg_compression columns:", avg_compression.columns.tolist())
print("avg_ranks columns:", avg_ranks.columns.tolist())
print()

# Merge with suffixes
avg_compression = avg_compression.merge(avg_ranks, on='layer_type', suffixes=('_pct', '_rank'))

print("After merge:")
print("avg_compression columns:", avg_compression.columns.tolist())
print()

print("Results:")
print("-" * 40)
for _, row in avg_compression.iterrows():
    layer_type = row['layer_type']
    mean_pct = row['mean_pct']
    mean_rank = row['mean_rank']
    
    print(f"{layer_type:15}: {mean_pct:5.1f}% ({mean_rank:5.1f})")
    
    # Manual verification: calculate mean rank directly
    type_data = df[df['layer_type'] == layer_type]
    manual_mean_rank = type_data[f'threshold_rank_tau_{tau}'].mean()
    manual_mean_pct = type_data['compression_percentage'].mean()
    
    print(f"{'Manual check':15}: {manual_mean_pct:5.1f}% ({manual_mean_rank:5.1f})")
    
    # Check if they match
    rank_match = abs(mean_rank - manual_mean_rank) < 0.01
    pct_match = abs(mean_pct - manual_mean_pct) < 0.01
    
    if not rank_match or not pct_match:
        print(f"*** MISMATCH! ***")
    print()

print("\nSample data for verification:")
print("-" * 40)
for layer_type in df['layer_type'].unique():
    type_data = df[df['layer_type'] == layer_type].head(3)
    print(f"\n{layer_type} examples:")
    for _, row in type_data.iterrows():
        rank = row[f'threshold_rank_tau_{tau}']
        max_rank = row['max_rank']
        pct = row['compression_percentage']
        step = row['step']
        print(f"  Step {step:3}: rank={rank:3.0f}, max_rank={max_rank:3.0f}, pct={pct:5.1f}%") 