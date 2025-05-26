"""
Analysis and visualization script for gradient rank analysis results.
Reads the CSV output from gradient_rank_analysis.py and creates plots.

Usage:
$ python analyze_gradient_ranks.py --results_dir out-gradient-rank-analysis
"""

import argparse
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from pathlib import Path
import glob

def load_results(results_dir):
    """Load gradient rank analysis results."""
    csv_path = os.path.join(results_dir, 'gradient_rank_summary.csv')
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Results file not found: {csv_path}")
    
    df = pd.read_csv(csv_path)
    return df

def load_singular_values(results_dir):
    """Load singular values from JSON files."""
    json_files = glob.glob(os.path.join(results_dir, 'gradient_rank_analysis_step_*.json'))
    
    if not json_files:
        print("No JSON files with singular values found.")
        return None
    
    # Sort by step number
    json_files.sort(key=lambda x: int(x.split('_step_')[1].split('.')[0]))
    
    singular_value_data = []
    
    for json_file in json_files:
        step = int(os.path.basename(json_file).split('_step_')[1].split('.')[0])
        
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        for layer_name, layer_data in data.items():
            if layer_data.get('singular_values') is None:
                continue
                
            singular_values = np.array(layer_data['singular_values'])
            
            # Determine layer type
            layer_type = 'unknown'
            if 'attn' in layer_name and 'proj' in layer_name:
                layer_type = 'attention_proj'
            elif 'attn' in layer_name:
                layer_type = 'attention'
            elif 'mlp' in layer_name:
                layer_type = 'mlp'
            elif 'wte' in layer_name or 'wpe' in layer_name or 'embed' in layer_name:
                layer_type = 'embedding'
            elif 'head' in layer_name:
                layer_type = 'output_head'
            
            # Calculate statistics
            if len(singular_values) > 0:
                singular_value_data.append({
                    'step': step,
                    'layer_name': layer_name,
                    'layer_type': layer_type,
                    'sigma_max': float(singular_values[0]),  # Largest singular value
                    'sigma_min': float(singular_values[-1]),  # Smallest singular value
                    'sigma_median': float(np.median(singular_values)),
                    'sigma_mean': float(np.mean(singular_values)),
                    'sigma_std': float(np.std(singular_values)),
                    'num_singular_values': len(singular_values),
                    'condition_number': float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 1e-12 else float('inf')
                })
    
    return pd.DataFrame(singular_value_data) if singular_value_data else None

def clean_layer_types(df):
    """Clean up layer type names for better display in plots."""
    df = df.copy()
    # Replace 'unknown' with 'embedding' since those are actually embedding layers
    df['layer_type'] = df['layer_type'].replace('unknown', 'embedding')
    return df

def plot_rank_evolution(df, tau=0.1, save_dir=None):
    """Plot how effective rank evolves during training for different layer types."""
    
    plt.figure(figsize=(15, 10))
    
    # Define layer types and their colors
    layer_types = df['layer_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_types)))
    
    # Plot threshold rank evolution
    plt.subplot(2, 2, 1)
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        
        # Group by step and compute mean/std
        grouped = type_data.groupby('step')[f'threshold_rank_tau_{tau}'].agg(['mean', 'std']).reset_index()
        
        plt.plot(grouped['step'], grouped['mean'], label=layer_type, color=color, linewidth=2)
        if len(grouped) > 1:  # Only add error bars if we have multiple points
            plt.fill_between(grouped['step'], 
                           grouped['mean'] - grouped['std'], 
                           grouped['mean'] + grouped['std'], 
                           alpha=0.2, color=color)
    
    plt.xlabel('Training Step')
    plt.ylabel(f'Threshold Rank (τ={tau})')
    plt.title('Threshold Rank Evolution by Layer Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot entropy rank evolution
    plt.subplot(2, 2, 2)
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        grouped = type_data.groupby('step')[f'entropy_rank_tau_{tau}'].agg(['mean', 'std']).reset_index()
        
        plt.plot(grouped['step'], grouped['mean'], label=layer_type, color=color, linewidth=2)
        if len(grouped) > 1:
            plt.fill_between(grouped['step'], 
                           grouped['mean'] - grouped['std'], 
                           grouped['mean'] + grouped['std'], 
                           alpha=0.2, color=color)
    
    plt.xlabel('Training Step')
    plt.ylabel(f'Entropy Rank (τ={tau})')
    plt.title('Entropy Rank Evolution by Layer Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot stable rank evolution
    plt.subplot(2, 2, 3)
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        grouped = type_data.groupby('step')[f'stable_rank_tau_{tau}'].agg(['mean', 'std']).reset_index()
        
        plt.plot(grouped['step'], grouped['mean'], label=layer_type, color=color, linewidth=2)
        if len(grouped) > 1:
            plt.fill_between(grouped['step'], 
                           grouped['mean'] - grouped['std'], 
                           grouped['mean'] + grouped['std'], 
                           alpha=0.2, color=color)
    
    plt.xlabel('Training Step')
    plt.ylabel(f'Stable Rank (τ={tau})')
    plt.title('Stable Rank Evolution by Layer Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot rank vs max possible rank
    plt.subplot(2, 2, 4)
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        # Take the last step for each layer
        last_step_data = type_data.loc[type_data.groupby('layer_name')['step'].idxmax()]
        
        plt.scatter(last_step_data['max_rank'], 
                   last_step_data[f'threshold_rank_tau_{tau}'], 
                   label=layer_type, color=color, alpha=0.7, s=60)
    
    # Add diagonal line (rank = max_rank)
    max_possible = df['max_rank'].max()
    plt.plot([0, max_possible], [0, max_possible], 'k--', alpha=0.5, label='Max Possible')
    
    plt.xlabel('Maximum Possible Rank')
    plt.ylabel(f'Effective Rank (τ={tau})')
    plt.title('Effective Rank vs Maximum Possible Rank (Final Step)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'rank_evolution_tau_{tau}.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'rank_evolution_tau_{tau}.pdf'), bbox_inches='tight')
    
    plt.close()  # Free memory

def plot_tau_comparison(df, save_dir=None):
    """Compare effective ranks for different tau thresholds."""
    
    # Get the available tau values from column names
    tau_cols = [col for col in df.columns if col.startswith('threshold_rank_tau_')]
    tau_values = [float(col.split('_')[-1]) for col in tau_cols]
    tau_values.sort()
    
    plt.figure(figsize=(12, 8))
    
    # Plot for each layer type
    layer_types = df['layer_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_types)))
    
    plt.subplot(2, 2, 1)
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        # Take last step data
        last_step = type_data['step'].max()
        last_data = type_data[type_data['step'] == last_step]
        
        mean_ranks = []
        for tau in tau_values:
            mean_rank = last_data[f'threshold_rank_tau_{tau}'].mean()
            mean_ranks.append(mean_rank)
        
        plt.semilogx(tau_values, mean_ranks, 'o-', label=layer_type, color=color, linewidth=2)
    
    plt.xlabel('Threshold τ')
    plt.ylabel('Mean Threshold Rank')
    plt.title('Threshold Rank vs τ (Final Step)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot rank distribution for a specific tau
    plt.subplot(2, 2, 2)
    tau_mid = tau_values[len(tau_values)//2]  # Middle tau value
    
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        last_step = type_data['step'].max()
        last_data = type_data[type_data['step'] == last_step]
        
        ranks = last_data[f'threshold_rank_tau_{tau_mid}']
        plt.hist(ranks, bins=10, alpha=0.6, label=f'{layer_type}', color=color, density=True)
    
    plt.xlabel(f'Threshold Rank (τ={tau_mid})')
    plt.ylabel('Density')
    plt.title(f'Rank Distribution at Final Step (τ={tau_mid})')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Rank compression ratio
    plt.subplot(2, 2, 3)
    for layer_type, color in zip(layer_types, colors):
        type_data = df[df['layer_type'] == layer_type]
        last_step = type_data['step'].max()
        last_data = type_data[type_data['step'] == last_step]
        
        compression_ratios = []
        for tau in tau_values:
            ratio = (last_data[f'threshold_rank_tau_{tau}'] / last_data['max_rank']).mean()
            compression_ratios.append(ratio)
        
        plt.semilogx(tau_values, compression_ratios, 'o-', label=layer_type, color=color, linewidth=2)
    
    plt.xlabel('Threshold τ')
    plt.ylabel('Compression Ratio (Rank/Max Rank)')
    plt.title('Rank Compression vs τ')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Summary statistics
    plt.subplot(2, 2, 4)
    
    # Calculate overall statistics
    last_step = df['step'].max()
    last_data = df[df['step'] == last_step]
    
    stats_text = f"Summary Statistics (Final Step {last_step}):\n\n"
    
    for tau in [0.01, 0.1, 0.5]:  # Show key tau values
        mean_rank = last_data[f'threshold_rank_tau_{tau}'].mean()
        mean_compression = (last_data[f'threshold_rank_tau_{tau}'] / last_data['max_rank']).mean()
        stats_text += f"τ = {tau}:\n"
        stats_text += f"  Mean Rank: {mean_rank:.1f}\n"
        stats_text += f"  Mean Compression: {mean_compression:.2f}\n\n"
    
    plt.text(0.1, 0.9, stats_text, transform=plt.gca().transAxes, fontsize=10, 
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'tau_comparison.png'), dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, 'tau_comparison.pdf'), bbox_inches='tight')
    
    plt.close()  # Free memory

def print_summary(df):
    """Print a summary of the gradient rank analysis."""
    print("=" * 60)
    print("GRADIENT RANK ANALYSIS SUMMARY")
    print("=" * 60)
    
    # Basic info
    total_steps = df['step'].nunique()
    total_layers = df['layer_name'].nunique()
    layer_types = df['layer_type'].unique()
    
    print(f"Training steps analyzed: {total_steps}")
    print(f"Total layers: {total_layers}")
    print(f"Layer types: {', '.join(layer_types)}")
    print()
    
    # Final step analysis
    final_step = df['step'].max()
    final_data = df[df['step'] == final_step]
    
    print(f"FINAL STEP ANALYSIS (Step {final_step}):")
    print("-" * 40)
    
    # For each layer type
    for layer_type in layer_types:
        type_data = final_data[final_data['layer_type'] == layer_type]
        if len(type_data) == 0:
            continue
            
        print(f"\n{layer_type.upper()} layers ({len(type_data)} layers):")
        
        # Statistics for tau=0.1 (commonly used threshold)
        tau = 0.1
        threshold_ranks = type_data[f'threshold_rank_tau_{tau}']
        entropy_ranks = type_data[f'entropy_rank_tau_{tau}']
        max_ranks = type_data['max_rank']
        compression_ratios = threshold_ranks / max_ranks
        
        print(f"  Threshold Rank (τ={tau}): {threshold_ranks.mean():.1f} ± {threshold_ranks.std():.1f}")
        print(f"  Entropy Rank (τ={tau}): {entropy_ranks.mean():.1f} ± {entropy_ranks.std():.1f}")
        print(f"  Max Possible Rank: {max_ranks.mean():.1f} ± {max_ranks.std():.1f}")
        print(f"  Compression Ratio: {compression_ratios.mean():.2f} ± {compression_ratios.std():.2f}")
    
    print("\n" + "=" * 60)

def plot_compression_ratio(df, tau=0.1, save_dir=None, sv_df=None):
    """Plot compression ratio (effective rank / matrix dimension) by layer type."""
    
    plt.figure(figsize=(20, 12))  # Increased width for 4 subplots
    
    # Define layer types and their colors
    layer_types = df['layer_type'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(layer_types)))
    
    # Calculate compression ratio for each row
    df_with_ratio = df.copy()
    df_with_ratio['compression_ratio'] = df_with_ratio[f'threshold_rank_tau_{tau}'] / df_with_ratio['max_rank']
    df_with_ratio['compression_percentage'] = df_with_ratio['compression_ratio'] * 100
    
    # Plot 1: Compression ratio evolution during training
    plt.subplot(2, 2, 1)
    for layer_type, color in zip(layer_types, colors):
        type_data = df_with_ratio[df_with_ratio['layer_type'] == layer_type]
        
        # Group by step and compute mean/std
        grouped = type_data.groupby('step')['compression_percentage'].agg(['mean', 'std']).reset_index()
        
        plt.plot(grouped['step'], grouped['mean'], label=layer_type, color=color, linewidth=2)
        if len(grouped) > 1:
            plt.fill_between(grouped['step'], 
                           grouped['mean'] - grouped['std'], 
                           grouped['mean'] + grouped['std'], 
                           alpha=0.2, color=color)
    
    plt.xlabel('Training Step')
    plt.ylabel(f'Compression Ratio (%) [τ={tau}]')
    plt.title('Gradient Compression Ratio Evolution by Layer Type')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Set adaptive y-axis limits based on actual data range
    all_data = df_with_ratio['compression_percentage']
    y_min = max(0, all_data.min() - 2)  # Add 2% padding below, but don't go below 0
    y_max = all_data.max() + 5  # Add 5% padding above
    plt.ylim(y_min, y_max)
    
    # Plot 2: Final compression ratio distribution by layer type
    plt.subplot(2, 2, 2)
    final_step = df_with_ratio['step'].max()
    final_data = df_with_ratio[df_with_ratio['step'] == final_step]
    
    # Create box plot
    layer_data = []
    layer_labels = []
    for layer_type in layer_types:
        type_data = final_data[final_data['layer_type'] == layer_type]['compression_percentage']
        if len(type_data) > 0:
            layer_data.append(type_data)
            layer_labels.append(layer_type)
    
    bp = plt.boxplot(layer_data, labels=layer_labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors[:len(layer_data)]):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    plt.xlabel('Layer Type')
    plt.ylabel(f'Compression Ratio (%) [τ={tau}]')
    plt.title(f'Final Compression Ratio Distribution (Step {final_step})')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Average compression ratio by layer type
    plt.subplot(2, 2, 3)
    avg_compression = df_with_ratio.groupby('layer_type')['compression_percentage'].agg(['mean', 'std']).reset_index()
    
    # Calculate mean ranks for each layer type
    avg_ranks = df_with_ratio.groupby('layer_type')[f'threshold_rank_tau_{tau}'].agg(['mean']).reset_index()
    avg_compression = avg_compression.merge(avg_ranks, on='layer_type', suffixes=('_pct', '_rank'))
    
    # Ensure consistent ordering with other plots
    layer_type_order = {layer_type: i for i, layer_type in enumerate(layer_types)}
    avg_compression['order'] = avg_compression['layer_type'].map(layer_type_order)
    avg_compression = avg_compression.sort_values('order')
    
    x_pos = np.arange(len(avg_compression))
    # Use the same colors as other plots, ordered consistently
    plot_colors = [colors[layer_type_order[lt]] for lt in avg_compression['layer_type']]
    
    bars = plt.bar(x_pos, avg_compression['mean_pct'], 
                   yerr=avg_compression['std'], 
                   color=plot_colors, 
                   alpha=0.7, capsize=5)
    
    plt.xlabel('Layer Type')
    plt.ylabel(f'Average Compression Ratio (%) [τ={tau}]')
    plt.title('Average Compression Ratio by Layer Type')
    plt.xticks(x_pos, avg_compression['layer_type'], rotation=45)
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars with both percentage and mean rank
    for bar, mean_pct, std_val, mean_rank in zip(bars, avg_compression['mean_pct'], 
                                                 avg_compression['std'], avg_compression['mean_rank']):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.5,
                f'{mean_pct:.1f}% ({mean_rank:.0f})', ha='center', va='bottom', fontsize=10)
    
    # Plot 4: Singular value statistics by layer type
    plt.subplot(2, 2, 4)
    
    if sv_df is not None and len(sv_df) > 0:
        # Clean layer types for singular value data
        sv_df = sv_df.copy()
        sv_df['layer_type'] = sv_df['layer_type'].replace('unknown', 'embedding')
        
        # Plot all singular value statistics for each layer type
        for layer_type, color in zip(layer_types, colors):
            type_data = sv_df[sv_df['layer_type'] == layer_type]
            
            if len(type_data) > 0:
                # Group by step and compute mean/std for different statistics
                grouped_max = type_data.groupby('step')['sigma_max'].agg(['mean', 'std']).reset_index()
                grouped_mean = type_data.groupby('step')['sigma_mean'].agg(['mean', 'std']).reset_index()
                grouped_median = type_data.groupby('step')['sigma_median'].agg(['mean', 'std']).reset_index()
                grouped_min = type_data.groupby('step')['sigma_min'].agg(['mean', 'std']).reset_index()
                
                # Plot max singular value (solid line, thick)
                plt.plot(grouped_max['step'], grouped_max['mean'], 
                        color=color, linewidth=2.5, linestyle='-', alpha=0.9,
                        label=f'{layer_type} (max)' if layer_type == layer_types[0] else "")
                
                # Plot mean singular value (dashed line, medium)
                plt.plot(grouped_mean['step'], grouped_mean['mean'], 
                        color=color, linewidth=2, linestyle='--', alpha=0.8)
                
                # Plot median singular value (dotted line, medium)
                plt.plot(grouped_median['step'], grouped_median['mean'], 
                        color=color, linewidth=1.5, linestyle=':', alpha=0.7)
                
                # Plot min singular value (dash-dot line, thin)
                plt.plot(grouped_min['step'], grouped_min['mean'], 
                        color=color, linewidth=1, linestyle='-.', alpha=0.6)
                
                # Note: No confidence bands in this plot to keep it clean with multiple lines per layer type
        
        plt.xlabel('Training Step')
        plt.ylabel('Singular Value Magnitude')
        plt.title('Singular Value Statistics Evolution by Layer Type\n(Solid: Max, Dashed: Mean, Dotted: Median, Dash-dot: Min)')
        
        # Create custom legend for line styles
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], color='black', linewidth=2.5, linestyle='-', label='Max σ'),
            Line2D([0], [0], color='black', linewidth=2, linestyle='--', label='Mean σ'),
            Line2D([0], [0], color='black', linewidth=1.5, linestyle=':', label='Median σ'),
            Line2D([0], [0], color='black', linewidth=1, linestyle='-.', label='Min σ')
        ]
        
        # Add layer type legend
        layer_legend_elements = [Line2D([0], [0], color=color, linewidth=2, label=layer_type) 
                               for layer_type, color in zip(layer_types, colors)]
        
        # Combine legends
        all_legend_elements = legend_elements + [Line2D([0], [0], color='white', linewidth=0, label='')] + layer_legend_elements
        
        plt.legend(handles=all_legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')  # Log scale for better visualization
        
        # Set adaptive y-axis limits
        all_sigma_max = sv_df['sigma_max']
        all_sigma_min = sv_df['sigma_min']
        y_min_sigma = max(1e-8, all_sigma_min.min() * 0.5)  # Include min values
        y_max_sigma = all_sigma_max.max() * 2
        plt.ylim(y_min_sigma, y_max_sigma)
    else:
        # Fallback: show stable rank if no singular value data
        stable_rank_col = f'stable_rank_tau_{tau}'
        
        for layer_type, color in zip(layer_types, colors):
            type_data = df_with_ratio[df_with_ratio['layer_type'] == layer_type]
            
            # Group by step and compute mean/std for stable rank
            grouped = type_data.groupby('step')[stable_rank_col].agg(['mean', 'std']).reset_index()
            
            plt.plot(grouped['step'], grouped['mean'], label=layer_type, color=color, linewidth=2)
            if len(grouped) > 1:
                plt.fill_between(grouped['step'], 
                               grouped['mean'] - grouped['std'], 
                               grouped['mean'] + grouped['std'], 
                               alpha=0.2, color=color)
        
        plt.xlabel('Training Step')
        plt.ylabel(f'Stable Rank [τ={tau}]')
        plt.title('Stable Rank Evolution by Layer Type\n(||A||²_F / ||A||²_2 - Fallback)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Set adaptive y-axis limits for stable rank
        all_stable_rank = df_with_ratio[stable_rank_col]
        y_min_stable = max(1.0, all_stable_rank.min() - 0.1)
        y_max_stable = all_stable_rank.max() + 0.2
        plt.ylim(y_min_stable, y_max_stable)
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, f'compression_ratio_tau_{tau}.pdf'), 
                   dpi=300, bbox_inches='tight')
        plt.savefig(os.path.join(save_dir, f'compression_ratio_tau_{tau}.png'), 
                   dpi=300, bbox_inches='tight')
        print(f"Compression ratio plots saved to {save_dir}")
    
    plt.close()  # Free memory

def main():
    parser = argparse.ArgumentParser(description='Analyze gradient rank analysis results')
    parser.add_argument('--results_dir', type=str, default='out-gradient-rank-analysis',
                       help='Directory containing gradient rank analysis results')
    parser.add_argument('--tau', type=float, default=0.1,
                       help='Tau threshold for main analysis plots')
    parser.add_argument('--save_plots', action='store_true',
                       help='Save plots to results directory')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from: {args.results_dir}")
    try:
        df = load_results(args.results_dir)
        df = clean_layer_types(df)  # Clean up layer type names
        print(f"Loaded {len(df)} records from gradient rank analysis")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        return
    
    # Load singular value data
    print("Loading singular value data from JSON files...")
    sv_df = load_singular_values(args.results_dir)
    if sv_df is not None:
        print(f"Loaded {len(sv_df)} singular value records from {sv_df['step'].nunique()} steps")
    else:
        print("No singular value data found, will use fallback plots")
    
    # Print summary
    print_summary(df)
    
    # Set up save directory if requested
    save_dir = args.results_dir if args.save_plots else None
    
    # Create plots
    print(f"\nCreating rank evolution plots (τ={args.tau})...")
    plot_rank_evolution(df, tau=args.tau, save_dir=save_dir)
    
    print("Creating tau comparison plots...")
    plot_tau_comparison(df, save_dir=save_dir)
    
    print("Creating compression ratio plots with singular value analysis...")
    plot_compression_ratio(df, tau=args.tau, save_dir=save_dir, sv_df=sv_df)
    
    if args.save_plots:
        print(f"Plots saved to: {args.results_dir}")

if __name__ == "__main__":
    main() 