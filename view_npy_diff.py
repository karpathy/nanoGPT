import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

def plot_heatmaps(original_tensor, final_tensor, fontsize, fmt, cmap):
    diff_tensor = final_tensor - original_tensor
    
    # Determine the global min and max for consistent scaling
    vmin = min(np.min(original_tensor), np.min(final_tensor), np.min(diff_tensor))
    vmax = max(np.max(original_tensor), np.max(final_tensor), np.max(diff_tensor))

    plt.figure(figsize=(18, 6))

    # Original Tensor Heatmap
    plt.subplot(1, 3, 1)
    sns.heatmap(original_tensor, annot=True, cmap=cmap, cbar=True, center=0, annot_kws={"size": fontsize}, fmt=fmt, vmin=vmin, vmax=vmax)
    plt.title('Original Tensor')

    # Difference Tensor Heatmap
    plt.subplot(1, 3, 2)
    sns.heatmap(diff_tensor, annot=True, cmap=cmap, cbar=True, center=0, annot_kws={"size": fontsize}, fmt=fmt, vmin=vmin, vmax=vmax)
    plt.title('Difference (Final - Original)')

    # Final Tensor Heatmap
    plt.subplot(1, 3, 3)
    sns.heatmap(final_tensor, annot=True, cmap=cmap, cbar=True, center=0, annot_kws={"size": fontsize}, fmt=fmt, vmin=vmin, vmax=vmax)
    plt.title('Final Tensor')

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot heatmaps of original, final, and difference tensors.')
    parser.add_argument('original_filename', type=str, help='The filename of the original .npy file')
    parser.add_argument('final_filename', type=str, help='The filename of the final .npy file')
    parser.add_argument('--fontsize', type=int, default=10, help='Font size for the annotations in the heatmaps')
    parser.add_argument('--digits', type=int, default=2, help='Limit on the number of digits for annotations')
    parser.add_argument('--cmap', type=str, default='coolwarm', choices=['viridis', 'plasma', 'inferno', 'magma', 'cividis', 'Greys', 'Purples', 'Blues', 'Greens', 'Oranges', 'Reds', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn', 'BuGn', 'YlGn', 'coolwarm', 'bwr', 'seismic'], help='Colormap scheme for the heatmaps')
    args = parser.parse_args()

    original_tensor = np.load(args.original_filename)
    final_tensor = np.load(args.final_filename)

    fmt = f".{args.digits}f"

    plot_heatmaps(original_tensor, final_tensor, args.fontsize, fmt, args.cmap)

if __name__ == "__main__":
    main()

