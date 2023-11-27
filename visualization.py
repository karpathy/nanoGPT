"""
Module with functions for visualizing model loss & perplexity, in a few different manners
"""

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as patches 
from matplotlib.colors import LinearSegmentedColormap
from typing import Union, List




# General Parameters 
font_config = {
    "fontname": "JetBrains Mono",
    "fontsize": 24,
}
catpuccin_hue = ['#8aadf4', '#cad3f5', '#ee99a0'] #catppuccin colors -- from the Macchiato scheme


# Display Perplexities @ positions in a sequence of tokens
def display_colored_text(sentence: List[str], color_values, cmap: Union[List[str], str]='magma', outfilename: str="out.png"):
    assert len(sentence) == len(color_values), "Text and color values must be the same length"

    # set dims for the figure
    word_proportion = 2.5
    figure_width = word_proportion * len(sentence)
    fig, ax = plt.subplots(figsize=(figure_width, 1.5))

    if type(cmap) == str:
        cmap = plt.get_cmap(cmap)
    elif type(cmap) == list and type(cmap[0]) == str:
        cmap = LinearSegmentedColormap.from_list("mycmap", cmap)

    color_values_scaled = normalize_data(color_values)
    # color_values_scaled = [0] * len(color_values_scaled)
    # color_values_scaled = color_values
    colors = cmap(color_values_scaled)

    for i, (tok, color) in enumerate(zip(sentence, colors)):

        # Method 1 -- color the chars themselves
        # ax.text(i + 0.5, 0.5, ch, color=color ha='center', va='center', fontsize=font_size)

        rect = patches.Rectangle((i, 0), 1, 1, facecolor=color)
        ax.add_patch(rect)
        ax.text(i + 0.5, 0.5, tok, color='white', ha='center', va='center', **font_config)

    ax.set_xlim(0, len(sentence))
    ax.set_ylim(0, 1)
    ax.axis('off') #makes the bg plain white
    # ax.add_patch(patches.Rectangle((0, 0), len(sentence), 1, fill=False, edgecolor="black", linewidth=2))


    # Create a colorbar -- V2
    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])

    norm = mpl.colors.Normalize(vmin=np.min(color_values), vmax=np.max(color_values))
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label("Perplexities", loc="center", font=font_config["fontname"])
    # plt.show()
    plt.savefig(outfilename, format="png", dpi=1200)


def normalize_data(color_values):
    # cast dtype if list
    if type(color_values) != np.array:
        color_values = np.array(color_values)
    min_val = np.min(color_values)
    max_val = np.max(color_values)
    # for the actual task, perhaps we set hard bins on Perplexity values (mask out really large/small ppls)
    # min_val = 1.0  #the lowest possible ppl
    # max_val = 10.0 #if higher than this, big problem

    color_values_scaled = (color_values - min_val) / (max_val - min_val)
    return color_values_scaled




sent = ["The", "Ocean", "Waves", "is", "a", "great", "film"]
sent.extend(["woz"] * 5)
# print(len(sen))
# display_colored_text(sent, [1, 20, 0, 1, 1e13])


color_values = np.linspace(0, 1, len(sent))
display_colored_text(sent, color_values, catpuccin_hue)
