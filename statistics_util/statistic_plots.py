import os
import time
import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch

def initialize_statistics(num_layers, num_heads):
    stats = {
        'mean': [], #3-D data first D is layer, second D is head, third D is #iter
        'median': [],
        'stdev': [],
        'max': [],
        'min': [],
        'o_mean': [],
        'o_median': [],
        'o_stdev': [],
        'o_max': [],
        'o_min': []
    }

    for _ in range(num_layers):
        stats['mean'].append([[] for _ in range(num_heads)])
        stats['median'].append([[] for _ in range(num_heads)])
        stats['stdev'].append([[] for _ in range(num_heads)])
        stats['max'].append([[] for _ in range(num_heads)])
        stats['min'].append([[] for _ in range(num_heads)])
        stats['o_mean'].append([[] for _ in range(num_heads)])
        stats['o_median'].append([[] for _ in range(num_heads)])
        stats['o_stdev'].append([[] for _ in range(num_heads)])
        stats['o_max'].append([[] for _ in range(num_heads)])
        stats['o_min'].append([[] for _ in range(num_heads)])

    return stats

def create_box_plot(out_dir, plot_data, y_labels, timestamp, data_type, stat_type):
    directory_path = os.path.join(out_dir, 'images')
    os.makedirs(directory_path, exist_ok=True)

    # create a boxplot
    fig = plt.figure(figsize = (10, 7))
    ax = fig.add_subplot(111)

    # Creating axes instance
    ax.boxplot(plot_data, sym = '', patch_artist = True, vert = 0)

    # y-axis labels
    ax.set_yticklabels(y_labels)

    # Removing top axes and right axes
    # ticks
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

    ax.set_title(f"Boxplot of {data_type} {stat_type}")
    plt.savefig(f'{directory_path}/{data_type}_{stat_type}_boxplot_{timestamp}.png')
    plt.close()

def plot_statistics(args, stats, graph_y_labels):
    statistics_to_plot = []
    timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
    directory_path = os.path.join(args.out_dir, 'images')
    os.makedirs(directory_path, exist_ok=True)
    statistics_to_plot = [args.statistic]
    if args.statistic  == "all_stats":
        statistics_to_plot = ['input_mean', 'input_median', 'input_stdev', 'input_max', 'input_min',
                            'output_mean', 'output_median', 'output_stdev', 'output_max', 'output_min']
    elif args.statistic == 'input_all':
        statistics_to_plot = ['input_mean', 'input_median', 'input_stdev', 'input_max', 'input_min']
    elif args.statistic == 'output_all':
        statistics_to_plot = ['output_mean', 'output_median', 'output_stdev', 'output_max', 'output_min']
    for stat in statistics_to_plot:
        parts = stat.split('_')
        data_type = parts[0]  # 'input' or 'output'
        stat_type = parts[1]  # 'mean', 'median', 'stdev', 'max', 'min'

        # to decide whether to use the input or output statistics
        stat_prefix = 'o_' if data_type == 'output' else ''

        # draw the plot
        if args.graph_type == 'plot' or args.graph_type == 'all':
            fig = go.Figure()
            plt.figure(figsize=(18, 8))
            for layer_idx, stats_per_layer in enumerate(stats[stat_prefix + stat_type]):
                for head_idx, data in enumerate(stats_per_layer):
                    fig.add_trace(go.Scatter(
                        x=list(range(len(data))),
                        y=data,
                        mode='lines',
                        name=f'Layer {layer_idx + 1} Head {head_idx + 1}'
                    ))
                    plt.plot(data, label=f'Layer {layer_idx + 1} Head {head_idx + 1}')

            # add titles and legend to Plotly
            fig.update_layout(
                title=f'Change in {stat_type.title()} Values for {data_type.capitalize()} During Training',
                xaxis_title='Training Iteration',
                yaxis_title=f'{stat_type.title()} of {data_type.capitalize()}',
                legend_title='Head/Layer',
                height=890,
                width=1200
            )
            fig.write_html(f'{directory_path}/{data_type}_{stat_type}_changes_plotly_{timestamp}.html')
            fig.write_image(f'{directory_path}/{data_type}_{stat_type}_changes_plotly_{timestamp}.png')

            # add titles and lengend to Matplotlib
            plt.title(f'Change in {stat_type.title()} Values for {data_type.capitalize()} During Training')
            plt.xlabel('Training Iteration')
            plt.ylabel(f'{stat_type.title()} of {data_type.capitalize()}')
            plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), title='Head/Layer')
            # plt.legend(title='Head/Layer')
            plt.grid(True)
            plt.savefig(f'{directory_path}/{data_type}_{stat_type}_changes_plot_{timestamp}.png')
            plt.close()

        if args.graph_type == 'heatmap' or args.graph_type == 'all':
            #data is the value of #iter
            # create xlabels
            num_iters = len(data)
            unit_size = num_iters // 10
            x_labels = [i*unit_size for i in range(10)]

            # create plot_data
            plot_data = []
            for layer_idx, stats_per_layer in enumerate(stats[stat_prefix + stat_type]):
                for head_idx, data in enumerate(stats_per_layer):
                    plot_data.append([])
                    for i in x_labels:
                        plot_data[-1].append(data[i])
            plot_data = np.array(plot_data)

            ######
            fig, ax = plt.subplots(figsize=(8,10))
            im = ax.imshow(plot_data)
            # Name the x and y axis
            ax.set_xticks(np.arange(len(x_labels)), labels=x_labels)
            ax.set_yticks(np.arange(len(graph_y_labels)), labels=graph_y_labels)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
            ax.set_xlabel("Number of Iterations", fontweight="bold")

            # Create a colorbar
            cbar = ax.figure.colorbar(im, ax=ax)
            cbar.ax.set_ylabel(stat_type, rotation=-90, va="bottom")

            ax.set_title(f"Heatmap of {data_type} {stat_type}")
            plt.savefig(f'{directory_path}/{data_type}_{stat_type}_heatmap_{timestamp}.png')
            plt.close()

        if args.graph_type == 'boxplot' or args.graph_type == 'all':
            # create plot_data
            plot_data = []
            for layer_idx, stats_per_layer in enumerate(stats[stat_prefix + stat_type]):
                for head_idx, data in enumerate(stats_per_layer):
                    plot_data.append(np.array(data))

            create_box_plot(args.out_dir, plot_data, graph_y_labels, timestamp, data_type, stat_type)

def create_statistics(self, graph_y_labels):
    if self.args.softmax_variant_attn in ['consmax', 'polymax', 'strongermax']:
        betas = []
        gammas = []
        i_sum_vals = []
        i_means = []
        i_medians = []
        i_stdevs = []
        i_max_values = []
        i_min_values = []
        denominator = []
        o_sum_vals = []
        o_means = []
        o_medians = []
        o_stdevs = []
        o_max_values = []
        o_min_values = []

        box_plot_input_data = []
        box_plot_output_data = []

        for layer in range (self.args.n_layer):
            # Inputs
            inputs_location = f"transformer.h[{layer}].attn.softmax_layer_attn.inputs"

            softmax_input = eval(f"self.model.{inputs_location}").to('cpu').to(torch.float32)


            ## Get first batch
            i_first_batch = softmax_input[0]
            i_first_batch[i_first_batch == float('-inf')] = float('NaN')

            for i, i_head in enumerate(i_first_batch):
                ## Flatten across heads, height, and width
                flattened = i_head.view(-1)

                ## Calculate statistics
                i_means.append(torch.nanmean(flattened).item())
                i_medians.append(torch.nanmedian(flattened).item())

                # Standard deviation, ignoring NaNs
                mask = ~torch.isnan(i_head)
                i_stdevs.append(torch.std(i_head[mask]).item())
                i_sum_vals.append(torch.sum(i_head[mask]).item())

                if self.iter_num % self.args.box_plot_interval == 0 and (self.args.box_plot_statistic == "all" or self.args.box_plot_statistic == "input"):
                    box_plot_input_data.append(i_head[mask].detach().numpy())

                # Max, temporarily replacing NaNs with -inf for calculation
                i_max_values.append(torch.max(torch.where(torch.isnan(i_head), torch.tensor(float('-inf')), i_head)).item())
                i_min_values.append(torch.min(torch.where(torch.isnan(i_head), torch.tensor(float('inf')), i_head)).item())
                # Denominator computation for i_head
                exp_flattened = torch.exp(i_head[mask])
                sum = torch.sum(exp_flattened)
                denominator.append(sum.item())

                # Append statistic to the input list of each head in each layer
                self.stats['mean'][layer][i].append(torch.nanmean(flattened).item())
                self.stats['median'][layer][i].append(torch.nanmedian(flattened).item())
                self.stats['stdev'][layer][i].append(torch.std(i_head[mask]).item())
                self.stats['max'][layer][i].append(torch.max(torch.where(torch.isnan(i_head), torch.tensor(float('-inf')), i_head)).item())
                self.stats['min'][layer][i].append(torch.min(torch.where(torch.isnan(i_head), torch.tensor(float('inf')), i_head)).item())



            outputs_location = f"transformer.h[{layer}].attn.softmax_layer_attn.outputs"
            softmax_output = eval(f"self.model.{outputs_location}").to('cpu').to(torch.float32)

            o_first_batch = softmax_output[0]
            o_first_batch[o_first_batch == float('-inf')] = float('NaN')
            for i, o_head in enumerate(o_first_batch):

                # Step 3: Flatten across heads, height, and width
                flattened = o_head.view(-1)

                # Step 4: Calculate statistics
                ## Calculate statistics
                o_means.append(torch.nanmean(flattened).item())
                o_medians.append(torch.nanmedian(flattened).item())
                # Standard deviation, ignoring NaNs
                mask = ~torch.isnan(o_head)
                o_stdevs.append(torch.std(o_head[mask]).item())
                o_sum_vals.append(torch.sum(o_head[mask]).item())

                if self.iter_num % self.args.box_plot_interval == 0 and (self.args.box_plot_statistic == "all" or self.args.box_plot_statistic == "output"):
                    box_plot_output_data.append(o_head[mask].detach().numpy())

                # Max, temporarily replacing NaNs with -inf for calculation
                o_max_values.append(torch.max(torch.where(torch.isnan(o_head), torch.tensor(float('-inf')), o_head)).item())
                o_min_values.append(torch.min(torch.where(torch.isnan(o_head), torch.tensor(float('inf')), o_head)).item())

                # Append statistic to the output list of each head in each layer
                self.stats['o_mean'][layer][i].append(torch.nanmean(flattened).item())
                self.stats['o_median'][layer][i].append(torch.nanmedian(flattened).item())
                self.stats['o_stdev'][layer][i].append(torch.std(o_head[mask]).item())
                self.stats['o_max'][layer][i].append(torch.max(torch.where(torch.isnan(o_head), torch.tensor(float('-inf')), o_head)).item())
                self.stats['o_min'][layer][i].append(torch.min(torch.where(torch.isnan(o_head), torch.tensor(float('inf')), o_head)).item())

            #BETA GAMMA
            if self.args.softmax_variant_attn == 'consmax':
                gamma_location = f"transformer.h[{layer}].attn.softmax_layer_attn.gamma"
                beta_location = f"transformer.h[{layer}].attn.softmax_layer_attn.beta"

                gamma = eval(f"self.model.{gamma_location}")
                gammas.append(gamma[0].item()) # are there more than just gamma 0?
                # print("gammas",gamma) # are there more than just gamma 0?

                beta = eval(f"self.model.{beta_location}")
                betas.append(beta[0].item()) # are there more than just beta 0?
                # print("betas",beta,) # are there more than just beta 0?

                self.log_gamma_beta(gamma, beta, self.iter_num, layer)

        if self.args.box_plot_statistic and (self.iter_num % self.args.box_plot_interval == 0) and self.iter_num != 0:
            timestamp = time.strftime('%Y-%m-%d_%H-%M-%S', time.localtime())
            if self.args.box_plot_statistic == "all":
                create_box_plot(self.args.out_dir, box_plot_input_data, graph_y_labels, timestamp, "input", self.iter_num)
                create_box_plot(self.args.out_dir, box_plot_output_data, graph_y_labels, timestamp, "output", self.iter_num)
            elif self.args.box_plot_statistic == "input":
                create_box_plot(self.args.out_dir, box_plot_input_data, graph_y_labels, timestamp, self.args.box_plot_statistic, self.iter_num)
            else:
                create_box_plot(self.args.out_dir, box_plot_output_data, graph_y_labels, timestamp, self.args.box_plot_statistic, self.iter_num)


        self.write_to_csv(self.iter_num,
                            *i_sum_vals,
                            *i_means,
                            *i_medians,
                            *i_stdevs,
                            *i_max_values,
                            *i_min_values,
                            *denominator,
                            prefix="inputs")
        self.write_to_csv(self.iter_num,
                            *o_sum_vals,
                            *o_means,
                            *o_medians,
                            *o_stdevs,
                            *o_max_values,
                            *o_min_values,
                            prefix="outputs")
        if self.args.softmax_variant_attn == 'consmax':
            self.write_to_csv(self.iter_num, *betas, *gammas, prefix="beta_gamma")

        """
        if self.iter_num % 50 == 0:
            inputs = []
            outputs = []

            for layer in range (self.args.n_layer):
                inputs_location = f"transformer.h[{layer}].attn.softmax_layer.inputs"
                outputs_location = f"transformer.h[{layer}].attn.softmax_layer.outputs"

                gamma = eval(f"self.model.{gamma_location}")
                gammas.append(gamma[0].item()) # are there more than just gamma 0?
                # print("gammas",gamma) # are there more than just gamma 0?

                beta = eval(f"self.model.{beta_location}")
                betas.append(beta[0].item()) # are there more than just beta 0?
                # print("betas",beta,) # are there more than just beta 0?

                self.log_gamma_beta(gamma, beta, self.iter_num, layer)


            self.write_to_csv(self.iter_num, *betas, *gammas, prefix="beta_gamma")


        """
