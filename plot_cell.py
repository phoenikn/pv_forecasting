import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl

mpl.rcParams["font.size"] = 18

def split_metric(table: pd.DataFrame):
    metrics_dict = {}
    for i in range(int(table.shape[0] / 7)):
        row = i * 7
        metric = table[row: row + 7]
        metric_name = metric.index[0]
        metric_matrix = metric[1:]
        metrics_dict[metric_name] = metric_matrix
    return metrics_dict


def plot_metric(figure_title: str, metric: pd.DataFrame, ax, show_legend=0):
    colors = ["#334f65", "#0c84c6", "#41b7ac", "#eed777", "#b3974e", "#5f6694"]
    metric = pd.DataFrame(metric.values.T, columns=metric.index, index=metric.columns)
    metric.plot(kind='bar', color=colors, zorder=100, ax=ax, edgecolor='black', width=0.6)
    if show_legend == 1:
        ax.legend(prop={}, loc='upper center', bbox_to_anchor=(0.5, -0.12),
                  fancybox=True, shadow=True, ncol=3, frameon=False)
    elif show_legend == 0:
        ax.get_legend().remove()
    elif show_legend == 2:
        ax.legend(prop={}, loc='upper center', bbox_to_anchor=(1.2, -0.18),
                  fancybox=True, shadow=True, ncol=6, frameon=False)
    ax.set_xlabel("Forecasting Horizons")
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    ax.set_title(figure_title, size=20)
    ax.grid(zorder=0)


def plot_pair(figure_titles, metrics, filename, figsize=(8, 12)):
    fig = plt.figure(figsize=figsize)
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    fig.tight_layout(pad=5)
    plot_metric(figure_titles[0], metrics[0], ax1)
    plot_metric(figure_titles[1], metrics[1], ax2, 1)

    plt.savefig(filename, bbox_inches='tight', pad_inches=0.00)

    # plt.show(pad_inches=0.2)


def plot_pairs(figure_titles, metrics, filename, figsize=(8, 12)):
    fig = plt.figure(figsize=figsize)
    axs = []
    for i in range(8):
        axs.append(fig.add_subplot(241 + i))
    fig.tight_layout(pad=4)

    for i in range(len(axs)):
        plot_metric(figure_titles[i], metrics[i], axs[i], 2 if i == 5 else 0)

    plt.savefig(filename, bbox_inches='tight')

    # plt.show()


excel = pd.read_csv("vit/formal metrics pd.csv", index_col=0)
metrics = split_metric(excel)
fig_name = ['MAE', 'RMSE', 'R2', 'FS', 'Sunny MAE',
            'Sunny RMSE', 'Cloudy MAE', 'Cloudy RMSE',
            'Overcast MAE', 'Overcast RMSE', 'Rainy MAE',
            'Rainy RMSE']
fig_title = ['MAE', 'RMSE', 'R-Square', 'Forecast Skill (RMSE)', 'Sunny MAE',
             'Sunny RMSE', 'Cloudy MAE', 'Cloudy RMSE',
             'Overcast MAE', 'Overcast RMSE', 'Rainy MAE',
             'Rainy RMSE']

result1_title = ['MAE', 'RMSE']
result1_metrics = [metrics[key] for key in ['MAE', 'RMSE']]

result2_title = ['R-Square', 'Forecast Skill (RMSE)']
result2_metrics = [metrics[key] for key in ['R2', 'FS']]

result3_title = ['Sunny MAE', 'Cloudy MAE', 'Overcast MAE', 'Rainy MAE', 'Sunny RMSE', 'Cloudy RMSE', 'Overcast RMSE',
                 'Rainy RMSE']
result3_metrics = [metrics[key] for key in result3_title]

small_size = (10, 14)
plot_pair(result1_title, result1_metrics, "vit/figures/FIG_result1.eps", small_size)
plot_pair(result2_title, result2_metrics, "vit/figures/FIG_result2.eps", small_size)

big_size = (30, 10)
plot_pairs(result3_title, result3_metrics, "vit/figures/FIG_result3.eps", big_size)
