

import numpy as np
import pandas as pd
from sklearn.model_selection import learning_curve
import matplotlib.pyplot as plt

from RTS.svm_read_utils import read_ratio, read_index, read_ratio1


def plot_learning_curve(estimator, title, X, y, ax=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):

    ax.set_title(title)
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel("Training examples")
    ax.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    # 绘制学习曲线
    ax.grid()
    ax.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.2,
                         color="r")
    ax.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.2,
                         color="g")
    ax.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    ax.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    ax.legend(loc="best")
    return plt


def plot_sr(y, x, show_s2=False, z_std=False):
    df = read_ratio(y, x, smooth=3, z_std=z_std,)
    if show_s2:
        df.plot()
    else:
        df['RNDVI'].plot()

    plt.show()


def plot_index(y, x, show_s2=False, **kwargs):
    df = read_index(y, x, smooth=3, **kwargs)
    if show_s2:
        df.plot()
    else:
        df['NDVI'].plot()

    plt.show()

def union_plot(y, x, show_s2=False, RTS_start_year_range=None, **kwargs):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6, 6), sharex=True)
    df = read_index(y, x, **kwargs)
    df_r = read_ratio1(y, x)
    # df_r = read_ratio1(y, x)
    # df_r = pd.concat([df_r['RNDVI'], df_r_1['RNDVI']],  axis=1)
    # df_r.columns = ['R1', 'R2']
    if show_s2:
        df['NDVI'].plot(ax=axes[0], label='NDVI', color='b')
        twin_axes = axes[0].twinx()
        df['SWIR2'].plot(ax=twin_axes, color='y', label='SWIR2')
        df_r.plot(ax=axes[1], linestyle='--', legend=False)
        twin_axes.set_ylabel('SWIR2', color='y')
    else:
        df['NDVI'].plot(ax=axes[0], )
        df_r['RNDVI'].plot(ax=axes[1], linestyle='--', legend=False)

    lines1, labels1 = axes[0].get_legend_handles_labels()
    if show_s2:
        lines2, labels2 = twin_axes.get_legend_handles_labels()
        lines1.extend(lines2)
        labels1.extend(labels2)

    if RTS_start_year_range:
        start, end, prob = RTS_start_year_range
        index = pd.date_range('1986-01-01', '2024-01-01', freq='YE')
        if start == end:
            axes[1].vlines(x=index[start], ymin=0, ymax=1, color='lightgreen', linestyle='--',
                           transform=axes[1].get_xaxis_transform(), label='RTS signal')
        else:
            axes[1].fill_between(index[int(start):int(end+1)], 0, 1, color='lightgreen', alpha=0.5,
                                 transform=axes[1].get_xaxis_transform(),
                            label="RTS signal")
        axes[1].text(0.8, 0.1, f'prob:{prob:.2f}', transform=axes[1].transAxes,)


    axes[0].legend(lines1, labels1)

    axes[1].set_xlabel('time')
    axes[1].set_ylabel('$Z_{R}$')
    axes[0].set_ylabel('NDVI', color='b')

    plt.subplots_adjust(top=0.98, bottom=0.15, left=0.12, right=0.88, )
    axes[1].legend(ncol=3, bbox_to_anchor=(0.5, -0.35), loc='lower center')
    plt.show()



if __name__ == "__main__":
    union_plot(34.32991336, 91.34220247, show_s2=True)