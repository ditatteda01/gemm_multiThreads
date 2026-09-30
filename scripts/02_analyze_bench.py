import numpy as np
import pandas as pd
import toolkits as tlks
import matplotlib.pyplot as plt
import seaborn as sns


def plot_PCluster(data_list, title_list, ylabel_list, xlabel, wsize, ofilepath=None):
    fig, axes = plt.subplots(
        nrows=len(data_list),
        ncols=1,
        sharex=True,
        figsize=(4 * len(data_list), 7),
        constrained_layout=True
    )

    for idx, data in enumerate(data_list):
        ax = axes[idx]
        title = title_list[idx]
        ylabel = ylabel_list[idx]

        x = [i for i in range(1, len(data) + 1)]
        xticks = [i for i in x if i % wsize == 0]

        ax.plot(x, data)
        ax.set_xticks(xticks)
        ax.tick_params(axis='x', labelbottom=True)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis='x')

    ax.set_xlabel(xlabel)

    if ofilepath is None:
        plt.show()
    else:
        plt.savefig(ofilepath, dpi=300, bbox_inches='tight')
        plt.close()

def plot_heatmap(df, title, ofilepath=None):
    """Plot heat map of MC vs NC per KC.
    
    Args:
        df: Data frame with columns ['mc', 'nc', 'kc', 'score']
    """
    unique_kcs = sorted(df['kc'].unique())
    num_kcs = len(unique_kcs)

    nrows = int(np.ceil(num_kcs / 2))
    ncols = 2

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(6 * ncols, 4 * nrows),
        constrained_layout=True
    )
    axes = axes.flatten()

    if num_kcs == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        if i < num_kcs:
            kc = unique_kcs[i]

            df_sliced = df_metric[df_metric['kc'] == kc]

            # Index=Y, Column=X
            heatmap_matrix = df_sliced.pivot(
                index='nc', columns='mc', values='score'
            ).sort_index(ascending=False)

            sns.heatmap(
                heatmap_matrix,
                ax=ax,
                annot=True,
                cmap='GnBu',
                fmt='.3f',
                vmin=0.0,
                vmax=1.0,
                cbar=False,
            )

            ax.set_title(f"KC = {kc}")
            ax.patch.set_edgecolor('black')
            ax.patch.set_linewidth(2)

        else:
            ax.set_visible(False)

    mappable = axes[0].collections[0]

    fig.suptitle(title)
    fig.colorbar(mappable, ax=axes, shrink=0.8, aspect=30, label='Score')

    if ofilepath is None:
        plt.show()
    else:
        plt.savefig(ofilepath, dpi=300, bbox_inches='tight')
        plt.close()
    

if __name__ == '__main__':
    bench_fpath = tlks.mk_fpath('data', 'integrated_bench.csv')
    df = pd.read_csv(bench_fpath)

    # Fill NaN with interpolation of its surrounding
    df['P-Cluster Frequency'] = df['P-Cluster Frequency'].interpolate()
    pclusters = df['P-Cluster Frequency']

    # Calculate rolling standard-deviation/median in forward-looking
    wsize = 200
    reversed_pclusters = df['P-Cluster Frequency'].iloc[::-1]
    reversed_rstdev = reversed_pclusters.rolling(window=wsize).std()
    reversed_rmedian = reversed_pclusters.rolling(window=wsize+1).median()
    rstdev = reversed_rstdev.iloc[::-1]
    rmedian = reversed_rmedian.iloc[::-1]

    plot_PCluster(
        [pclusters, rstdev, rmedian],
        ["P-Cluster Frequency",
         f"Rolling Standard Deviation (WinSize={wsize})",
         f"Rolling Median (WinSize={wsize})"
        ],
        ["HMz"] * 3,
        "Position",
        wsize,
        tlks.mk_fpath("png", "pcluster_freq.png")
    )

    # Find stable point from rolling median
    slopes = rmedian.diff().abs()
    stable_mask = slopes.rolling(window=wsize).mean() < 0.3
    stable_pos = stable_mask[stable_mask].index[0]
    print(f"P-Cluster frequency stable position: {stable_pos}")

    df_ss = df.iloc[stable_pos:].sort_values(by='time_ms', ascending=True)

    n = len(df_ss)
    if n == 0:
        raise ValueError("df_ss is empty - nothing to aggregate")
    
    top_cnt = max(1, round(n * 0.1))
    df_ss = df_ss.head(top_cnt).reset_index(drop=True)
    print(f"Total data (Top 10%): {top_cnt}")
    print(f"Time range from {df_ss['time_ms'].iat[0]} to {df_ss['time_ms'].iat[-1]} ms")

    # Change the weight ratio
    W_RANK, W_FREQ = 0.7, 0.3

    df_ss['score'] = np.exp(-np.arange(top_cnt) / top_cnt)

    mnk_grp = df_ss.groupby(['mc', 'nc', 'kc'], observed=True)
    agg = mnk_grp.agg(
        # new_column_name=('column_to_apply_to', 'function_name') 
        rank_mass=('score', 'sum'),
        freq=('score', 'size')
    ).reset_index()

    # Maximum normalization is for determining
    # 'how close is this config to the most-weighted config (best one)?'
    agg['rank_norm'] = agg['rank_mass'] / agg['rank_mass'].max()
    agg['freq_norm'] = np.log1p(agg['freq']) / np.log1p(agg['freq'].max())
    agg['score'] = W_RANK * agg['rank_norm'] + W_FREQ * agg['freq_norm']

    # Construct configure table for (mc, nc, kc) pair with score repectively
    grid = pd.MultiIndex.from_product(
        [df_ss[c].drop_duplicates().sort_values(ascending=True) for c in ['mc', 'nc', 'kc']],
        names=['mc', 'nc', 'kc']
    ).to_frame(index=False)

    df_metric = grid.merge(
        agg[['mc', 'nc', 'kc', 'score']],
        on=['mc', 'nc', 'kc'],
        how='left'
    ).fillna(0.0)

    plot_heatmap(df_metric, "Config. Score", tlks.mk_fpath("png", "config_score.png"))
    