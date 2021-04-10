import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def plot_results(log_dir, save_dir=None, show=False, min_max=False):
    # Get all related csv files
    files = sorted(glob(os.path.join(log_dir, '*.csv')))
    # Don't use already aggregated data if it is present.
    if os.path.join(log_dir, 'aggregated-data.csv') in files:
        files.remove(os.path.join(log_dir, 'aggregated-data.csv'))
    # Load all of the data
    dataframes = [pd.read_csv(filename, skiprows=1, index_col=2) for filename in files]
    # Build up a full set of data.
    full_data = dataframes[0]
    for i, df in enumerate(dataframes[1:]):
        full_data = full_data.append(df)
    # Sort by the time since the experiment began.
    full_data = full_data.sort_index()
    # Form groups of size equal to the number of processes used.
    groups = ((np.arange(len(full_data)) // len(dataframes)) + 1) * len(dataframes)
    # Fix the final index so we dont overstate the number of episodes run.
    if len(full_data) % len(dataframes) != 0:
        groups[-(len(full_data) % len(dataframes)):] = len(full_data)
    # Group the data using mean as an aggregating function.
    grouped_data = full_data.groupby(groups, as_index=True).mean()
    if min_max:
        max_data = full_data.groupby(groups, as_index=True).max()
        min_data = full_data.groupby(groups, as_index=True).min()
    # Add a column for number of steps.
    steps = np.cumsum(grouped_data.l * len(dataframes))
    if len(full_data) % len(dataframes) != 0:
        steps[-1] = full_data.l.sum()
    grouped_data['steps'] = steps
    # Plot the data.
    plt.figure(constrained_layout=True, figsize=(10, 6))
    plt.plot(grouped_data['steps'], grouped_data['r'])
    if min_max:
        plt.plot(grouped_data['steps'], max_data['r'], c='r', alpha=0.5)
        plt.plot(grouped_data['steps'], min_data['r'], c='r', alpha=0.5)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    # Possibly save the data.
    if save_dir and os.path.exists(save_dir):
        plt.savefig(os.path.join(save_dir, save_dir.split('/')[-1] + '_reward.pdf'))
        grouped_data.to_csv(os.path.join(save_dir, 'aggregated-data.csv'))
    # Possibly show the plot immediately.
    if show:
        plt.show()


if __name__ == '__main__':
    # seaborn styling
    sns.set()

    args = argparse.ArgumentParser('Plot Params')
    args.add_argument('--log_dir', type=str, default='~/code/DiffDAC-Experiments/diffdac/diffdac_8x2_adam_ring_60985')
    args.add_argument('--save_dir', type=str, default=None)
    args.add_argument('--show', default=False, action='store_true')
    args.add_argument('--min_max', default=False, action='store_true')
    args = args.parse_args()
    if not args.show:
        args.save_dir = args.save_dir or args.log_dir
    plot_results(args.log_dir, args.save_dir, args.show, args.min_max)
