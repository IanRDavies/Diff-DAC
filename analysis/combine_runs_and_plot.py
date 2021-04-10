import argparse
import os
from glob import glob
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import sem
import matplotlib.pyplot as plt
import tikzplotlib


MAX_STEPS = 10_000_000


def aggregate_one_run_results(log_dir, save=True, plot=False):
    """
    Aggregate data across processes and agents for a single DiffDAC or GALA run.
    This yields a dataframe where metrics are reported at the overall system level in the run.

    If this data has already been aggregated and saved by some other plotting function then this
    data is simply loaded and returned.
    """
    # Get all related csv files
    files = sorted(glob(os.path.join(log_dir, '*.csv')))
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
    # Fix the final index so we don't overstate the number of episodes run.
    if len(full_data) % len(dataframes) != 0:
        groups[-(len(full_data) % len(dataframes)):] = len(full_data)
    # Group the data using mean as an aggregating function.
    grouped_data = full_data.groupby(groups, as_index=True).mean()
    # Add a column for number of steps.
    steps = np.cumsum(grouped_data.l * len(dataframes))
    if len(full_data) % len(dataframes) != 0:
        steps[-1] = full_data.l.sum()
    grouped_data['steps'] = steps
    # Possibly save the data.
    if save:
        grouped_data.to_csv(os.path.join(log_dir, 'aggregated-data.csv'))
    # Create the individual run plot as in plot_single_run.py if needed. Only plotting steps and not
    # episodes for simplicity.
    if plot:
        plt.figure(figsize=(10, 6))
        plt.plot(grouped_data['steps'], grouped_data['r'])
        plt.xlabel('Steps')
        plt.ylabel('Reward')
        plt.savefig(os.path.join(log_dir, log_dir.split('/')[-1] + '_reward_plot.pdf'))
    return grouped_data


def process_agg_df(df, step_window_size=10000):
    """
    Process the aggregated data from each run ready for aggregation across runs.
    This is done by averaging over non-overlapping windows of step_window_size and resetting the
    index to be the upper limit of each window. This is required as different runs will have
    different steps at each recording point as a record is made each episode and episodes may vary
    in length.
    """
    _df = df.copy()
    _df = _df.drop('l', axis=1)
    # Perform the grouping.
    groups = (_df.steps.array // step_window_size).astype(int) * step_window_size
    # An alternative to the mean could be used here.
    grouped = _df.groupby(groups, as_index=True).mean()
    grouped["steps"] = np.unique(groups)
    return grouped[grouped['steps'] <= MAX_STEPS].set_index("steps")


def combine_and_plot(base_directory_form, a2c_directory_form, specialised_directory_form, save_dir, step_window_size=10000, plot_individual=False):
    """
    Collects data across runs with different seeds and plots an average.
    This function assumes that logs are saved in directories differing only in the seed value which
    is part of the directory name.
    """
    sns.set()
    # Set up a place to save the plot drawing on the directory name for naming convention.
    save_location = os.path.join(save_dir, base_directory_form.split('/')[-2]
                                 .replace('*', 'combined_')
                                 + '.pdf')
    # Work out where the data is and aggregate it.
    base_directories = sorted(glob(base_directory_form))
    base_dataframes = [process_agg_df(aggregate_one_run_results(dd, plot=plot_individual),
                                      step_window_size) for dd in base_directories]
    base_combined = pd.concat(base_dataframes, axis=1, join='inner')
    plot_a2c = False
    plot_s = False
    if a2c_directory_form is not None:
        plot_a2c = True
        a2c_directories = sorted(glob(a2c_directory_form))
        a2c_dataframes = [process_agg_df(aggregate_one_run_results(dd, plot=plot_individual),
                                          step_window_size) for dd in a2c_directories]
        a2c_combined = pd.concat(a2c_dataframes, axis=1, join='inner')
        a2c_average = a2c_combined.mean(axis=1)
        a2c_se = sem(a2c_combined.values, axis=1)

    if specialised_directory_form is not None:
        plot_s = True
        s_directories = sorted(glob(specialised_directory_form))
        s_dataframes = [process_agg_df(aggregate_one_run_results(dd, plot=plot_individual),
                                          step_window_size) for dd in s_directories]
        s_combined = pd.concat(s_dataframes, axis=1, join='inner')
        s_average = s_combined.mean(axis=1)
        s_se = sem(s_combined.values, axis=1)

    # Average over runs.
    average = base_combined.mean(axis=1)
    se = sem(base_combined.values, axis=1)
    # Plot the results and save the figure to disk.
    plt.figure(figsize=(10, 6))
    plt.fill_between(average.index, average-se, average+se, color='b', alpha=0.2)
    plt.plot(average, color='b', label='DiffDAC-A2C')
    if plot_a2c:
        plt.fill_between(a2c_average.index, a2c_average - a2c_se, a2c_average + a2c_se, color='r', alpha=0.2)
        plt.plot(a2c_average, color='r', label='Centralised A2C')
    if plot_s:
        plt.fill_between(s_average.index, s_average - s_se, s_average + s_se, color='g', alpha=0.2)
        plt.plot(s_average, color='g', label='Specialised Agents')

    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.legend()
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)
    plt.savefig(save_location)
    tikzplotlib.save(save_location.replace('.pdf', '.tex'), strict=True)
    print(f'Saved Aggregated Plot at {save_location}')


if __name__ == '__main__':
    args = argparse.ArgumentParser('Plot Params')
    args.add_argument('--dir_base', type=str, help="Base directory path with * in place of seed.",
                      default="/home/ian.davies/Documents/code/DiffDAC-Experiments/to_upload/diffdac/RandomExtremeAcrobot/diffdac_25x1_ExtremeAcrobot_ring_5spu_lr0.005_*_save_dict")
    args.add_argument('--a2c_dir_base', type=str, help="Directory path for secondary plot with * in place of seed.",
                      default="/home/ian.davies/Documents/code/DiffDAC-Experiments/to_upload/A2C (Centralised)/RandomExtremeAcrobot/a2c_1x25_ExtremeAcrobot_5spu_lr0.002_*_save_dict")
    args.add_argument('--specialised_dir_base', type=str,
                      help="Directory path for secondary plot with * in place of seed.",
                      default="/home/ian.davies/Documents/code/DiffDAC-Experiments/to_upload/specialised (A2C all processes have the same env seed)/master_seed_1/diffdac_specialisingagents_RandomExtremeAcrobot_*")

    args.add_argument('--save_dir', type=str, default='/home/ian.davies/Documents/code/DiffDAC-Experiments/final/plots',
                      help="Directory in which to save aggregated plot.")
    args.add_argument('--step_window_size', type=int, default=15000,
                      help='Number of time steps to average over in each run before averaging over'
                           'runs.')
    args.add_argument('--plot_individual', default=True, action='store_true',
                      help='Save plots for each individual run.')
    args = args.parse_args()
    combine_and_plot(args.dir_base, args.a2c_dir_base, args.specialised_dir_base, args.save_dir, args.step_window_size, args.plot_individual)
