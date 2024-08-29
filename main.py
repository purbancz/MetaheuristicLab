from experiment.retrieve import plot_all_from_pickle, plot_combined_data_from_pickles
from experiment.runner import run_all_experiments


if __name__ == "__main__":
    run_all_experiments()

    # pkl_files = [
    #     'experiment_results/dim30_runs3/20240829_210621_experiment_data.pkl',
    #     'experiment_results/dim30_runs7/20240829_212510_experiment_data.pkl'
    # ]
    #
    # # plot_all_from_pickle(pkl_files[0])
    # plot_combined_data_from_pickles(pkl_files)
