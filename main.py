from experiment.retrieve import plot_all_from_pickle, plot_combined_data_from_pickles, kruskal_wallis_with_posthoc
from experiment.runner import run_all_experiments

if __name__ == "__main__":
    run_all_experiments()

    # pkl_files = [
    #     'experiment_results/dim100_runs50/20250412_224013_experiment_data.pkl',
    #     # 'experiment_results/dim50_runs10/20240830_094840_experiment_data.pkl',
    #     # 'experiment_results/dim100_runs10/20240830_003728_experiment_data.pkl',
    #     # 'experiment_results/dim500_runs10/20240830_055451_experiment_data.pkl',
    #     # 'experiment_results/dim1000_runs10/20240831_020336_experiment_data.pkl',
    # ]
    #
    # # plot_all_from_pickle(pkl_files[0])
    # plot_combined_data_from_pickles(pkl_files)

    # kruskal_wallis_with_posthoc(pkl_files)
