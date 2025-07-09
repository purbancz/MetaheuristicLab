import os
from multiprocessing import cpu_count

from experiment.retrieve import plot_all_from_pickle, plot_combined_data_from_pickles, kruskal_wallis_with_posthoc, \
    extract_results_to_csv, collect_pickle_files_from_paths
from experiment.runner import run_all_experiments, run_all_experiments_multi

if __name__ == "__main__":
    # try:
    #     slurm_cpus_per_task = os.environ.get('SLURM_CPUS_PER_TASK')
    #     slurm_ntasks = os.environ.get('SLURM_NTASKS')
    #     print(f"Read SLURM_CPUS_PER_TASK: {slurm_cpus_per_task}")
    #     print(f"Read SLURM_NTASKS: {slurm_ntasks}")
    #
    #     slurm_cpus = int(os.environ.get('SLURM_CPUS_PER_TASK', os.environ.get('SLURM_NTASKS', 0)))
    #     if slurm_cpus > 0:
    #         num_workers = slurm_cpus
    #         print(f"Detected Slurm allocation: Setting num_workers = {num_workers}")
    #     else:
    #         num_workers = cpu_count()
    #         print(f"Slurm variables not detected. Defaulting num_workers to cpu_count(): {num_workers}")
    # except Exception as e:
    #     print(f"Could not read Slurm environment variables ({e}). Defaulting num_workers.")
    #     num_workers = cpu_count()
    #
    # # Limit workers if needed (e.g., memory constraints)
    # # num_workers = min(num_workers, 16) # Example limit
    #
    # # run_all_experiments()
    # run_all_experiments_multi(num_parallel_workers=num_workers)

    ###############

    # pkl_files = [
    #     'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_RebelPSO_experiment_data.pkl',
    #     'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_RejectorPSO_experiment_data.pkl',
    # ]

    paths = [
        'experiment_results/dim100_runs50',
        # 'experiment_results/dim500_runs50',
        # 'experiment_results/dim1000_runs50',
        # 'experiment_results/wybrane/100',
        # 'experiment_results/wybrane/500',
        # 'experiment_results/wybrane/1000',
    ]

    pkl_files = collect_pickle_files_from_paths(paths)


    # plot_all_from_pickle(pkl_files[0])
    plot_combined_data_from_pickles(pkl_files)

    # kruskal_wallis_with_posthoc(pkl_files)

    # extract_results_to_csv(pkl_files, output_prefix="all_algorithms_all_problems")
