import os
from multiprocessing import cpu_count

from experiment.retrieve import plot_all_from_pickle, plot_combined_data_from_pickles, kruskal_wallis_with_posthoc, \
    extract_results_to_csv
from experiment.runner import run_all_experiments, run_all_experiments_multi

if __name__ == "__main__":
    # try:
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

    pkl_files = [
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_RebelPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_RejectorPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_RebelRejectorPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_RRAPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_ContrarianPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_DefeatistPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_ContrarianDefeatistPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_CDAPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_EschewerPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_EscapistPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_EschewerEscapistPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_EEAPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_ReverseLearningPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_PSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_ReverseLearningGlobalAttractorPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_ReverseLearningPersonalAttractorPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_CombinedLearningPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_AnarchicPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_AmnesiacPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_WandererPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_NoisyPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_PerturbationPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_PartialResetPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_CollectiveResetPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_FRAPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_HybridFullDisjointPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_HybridPartialDisjointPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_HybridAdditivePSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_CAPSO_experiment_data.pkl',
        'experiment_results/dim100_runs5/Rastrigin_dim100_runs5_IAPSO_experiment_data.pkl',
    ]


    # # plot_all_from_pickle(pkl_files[0])
    # plot_combined_data_from_pickles(pkl_files)
    #
    # kruskal_wallis_with_posthoc(pkl_files)
    #
    extract_results_to_csv(pkl_files)
