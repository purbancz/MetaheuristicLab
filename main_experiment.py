import os
from multiprocessing import cpu_count

from experiment.runner import run_all_experiments_multi, run_all_experiments
from experiment.setup import setup_experiment


if __name__ == "__main__":
    try:
        slurm_cpus_per_task_str = os.environ.get('SLURM_CPUS_PER_TASK')
        slurm_ntasks_str = os.environ.get('SLURM_NTASKS')
        print(f"Read SLURM_CPUS_PER_TASK: {slurm_cpus_per_task_str}")
        print(f"Read SLURM_NTASKS: {slurm_ntasks_str}")

        slurm_cpus_per_task = int(slurm_cpus_per_task_str) if slurm_cpus_per_task_str else 0
        slurm_ntasks = int(slurm_ntasks_str) if slurm_ntasks_str else 0

        if slurm_cpus_per_task > 1:
            num_workers = slurm_cpus_per_task
            print(f"Detected Slurm multi-CPU task: Setting num_workers = {num_workers} (from CPUS_PER_TASK)")
        elif slurm_ntasks > 1 and slurm_cpus_per_task <= 1:
             num_workers = 1
             print(f"Detected Slurm multi-task single-CPU allocation: Setting num_workers = 1")
        else:
            local_cores = cpu_count()
            num_workers = local_cores
            print(f"Slurm vars not detected or indicate single core. Defaulting num_workers to cpu_count(): {num_workers}")

    except Exception as e:
        print(f"Could not read/parse Slurm environment variables ({e}). Defaulting num_workers.")
        num_workers = cpu_count()

    # Limit workers if needed
    # num_workers = min(num_workers, 16)

    # --- load setup? ---
    (algorithms_factory, group_of_algorithms, problems, no_of_runs, number_of_variables,
     solutions_size, max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()

    run_all_experiments_multi(num_parallel_workers=num_workers)
    # run_all_experiments()