import os
from multiprocessing import cpu_count

from experiment.runner import run_all_experiments_multi
from experiment.setup import setup_experiment

# ... other imports ...
# from experiment.runner import run_all_experiments_multi # Ensure this is imported
# from experiment.setup import setup_experiment # Ensure this is imported

if __name__ == "__main__":
    # --- Determine number of parallel workers ---
    try:
        # Read both variables
        slurm_cpus_per_task_str = os.environ.get('SLURM_CPUS_PER_TASK')
        slurm_ntasks_str = os.environ.get('SLURM_NTASKS')
        print(f"Read SLURM_CPUS_PER_TASK: {slurm_cpus_per_task_str}")
        print(f"Read SLURM_NTASKS: {slurm_ntasks_str}")

        slurm_cpus_per_task = int(slurm_cpus_per_task_str) if slurm_cpus_per_task_str else 0
        slurm_ntasks = int(slurm_ntasks_str) if slurm_ntasks_str else 0

        # **Revised Logic:** Prioritize CPUS_PER_TASK allocated to *this* process
        if slurm_cpus_per_task > 1:
            # If Slurm gave this task multiple CPUs, use that number
            num_workers = slurm_cpus_per_task
            print(f"Detected Slurm multi-CPU task: Setting num_workers = {num_workers} (from CPUS_PER_TASK)")
        elif slurm_ntasks > 1 and slurm_cpus_per_task <= 1:
             # This case *shouldn't* happen with the recommended sbatch script (--ntasks=1)
             # If it did, it means we are one of many single-core tasks. Use 1 worker.
             num_workers = 1
             print(f"Detected Slurm multi-task single-CPU allocation: Setting num_workers = 1")
        else:
            # Not in Slurm or single task/single core Slurm job
            local_cores = cpu_count()
            num_workers = local_cores
            print(f"Slurm vars not detected or indicate single core. Defaulting num_workers to cpu_count(): {num_workers}")

    except Exception as e:
        print(f"Could not read/parse Slurm environment variables ({e}). Defaulting num_workers.")
        num_workers = cpu_count()

    # Optional: Limit workers if needed
    # num_workers = min(num_workers, 16)

    # --- Load Setup ---
    # Ensure this defines algorithms_factory correctly using top-level functions
    (algorithms_factory, group_of_algorithms, problems, no_of_runs, number_of_variables,
     solutions_size, max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()

    # --- Run the experiments using YOUR runner function ---
    # Make sure run_all_experiments_multi has the Pool size fix: min(available_workers, no_of_runs)
    run_all_experiments_multi(num_parallel_workers=num_workers)