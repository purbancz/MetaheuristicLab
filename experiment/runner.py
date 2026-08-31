import copy
import csv
import random
import socket
import subprocess
import sys
from datetime import datetime

import h5py
import humanize
import numpy as np

import time
from multiprocessing import Pool, cpu_count
import traceback

from experiment.globals import BASE_SEED
from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box, \
    plot_final_raincloud, plot_final_petit_prince, plot_results_with_annotations
from experiment.setup import setup_experiment, make_dir
from observer.fitness_observer import FitnessObserver

# Configuration
(algorithms, group_of_algorithms, problems, no_of_runs, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()


def _git_commit():
    try:
        return subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return 'unknown'


def _write_manifest(h5file, n_vars, n_runs):
    h5file.attrs['git_commit'] = _git_commit()
    h5file.attrs['timestamp'] = datetime.now().isoformat()
    h5file.attrs['python_version'] = sys.version
    h5file.attrs['numpy_version'] = np.__version__
    h5file.attrs['hostname'] = socket.gethostname()
    h5file.attrs['n_vars'] = n_vars
    h5file.attrs['no_of_runs'] = n_runs


def _h5_path(dimensions_dir, problem_name, algo_name, n_vars, n_runs):
    safe = problem_name.replace(' ', '_').replace('-', '_')
    return f'{dimensions_dir}/{safe}_dim{n_vars}_runs{n_runs}_{algo_name}.h5'


def _write_h5_algo_result(path, problem_name, algo_name, result, n_vars, n_runs):
    with h5py.File(path, 'w') as h5:
        _write_manifest(h5, n_vars, n_runs)
        h5.attrs['problem_name'] = problem_name
        h5.attrs['algo_name'] = algo_name
        problem_grp = h5.require_group(problem_name)
        problem_grp.attrs['n_vars'] = n_vars
        grp = problem_grp.require_group(algo_name)
        grp.create_dataset('fitness_curves', data=result['data'].astype(np.float64), compression='gzip')
        grp.create_dataset('final_fitness', data=result['final_fitness'].astype(np.float64))
        grp.create_dataset('seeds', data=np.array(result['seeds'], dtype=np.int64))
        grp.create_dataset('run_times', data=result['run_times'].astype(np.float64))
        grp.attrs['avg_fitness'] = result['avg_fitness']
        grp.attrs['std_dev'] = result['std_dev']
        grp.attrs['avg_time'] = result['avg_time']


def run_all_experiments():
    run_seeds = generate_run_seeds(no_of_runs)
    dimensions_dir = results_dir + f'/dim{number_of_variables}_runs{no_of_runs}'
    make_dir(dimensions_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'{dimensions_dir}/{timestamp_str}_results.csv'

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness',
                         'Standard deviation', 'Average Computing Time (s)'])

        for problem in problems:
            problem = copy.deepcopy(problem)
            problem_data = {'problem': problem.name(), 'n_vars': problem.number_of_variables(), 'results': {}}

            for name, algorithm in algorithms.items():
                result = run_experiment(algorithm, problem, no_of_runs, frequency, run_seeds=run_seeds)
                problem_data['results'][name] = result

                print(f"Algorithm: {name}, Problem: {problem.name()}, Variables: {problem.number_of_variables()}, "
                      f"Runs: {no_of_runs}, Average Final Fitness: {result['avg_fitness']}, "
                      f"Standard deviation: {result['std_dev']}, Average Time: {result['avg_time']}, "
                      f"Finished at: {datetime.now()}")

                writer.writerow([name, problem.name(), problem.number_of_variables(), no_of_runs,
                                 result['avg_fitness'], result['std_dev'], result['avg_time']])

                h5_path = _h5_path(dimensions_dir, problem.name(), name, problem.number_of_variables(), no_of_runs)
                _write_h5_algo_result(h5_path, problem.name(), name, result, problem.number_of_variables(), no_of_runs)

            plot_results(problem_data['results'], problem, dimensions_dir, max_evaluations, no_of_runs,
                         algorithm_colors)
            plot_results_with_annotations(problem_data['results'], problem, dimensions_dir, max_evaluations, no_of_runs,
                                          algorithm_colors)
            plot_results_with_std(problem_data['results'], problem, dimensions_dir, max_evaluations,
                                  no_of_runs, algorithm_colors)
            # plot_box_at_intervals(problem_data['results'], problem, max_evaluations=max_evaluations,
            #                       no_of_runs=no_of_runs, algorithms_to_compare=algorithms.keys(),
            #                       results_dir=dimensions_dir,
            #                       algorithm_colors=algorithm_colors)
            #
            # for algorithm in algorithms.keys():
            #     plot_box_at_intervals(problem_data['results'], problem, max_evaluations=max_evaluations,
            #                           no_of_runs=no_of_runs, algorithms_to_compare=[algorithm],
            #                           results_dir=dimensions_dir,
            #                           algorithm_colors=algorithm_colors)
            # # for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
            # #     plot_box_at_intervals(problem_data['results'], problem, max_evaluations=max_evaluations,
            # #                           no_of_runs=no_of_runs, algorithms_to_compare=[algorithm] + ['GA', 'PSO'],
            # #                           results_dir=dimensions_dir,
            # #                           algorithm_colors=algorithm_colors)
            # plot_final_box(problem_data['results'], problem, dimensions_dir, algorithm_colors)
            # plot_final_raincloud(problem_data['results'], problem, dimensions_dir, algorithm_colors)
            # plot_final_raincloud(problem_data['results'], problem, dimensions_dir, algorithm_colors,
            #                      adaptive_height=True)
            # plot_final_petit_prince(problem_data['results'], problem, dimensions_dir, algorithm_colors)
            # plot_final_petit_prince(problem_data['results'], problem, dimensions_dir, algorithm_colors,
            #                         adaptive_width=True)

            for group_name, algorithm_list in group_of_algorithms.items():
                filtered_results = {algo: problem_data['results'][algo] for algo in algorithm_list if
                                    algo in problem_data['results']}

                if not filtered_results or set(filtered_results.keys()) == {'PSO'}:
                    print(f"Skipping {group_name}, no valid algorithms found (only 'PSO' or empty).")
                    continue

                plot_results(filtered_results, problem, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors,
                             group_name)
                plot_results_with_annotations(filtered_results, problem, dimensions_dir, max_evaluations, no_of_runs,
                                              algorithm_colors, group_name)
                plot_results_with_std(filtered_results, problem, dimensions_dir, max_evaluations, no_of_runs,
                                      algorithm_colors, group_name)
                # plot_box_at_intervals(filtered_results, problem, max_evaluations=max_evaluations, no_of_runs=no_of_runs,
                #                       algorithms_to_compare=list(filtered_results.keys()), results_dir=dimensions_dir,
                #                       algorithm_colors=algorithm_colors, group_name=group_name)
                plot_final_box(filtered_results, problem, dimensions_dir, algorithm_colors, group_name)
                plot_final_raincloud(filtered_results, problem, dimensions_dir, algorithm_colors, group_name)
                plot_final_raincloud(filtered_results, problem, dimensions_dir, algorithm_colors, group_name,
                                     adaptive_height=True)
                plot_final_petit_prince(filtered_results, problem, dimensions_dir, algorithm_colors)
                plot_final_petit_prince(filtered_results, problem, dimensions_dir, algorithm_colors,
                                        adaptive_width=True)

    print(f"Experiment data saved to {dimensions_dir}")


def run_experiment(algorithm_factory, problem, runs, interval, run_seeds=None):
    if run_seeds is None:
        run_seeds = generate_run_seeds(runs)

    if len(run_seeds) != runs:
        raise ValueError("Number of run seeds must match number of runs.")

    all_fitness_data = []
    total_times = []

    for seed in run_seeds:
        set_run_seed(seed)
        problem_instance = copy.deepcopy(problem)
        set_problem_run_seed(problem_instance, seed)
        algorithm = algorithm_factory(problem_instance)

        observer = FitnessObserver(interval=interval)
        algorithm.observable.register(observer)

        algorithm.run()

        total_time = algorithm.observable_data()['COMPUTING_TIME']
        last_fitness = algorithm.result().objectives[0]
        filled_fitness = (observer.best_fitness_history +
                          [last_fitness] * (max_evaluations // interval - len(observer.best_fitness_history)))
        all_fitness_data.append(filled_fitness)
        total_times.append(total_time)

    fitness_array = np.array(all_fitness_data)
    run_times = np.array(total_times)
    final_fitness = fitness_array[:, -1]
    valid_final = final_fitness[np.isfinite(final_fitness)]

    return {
        'data': fitness_array,
        'final_fitness': final_fitness,
        'seeds': list(run_seeds),
        'run_times': run_times,
        'avg_fitness': float(np.mean(valid_final)) if valid_final.size > 0 else float('inf'),
        'std_dev': float(np.std(valid_final)) if valid_final.size > 0 else float('nan'),
        'avg_time': float(np.mean(run_times)),
    }


### Multiprocessing
def run_single_instance(args):
    """
    Executes a single run of an algorithm on a problem.
    Designed to be called by multiprocessing.Pool.map.
    """
    algo_name, problem_instance_copy, algo_lambda, run_id, seed, max_evals, freq = args
    # print(f"  Starting Run {run_id} for {algo_name} on {problem_instance_copy.name()}...")

    try:
        set_run_seed(seed)
        set_problem_run_seed(problem_instance_copy, seed)

        algorithm = algo_lambda(problem_instance_copy)
        observer = FitnessObserver(interval=freq)
        algorithm.observable.register(observer)

        start_time = time.time()
        algorithm.run()
        end_time = time.time()

        total_time = end_time - start_time
        result = algorithm.result()

        if result is None or not hasattr(result, 'objectives') or not result.objectives:
            print(f"    Run {run_id} Warning: No valid result/objectives.")
            final_fitness = float('inf')
            filled_fitness = [float('inf')] * (max_evals // freq)
        else:
            final_fitness = result.objectives[0]
            best_fitness_history = observer.best_fitness_history
            expected_len = max_evals // freq
            filled_fitness = best_fitness_history + [final_fitness] * (expected_len - len(best_fitness_history))
            filled_fitness = filled_fitness[:expected_len]

        return {'fitness_history': filled_fitness, 'time': total_time, 'final_fitness': final_fitness, 'seed': seed}

    except Exception as e:
        print(f"    Run {run_id} ERROR for {algo_name} on {problem_instance_copy.name()}: {e}")
        traceback.print_exc()
        return {'fitness_history': [float('inf')] * (max_evals // freq), 'time': 0, 'final_fitness': float('inf'),
                'seed': seed}


def run_all_experiments_multi(num_parallel_workers=None):
    """
    Runs all algorithm-problem combinations, parallelizing the no_of_runs loop.
    """
    run_seeds = generate_run_seeds(no_of_runs)

    if num_parallel_workers is None or num_parallel_workers <= 0:
        num_parallel_workers = cpu_count()
        print(f"Number of parallel workers not specified, defaulting to {num_parallel_workers}")
    else:
        num_parallel_workers = min(num_parallel_workers, cpu_count())
        print(f"Using {num_parallel_workers} parallel workers for runs.")

    dimensions_dir = results_dir + f'/dim{number_of_variables}_runs{no_of_runs}'
    make_dir(dimensions_dir)

    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f'{dimensions_dir}/{timestamp_str}_results.csv'

    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness',
                         'Standard deviation', 'Average Computing Time (s)'])

        for problem in problems:
            try:
                problem_for_runs = copy.deepcopy(problem)
            except Exception as e:
                print(f"Warning: Could not deepcopy problem {problem.name()}. Using original. Error: {e}")
                problem_for_runs = problem

            problem_name = problem_for_runs.name() if hasattr(problem_for_runs, 'name') else problem_for_runs.__class__.__name__
            problem_n_vars = problem_for_runs.number_of_variables()
            problem_data = {'problem': problem_name, 'n_vars': problem_n_vars, 'results': {}}

            for algo_name, algo_lambda in algorithms.items():
                run_args = [(algo_name, copy.deepcopy(problem_for_runs), algo_lambda, run_id + 1, run_seeds[run_id],
                             max_evaluations, frequency)
                            for run_id in range(no_of_runs)]

                with Pool(processes=num_parallel_workers) as pool:
                    run_results = pool.map(run_single_instance, run_args)

                all_fitness_data_list = [res['fitness_history'] for res in run_results if res]
                total_times_list = [res['time'] for res in run_results if res]
                final_fitness_list = [res['final_fitness'] for res in run_results if res]

                if not all_fitness_data_list:
                    print(f"  ERROR: All runs failed for {algo_name} on {problem_name}.")
                    avg_fitness = float('inf')
                    std_dev = float('nan')
                    avg_time = 0.0
                    aggregated_fitness_array = np.full((1, max_evaluations // frequency), float('inf'))
                    final_fitness_array = np.array([float('inf')])
                    run_times_array = np.array([0.0])
                else:
                    valid_final = [f for f in final_fitness_list if np.isfinite(f)]
                    avg_fitness = float(np.mean(valid_final)) if valid_final else float('inf')
                    std_dev = float(np.std(valid_final)) if valid_final else float('nan')
                    avg_time = float(np.mean(total_times_list)) if total_times_list else 0.0
                    sum_time = float(np.sum(total_times_list)) if total_times_list else 0.0
                    humanized_duration = humanize.naturaldelta(sum_time)
                    aggregated_fitness_array = np.array(all_fitness_data_list)
                    final_fitness_array = np.array(final_fitness_list)
                    run_times_array = np.array(total_times_list)

                result = {
                    'data': aggregated_fitness_array,
                    'final_fitness': final_fitness_array,
                    'seeds': list(run_seeds),
                    'run_times': run_times_array,
                    'avg_fitness': avg_fitness,
                    'std_dev': std_dev,
                    'avg_time': avg_time,
                }
                problem_data['results'][algo_name] = result

                if all_fitness_data_list:
                    print(f"Aggregated: Algorithm: {algo_name}, Problem: {problem_name}, "
                          f"Avg Final Fitness: {avg_fitness:.4f}, Std Dev: {std_dev:.4f}, "
                          f"Avg single run duration: {avg_time:.2f}s, "
                          f"Duration: {humanized_duration}, Finished at: {datetime.now()}")
                writer.writerow([algo_name, problem_name, problem_n_vars, no_of_runs, avg_fitness,
                                 std_dev, avg_time])
                file.flush()

                h5_path = _h5_path(dimensions_dir, problem_name, algo_name, problem_n_vars, no_of_runs)
                _write_h5_algo_result(h5_path, problem_name, algo_name, result, problem_n_vars, no_of_runs)

            print(f"--- Finished all algorithms for Problem: {problem_name} ---")

            # try:
            #     print(f"  Generating plots for {problem_name}...")
            #     plot_results(problem_data['results'], problem_for_runs, dimensions_dir, max_evaluations,
            #                  no_of_runs, algorithm_colors)
            #     for group_name, algorithm_list in group_of_algorithms.items():
            #         filtered_results = {algo: problem_data['results'][algo] for algo in algorithm_list
            #                             if algo in problem_data['results']}
            #         if len(filtered_results) > 1 or (len(filtered_results) == 1 and 'PSO' not in filtered_results):
            #             print(f"    Generating plots for group: {group_name}")
            #             plot_results(filtered_results, problem_for_runs, dimensions_dir, max_evaluations,
            #                          no_of_runs, algorithm_colors, group_name)
            # except Exception as plot_err:
            #     print(f"  ERROR generating plots for {problem_name}: {plot_err}")

    print(f"\n===== All Problems Processed =====")
    print(f"Experiment data saved to {dimensions_dir}")


def generate_run_seeds(runs: int, base_seed: int = BASE_SEED) -> list[int]:
    return [base_seed + run_index for run_index in range(runs)]

def set_run_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def set_problem_run_seed(problem, seed: int) -> None:
    if hasattr(problem, "set_seed"):
        problem.set_seed(seed)
