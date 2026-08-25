import copy
import csv
import pickle
from datetime import datetime

import humanize
import numpy as np

import time
from multiprocessing import Pool, cpu_count
import traceback



from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box, \
    plot_final_raincloud, plot_final_petit_prince, plot_results_with_annotations
from experiment.setup import setup_experiment, make_dir
from observer.fitness_observer import FitnessObserver

# Configuration
(algorithms, group_of_algorithms, problems, no_of_runs, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()


def run_all_experiments():
    dimensions_dir = results_dir + f'/dim{number_of_variables}_runs{no_of_runs}'
    make_dir(dimensions_dir)
    csv_filename = f'{dimensions_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_results.csv'
    with (open(csv_filename, mode='w', newline='') as file):
        writer = csv.writer(file)
        writer.writerow(['Algorithm', 'Problem', 'Variables', 'Runs', 'Average Final Fitness',
                         'Standard deviation', 'Average Computing Time (s)'])

        all_data = []
        for problem in problems:
            problem = copy.deepcopy(problem)
            safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
            problem_data = {'problem': problem.name(), 'n_vars': problem.number_of_variables(), 'results': {}}
            for name, algorithm in algorithms.items():
                fitness_data, avg_fitness, std_dev, avg_time = run_experiment(algorithm, no_of_runs, frequency)
                problem_data['results'][name] = {'data': fitness_data, 'avg_fitness': avg_fitness, 'std_dev': std_dev,
                                                 'avg_time': avg_time}

                print(f"Algorithm: {name}, Problem: {problem.name()}, Variables: {problem.number_of_variables()}, "
                      f"Runs: {no_of_runs}, Average Final Fitness: {avg_fitness}, "
                      f"Standard deviation: {std_dev}, Average Time: {avg_time}, Finished at: {datetime.now()}")

                writer.writerow([name, problem.name(), problem.number_of_variables(), no_of_runs, avg_fitness,
                                 std_dev, avg_time])

                with open(
                        f'{dimensions_dir}/{safe_problem_name}_dim{number_of_variables}_runs{no_of_runs}_{name}_experiment_data.pkl',
                        'wb') as f:
                    pickle.dump(problem_data, f)

            all_data.append(problem_data)

            # plot results
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

            with open(
                    f'{dimensions_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{safe_problem_name}_dim{number_of_variables}_runs{no_of_runs}_all_algs_experiment_data.pkl',
                    'wb') as f:
                pickle.dump(all_data, f)

    with open(f'{dimensions_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_all_dim{number_of_variables}_runs{no_of_runs}_experiment_data.pkl', 'wb') as f:
        pickle.dump(all_data, f)


def run_experiment(algorithm_factory, problem, runs, interval):
    all_fitness_data = []
    total_times = []

    for _ in range(runs):
        problem_instance = copy.deepcopy(problem)
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

    average_final_fitness = np.mean([data[-1] for data in all_fitness_data])
    standard_deviation = np.std([data[-1] for data in all_fitness_data])
    average_time = np.mean(total_times)

    return np.array(all_fitness_data), average_final_fitness, standard_deviation, average_time

### Mulitprocessing
def run_single_instance(args):
    """
    Executes a single run of an algorithm on a problem.
    Designed to be called by multiprocessing.Pool.map.
    """
    algo_name, problem_instance_copy, algo_lambda, run_id, max_evals, freq = args
    # print(f"  Starting Run {run_id} for {algo_name} on {problem_instance_copy.name()}...")

    # Instantiate algorithm and observer *within the worker*
    try:
        algorithm = algo_lambda(problem_instance_copy) # Call the factory lambda
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
             # Fill fitness history if shorter than expected
            best_fitness_history = observer.best_fitness_history
            expected_len = max_evals // freq
            filled_fitness = best_fitness_history + [final_fitness] * (expected_len - len(best_fitness_history))
            # Ensure correct length if it somehow exceeds
            filled_fitness = filled_fitness[:expected_len]


        # print(f"  Finished Run {run_id} for {algo_name}. Final Fitness: {final_fitness:.4f}, Time: {total_time:.2f}s")
        # Return data needed for aggregation
        return {'fitness_history': filled_fitness, 'time': total_time, 'final_fitness': final_fitness}

    except Exception as e:
        print(f"    Run {run_id} ERROR for {algo_name} on {problem_instance_copy.name()}: {e}")
        traceback.print_exc()
        # Return failure indicators
        return {'fitness_history': [float('inf')] * (max_evals // freq), 'time': 0, 'final_fitness': float('inf')}

def run_all_experiments_multi(num_parallel_workers: int = None): # Add parameter for parallelism level
    """
    Runs all algorithm-problem combinations, parallelizing the 'no_of_runs' loop.
    """
    global algorithms_factory, group_of_algorithms, problems, no_of_runs, number_of_variables, solutions_size, \
           max_evaluations, frequency, algorithm_colors, results_dir # Access globals

    # Determine number of workers for the Pool
    if num_parallel_workers is None or num_parallel_workers <= 0:
        # Default to number of CPU cores if not specified or invalid
        num_parallel_workers = cpu_count()
        print(f"Number of parallel workers not specified, defaulting to {num_parallel_workers}")
    else:
        # Use specified number, but don't exceed available cores
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

        all_experiment_data = [] # Store data for final overall pickle

        for problem in problems:
            # Deep copy problem for isolation if needed (safer for multiprocessing)
            # Note: Depends on whether problem object is modified or has complex state
            # If problem is simple and read-only, deepcopy might be overkill
            try:
                problem_for_runs = copy.deepcopy(problem)
            except Exception as e:
                print(f"Warning: Could not deepcopy problem {problem.name()}. Using original. Error: {e}")
                problem_for_runs = problem

            problem_name = problem_for_runs.name() if hasattr(problem_for_runs, 'name') else problem_for_runs.__class__.__name__
            safe_problem_name = problem_name.replace(' ', '_').replace('-', '_')
            # print(f"\n===== Processing Problem: {problem_name} =====")

            problem_aggregated_data = {'problem': problem_name, 'n_vars': number_of_variables, 'results': {}}

            for algo_name, algo_lambda in algorithms.items():
                # print(f"--- Algorithm: {algo_name} ---")

                # Prepare arguments for parallel runs
                # Pass deep copies of problem if necessary/possible
                run_args = [(algo_name, copy.deepcopy(problem_for_runs), algo_lambda, run_id + 1, max_evaluations, frequency)
                            for run_id in range(no_of_runs)]

                # Execute runs in parallel
                with Pool(processes=num_parallel_workers) as pool:
                    # map blocks until all results are back
                    run_results = pool.map(run_single_instance, run_args)

                # --- Aggregate results from parallel runs ---
                all_fitness_data_list = [res['fitness_history'] for res in run_results if res] # Collect histories
                total_times_list = [res['time'] for res in run_results if res]
                final_fitness_list = [res['final_fitness'] for res in run_results if res]

                if not all_fitness_data_list: # Handle case where all runs failed
                     print(f"  ERROR: All runs failed for {algo_name} on {problem_name}.")
                     avg_fitness = float('inf')
                     std_dev = float('nan')
                     avg_time = 0.0
                     aggregated_fitness_array = np.array([[float('inf')]* (max_evaluations // frequency)]) # Placeholder array
                else:
                    # Filter out Inf before calculating stats if needed
                    valid_final_fitness = [f for f in final_fitness_list if np.isfinite(f)]
                    if not valid_final_fitness:
                        avg_fitness = float('inf')
                        std_dev = float('nan')
                    else:
                        avg_fitness = np.mean(valid_final_fitness)
                        std_dev = np.std(valid_final_fitness)

                    avg_time = np.mean(total_times_list) if total_times_list else 0.0
                    sum_time = np.sum(total_times_list) if total_times_list else 0.0
                    humanized_duration = humanize.naturaldelta(sum_time)
                    aggregated_fitness_array = np.array(all_fitness_data_list)


                problem_aggregated_data['results'][algo_name] = {
                    'data': aggregated_fitness_array, # Store the array of histories
                    'avg_fitness': avg_fitness,
                    'std_dev': std_dev,
                    'avg_time': avg_time
                }

                # --- Log and Write CSV Row ---
                print(f"Aggregated: Algorithm: {algo_name}, Problem: {problem_name}, "
                      f"Avg Final Fitness: {avg_fitness:.4f}, Std Dev: {std_dev:.4f}, Avg single run duration: {avg_time:.2f}s, "
                      f"Duration: {humanized_duration}, Finished at: {datetime.now()}, ")
                writer.writerow([algo_name, problem_name, number_of_variables, no_of_runs, avg_fitness,
                                 std_dev, avg_time])
                file.flush() # Ensure data is written periodically

                # --- Pickle per algorithm-problem (optional, maybe remove if overall pickle is enough) ---
                with open(
                        f'{dimensions_dir}/{safe_problem_name}_dim{number_of_variables}_runs{no_of_runs}_{algo_name}_experiment_data.pkl',
                        'wb') as f:
                    # Save only data relevant to this specific result
                    single_result_data = {
                         'problem': problem_name,
                         'n_vars': number_of_variables,
                         'results': { algo_name: problem_aggregated_data['results'][algo_name] }
                    }
                    pickle.dump(single_result_data, f)


            # --- Plotting and saving overall problem data ---
            print(f"--- Finished all algorithms for Problem: {problem_name} ---")
            all_experiment_data.append(problem_aggregated_data) # Add data for this problem

            # Optional: Perform plotting here using problem_aggregated_data
            try:
                 print(f"  Generating plots for {problem_name}...")
                 # Call your plotting functions (ensure they handle the aggregated data structure)
                 plot_results(problem_aggregated_data['results'], problem_for_runs, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors)
                 # ... other plot calls ...
                 for group_name, algorithm_list in group_of_algorithms.items():
                      filtered_results = {algo: problem_aggregated_data['results'][algo] for algo in algorithm_list if algo in problem_aggregated_data['results']}
                      if len(filtered_results) > 1 or (len(filtered_results) == 1 and 'PSO' not in filtered_results): # Avoid plotting single-algo groups unless it's not just PSO
                          print(f"    Generating plots for group: {group_name}")
                          plot_results(filtered_results, problem_for_runs, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors, group_name)
                          # ... other plot calls for groups ...
            except Exception as plot_err:
                 print(f"  ERROR generating plots for {problem_name}: {plot_err}")


            # --- Pickle cumulative data after each problem ---
            # Saves progress more frequently
            overall_pickle_filename = f'{dimensions_dir}/{timestamp_str}_{safe_problem_name}_all_problems_dim{number_of_variables}_runs{no_of_runs}_experiment_data.pkl'
            print(f"  Saving cumulative results to {overall_pickle_filename}")
            with open(overall_pickle_filename, 'wb') as f:
                pickle.dump(all_experiment_data, f)


    print("\n===== All Problems Processed =====")
    # Final save (redundant if saved after each problem, but safe)
    overall_pickle_filename = f'{dimensions_dir}/{timestamp_str}_all_problems_dim{number_of_variables}_runs{no_of_runs}_experiment_data.pkl'
    with open(overall_pickle_filename, 'wb') as f:
        pickle.dump(all_experiment_data, f)
    print(f"Experiment data saved in {dimensions_dir}")