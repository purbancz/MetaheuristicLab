import pickle

import numpy as np

from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box
from experiment.setup import setup_experiment, make_dir

# Setup experiment to retrieve settings like algorithm_colors, max_evaluations, etc.
(algorithms, problems, _, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()


def load_data_from_pickle(file_path):
    with open(file_path, 'rb') as f:
        loaded_data = pickle.load(f)
    return loaded_data


def plot_all_from_pickle(file_path):
    loaded_data = load_data_from_pickle(file_path)

    for problem_data in loaded_data:
        problem_name = problem_data['problem']
        n_vars = problem_data['n_vars']
        results = problem_data['results']

        # Matching problem instance
        matched_problem = next((prob for prob in problems if prob.name() == problem_name), None)

        if matched_problem is None:
            print(f"Problem {problem_name} not found in the setup experiment list.")
            continue

        no_of_runs = problem_data['results']['GA']['data'].shape[0]

        # Directory to save plots for the specific problem
        dimensions_dir = f"{results_dir}/dim{n_vars}_runs{no_of_runs}"
        make_dir(dimensions_dir)

        # Plotting all required graphs
        plot_results(results, matched_problem, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors)
        plot_results_with_std(results, matched_problem, dimensions_dir, max_evaluations, no_of_runs, algorithm_colors)
        plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                              no_of_runs=no_of_runs, algorithms_to_compare=algorithms.keys(),
                              results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots for each individual algorithm
        for algorithm in algorithms.keys():
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=no_of_runs, algorithms_to_compare=[algorithm],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=no_of_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting final box plot comparing all algorithms
        plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)


def combine_data(data_list):
    combined_data = {}
    total_runs = 0

    for data in data_list:
        # Accumulate the number of runs from each data set
        total_runs += data[0]['results']['GA']['data'].shape[0]  # You can change 'GA' to any consistently run algorithm

        for problem_data in data:
            problem_name = problem_data['problem']
            n_vars = problem_data['n_vars']
            results = problem_data['results']

            if problem_name not in combined_data:
                combined_data[problem_name] = {
                    'n_vars': n_vars,
                    'results': {algo: {'data': [], 'avg_fitness': [], 'std_dev': [], 'avg_time': []}
                                for algo in results.keys()}
                }

            for algo, algo_data in results.items():
                combined_data[problem_name]['results'][algo]['data'].append(algo_data['data'])
                combined_data[problem_name]['results'][algo]['avg_fitness'].append(algo_data['avg_fitness'])
                combined_data[problem_name]['results'][algo]['std_dev'].append(algo_data['std_dev'])
                combined_data[problem_name]['results'][algo]['avg_time'].append(algo_data['avg_time'])

    # Aggregating the data
    for problem_name, problem_data in combined_data.items():
        for algo, algo_data in problem_data['results'].items():
            # Concatenate the list of arrays into a single array
            algo_data['data'] = np.concatenate(algo_data['data'], axis=0)
            algo_data['avg_fitness'] = np.mean(algo_data['avg_fitness'])
            algo_data['std_dev'] = np.std(algo_data['avg_fitness'])
            algo_data['avg_time'] = np.mean(algo_data['avg_time'])

    return combined_data, total_runs


def plot_combined_data_from_pickles(pickle_files):
    data_list = [load_data_from_pickle(file) for file in pickle_files]
    combined_data, total_runs = combine_data(data_list)

    for problem_name, problem_data in combined_data.items():
        n_vars = problem_data['n_vars']
        results = problem_data['results']

        # Matching problem instance
        matched_problem = next((prob for prob in problems if prob.name() == problem_name), None)

        if matched_problem is None:
            print(f"Problem {problem_name} not found in the setup experiment list.")
            continue

        # Directory to save plots for the specific problem
        dimensions_dir = f"{results_dir}/dim{n_vars}_runs{total_runs}"
        make_dir(dimensions_dir)

        # Plotting all required graphs
        plot_results(results, matched_problem, dimensions_dir, max_evaluations, total_runs, algorithm_colors)
        plot_results_with_std(results, matched_problem, dimensions_dir, max_evaluations, total_runs, algorithm_colors)
        plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                              no_of_runs=total_runs, algorithms_to_compare=algorithms.keys(),
                              results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots for each individual algorithm
        for algorithm in algorithms.keys():
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=total_runs, algorithms_to_compare=[algorithm],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
            plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=total_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
                                  results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting final box plot comparing all algorithms
        plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)
