import pickle
import numpy as np
import scikit_posthocs as sp
import pandas as pd
from scipy.stats import kruskal, f_oneway, shapiro
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from scikit_posthocs import posthoc_dunn

from experiment.plotting_utilities import plot_results, plot_results_with_std, plot_box_at_intervals, plot_final_box, \
    plot_final_raincloud, plot_final_petit_prince, plot_results_with_annotations
from experiment.setup import setup_experiment, make_dir

# Setup experiment to retrieve settings like algorithm_colors, max_evaluations, etc.
(algorithms, group_of_algorithms, problems, _, number_of_variables, solutions_size,
 max_evaluations, frequency, algorithm_colors, results_dir) = setup_experiment()


# def load_data_from_pickle(file_path):
#     with open(file_path, 'rb') as f:
#         loaded_data = pickle.load(f)
#     return loaded_data


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

        # # Plotting box plots comparing PGxHEA algorithms with GA and PSO
        # for algorithm in ['PGSHEA', 'PGPHEA', 'PGCHEA']:
        #     plot_box_at_intervals(results, matched_problem, max_evaluations=max_evaluations,
        #                           no_of_runs=no_of_runs, algorithms_to_compare=[algorithm, 'GA', 'PSO'],
        #                           results_dir=dimensions_dir, algorithm_colors=algorithm_colors)

        # Plotting final box plot comparing all algorithms
        plot_final_box(results, matched_problem, dimensions_dir, algorithm_colors)
        plot_final_raincloud(results, matched_problem, dimensions_dir, algorithm_colors)
        plot_final_petit_prince(results, matched_problem, dimensions_dir, algorithm_colors)


# def combine_data(data_list):
#     combined_data = {}
#     total_runs = 0
#
#     for data in data_list:
#         # Accumulate the number of runs from each data set
#         total_runs += data[0]['results']['PSO']['data'].shape[
#             0]  # You can change 'GA' to any consistently run algorithm
#
#         for problem_data in data:
#             problem_name = problem_data['problem']
#             n_vars = problem_data['n_vars']
#             results = problem_data['results']
#
#             if problem_name not in combined_data:
#                 combined_data[problem_name] = {
#                     'n_vars': n_vars,
#                     'results': {algo: {'data': [], 'avg_fitness': [], 'std_dev': [], 'avg_time': []}
#                                 for algo in results.keys()}
#                 }
#
#             for algo, algo_data in results.items():
#                 combined_data[problem_name]['results'][algo]['data'].append(algo_data['data'])
#                 combined_data[problem_name]['results'][algo]['avg_fitness'].append(algo_data['avg_fitness'])
#                 combined_data[problem_name]['results'][algo]['std_dev'].append(algo_data['std_dev'])
#                 combined_data[problem_name]['results'][algo]['avg_time'].append(algo_data['avg_time'])
#
#     # Aggregating the data
#     for problem_name, problem_data in combined_data.items():
#         for algo, algo_data in problem_data['results'].items():
#             # Concatenate the list of arrays into a single array
#             algo_data['data'] = np.concatenate(algo_data['data'], axis=0)
#             algo_data['avg_fitness'] = np.mean(algo_data['avg_fitness'])
#             algo_data['std_dev'] = np.std(algo_data['avg_fitness'])
#             algo_data['avg_time'] = np.mean(algo_data['avg_time'])
#
#     return combined_data, total_runs


def load_data_from_pickle(filepath):
    """Loads data from a pickle file."""
    try:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            print(f"Successfully loaded data from: {filepath}")
            if isinstance(data, list) and data:
                 if not all(isinstance(item, dict) and 'problem' in item and 'results' in item for item in data):
                      print(f"Warning: List structure in {filepath} doesn't match expected format.")
            elif isinstance(data, dict):
                 if not ('problem' in data and 'results' in data):
                      print(f"Warning: Dictionary structure in {filepath} doesn't match expected format.")
            elif not data:
                 print(f"Warning: Pickle file {filepath} loaded empty data.")
            else:
                 print(f"Warning: Unexpected data type loaded from {filepath}: {type(data)}")
            return data
    except FileNotFoundError:
        print(f"Error: Pickle file not found at {filepath}")
        return None
    except Exception as e:
        print(f"Error loading pickle file {filepath}: {e}")
        return None

def combine_data(data_list):
    """
    Combines data from multiple experiment runs, handling different
    pickle file structures (list of problems or single problem dict).
    """
    combined_data = {}
    total_runs_calculated = 0
    runs_set = False

    valid_data_list = [data for data in data_list if data is not None]

    if not valid_data_list:
         print("Warning: No valid data loaded from pickle files.")
         return {}, 0

    for data_source in valid_data_list:
        if isinstance(data_source, list):
            problems_in_source = data_source
        elif isinstance(data_source, dict) and 'problem' in data_source and 'results' in data_source:
            problems_in_source = [data_source]
        else:
            print(f"Warning: Skipping unrecognized data source structure: {type(data_source)}")
            continue

        temp_runs_set_for_source = False
        if not runs_set:
            for p_data_check in problems_in_source:
                if p_data_check and 'results' in p_data_check and p_data_check['results']:
                     first_algo_name = next(iter(p_data_check['results']))
                     if 'data' in p_data_check['results'][first_algo_name] and \
                        isinstance(p_data_check['results'][first_algo_name]['data'], np.ndarray):
                          current_source_runs = p_data_check['results'][first_algo_name]['data'].shape[0]
                          total_runs_calculated += current_source_runs # Add runs from this source
                          temp_runs_set_for_source = True
                          runs_set = True
                          # Option B: Sum runs (if combining results from different experiments) -> already doing this
                          break # Stop checking after finding runs in this source
            if not temp_runs_set_for_source:
                 print("Warning: Could not determine number of runs from data source.")


        for problem_data in problems_in_source:
            if not isinstance(problem_data, dict) or 'problem' not in problem_data or 'results' not in problem_data:
                print(f"Warning: Skipping invalid problem data entry: {problem_data}")
                continue

            problem_name = problem_data['problem']
            n_vars = problem_data.get('n_vars', -1)
            results = problem_data['results']

            if problem_name not in combined_data:
                combined_data[problem_name] = {
                    'n_vars': n_vars,
                    'results': {}
                }
            if combined_data[problem_name]['n_vars'] == -1 and n_vars != -1:
                 combined_data[problem_name]['n_vars'] = n_vars


            for algo, algo_data_in in results.items():
                if not isinstance(algo_data_in, dict) or 'data' not in algo_data_in:
                     print(f"Warning: Skipping invalid algorithm data for '{algo}' in problem '{problem_name}'.")
                     continue

                if algo not in combined_data[problem_name]['results']:
                    combined_data[problem_name]['results'][algo] = {
                        'data_list': [], # Store individual data arrays temporarily
                        'avg_fitness_list': [],
                        'std_dev_list': [],
                        'avg_time_list': []
                    }

                if isinstance(algo_data_in['data'], np.ndarray):
                     combined_data[problem_name]['results'][algo]['data_list'].append(algo_data_in['data'])
                if 'avg_fitness' in algo_data_in: combined_data[problem_name]['results'][algo]['avg_fitness_list'].append(algo_data_in['avg_fitness'])
                if 'std_dev' in algo_data_in: combined_data[problem_name]['results'][algo]['std_dev_list'].append(algo_data_in['std_dev'])
                if 'avg_time' in algo_data_in: combined_data[problem_name]['results'][algo]['avg_time_list'].append(algo_data_in['avg_time'])


    # --- Aggregating the collected data ---
    final_aggregated_data = {}
    actual_total_runs = 0

    for problem_name, problem_data in combined_data.items():
        final_aggregated_data[problem_name] = {
             'n_vars': problem_data['n_vars'],
             'results': {}
        }
        first_algo_runs_set = False

        for algo, collected_data in problem_data['results'].items():
            if not collected_data['data_list']:
                 print(f"Warning: No data found to aggregate for '{algo}' in problem '{problem_name}'. Skipping.")
                 continue

            try:
                valid_data_arrays = [arr for arr in collected_data['data_list'] if isinstance(arr, np.ndarray)]
                if not valid_data_arrays:
                     print(f"Warning: No valid numpy arrays found for '{algo}' in problem '{problem_name}'. Skipping.")
                     continue
                concatenated_data = np.concatenate(valid_data_arrays, axis=0)
            except ValueError as e:
                 print(f"Error concatenating data for '{algo}' in problem '{problem_name}'. Skipping. Error: {e}")
                 for i, arr in enumerate(collected_data['data_list']):
                     print(f"  Array {i} type: {type(arr)}, shape: {getattr(arr, 'shape', 'N/A')}")
                 continue

            final_run_fitness = concatenated_data[:, -1] if concatenated_data.ndim == 2 else concatenated_data
            valid_final_fitness = final_run_fitness[np.isfinite(final_run_fitness)]

            if valid_final_fitness.size > 0:
                 final_avg_fitness = np.mean(valid_final_fitness)
                 final_std_dev = np.std(valid_final_fitness)
            else:
                 final_avg_fitness = float('inf')
                 final_std_dev = float('nan')

            valid_avg_times = [t for t in collected_data['avg_time_list'] if isinstance(t, (int, float)) and np.isfinite(t)]
            final_avg_time = np.mean(valid_avg_times) if valid_avg_times else 0.0

            final_aggregated_data[problem_name]['results'][algo] = {
                'data': concatenated_data,
                'avg_fitness': final_avg_fitness,
                'std_dev': final_std_dev,
                'avg_time': final_avg_time
            }

            if not first_algo_runs_set:
                 actual_total_runs = concatenated_data.shape[0]
                 first_algo_runs_set = True


    if actual_total_runs == 0 and total_runs_calculated != 0:
         print(f"Warning: Calculated total runs ({total_runs_calculated}) but aggregated data has 0 runs. Using calculated value.")
         actual_total_runs = total_runs_calculated
    elif actual_total_runs == 0:
         print("Warning: Could not determine total runs from aggregated data.")

    print(f"Data combined. Total runs detected: {actual_total_runs}")
    return final_aggregated_data, actual_total_runs

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
        plot_results_with_annotations(results, matched_problem, dimensions_dir, max_evaluations, total_runs,
                                      algorithm_colors)
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
        plot_final_raincloud(results, matched_problem, dimensions_dir, algorithm_colors)
        plot_final_raincloud(results, matched_problem, dimensions_dir, algorithm_colors, adaptive_height=True)
        plot_final_petit_prince(results, matched_problem, dimensions_dir, algorithm_colors)
        plot_final_petit_prince(results, matched_problem, dimensions_dir, algorithm_colors, adaptive_width=True)

        for group_name, algorithm_list in group_of_algorithms.items():
            filtered_results = {algo: problem_data['results'][algo] for algo in algorithm_list if
                                algo in problem_data['results']}

            if not filtered_results or set(filtered_results.keys()) == {'PSO'}:
                print(f"Skipping {group_name}, no valid algorithms found (only 'PSO' or empty).")
                continue

            plot_results(filtered_results, matched_problem, dimensions_dir, max_evaluations, total_runs,
                         algorithm_colors, group_name)
            plot_results_with_annotations(filtered_results, matched_problem, dimensions_dir, max_evaluations,
                                          total_runs, algorithm_colors, group_name)
            plot_results_with_std(filtered_results, matched_problem, dimensions_dir, max_evaluations, total_runs,
                                  algorithm_colors, group_name)
            plot_box_at_intervals(filtered_results, matched_problem, max_evaluations=max_evaluations,
                                  no_of_runs=total_runs,
                                  algorithms_to_compare=list(filtered_results.keys()), results_dir=dimensions_dir,
                                  algorithm_colors=algorithm_colors, group_name=group_name)
            plot_final_box(filtered_results, matched_problem, dimensions_dir, algorithm_colors, group_name)
            plot_final_raincloud(filtered_results, matched_problem, dimensions_dir, algorithm_colors, group_name)
            plot_final_raincloud(filtered_results, matched_problem, dimensions_dir, algorithm_colors, group_name,
                                 adaptive_height=True)
            plot_final_petit_prince(filtered_results, matched_problem, dimensions_dir, algorithm_colors)
            plot_final_petit_prince(filtered_results, matched_problem, dimensions_dir, algorithm_colors,
                                    adaptive_width=True)


def extract_best_algorithms_from_experiment_data(aggregated_data_dict):
    """
    Extracts the best algorithm(s) for each problem/dimension using the
    aggregated average fitness from combined data.
    Expects a dictionary keyed by problem name.
    """
    best_algorithms_per_problem = {}
    # Iterate through the dictionary returned by combine_data
    for problem_name, problem_data in aggregated_data_dict.items():
        dimension = problem_data.get("n_vars", "Unknown")
        results = problem_data.get("results", {})
        if not results: continue

        avg_fitness_values = {}
        for algo_name, algo_data in results.items():
            # Use the 'avg_fitness' calculated by combine_data
            if isinstance(algo_data, dict) and 'avg_fitness' in algo_data and np.isfinite(algo_data['avg_fitness']):
                avg_fitness_values[algo_name] = algo_data['avg_fitness']

        if not avg_fitness_values: continue

        try:
            best_value = min(avg_fitness_values.values())
            # Find all algorithms matching the best value (handling ties)
            best_algos = [algo for algo, fitness in avg_fitness_values.items() if np.isclose(fitness, best_value)]

            if problem_name not in best_algorithms_per_problem:
                best_algorithms_per_problem[problem_name] = {}
            best_algorithms_per_problem[problem_name][dimension] = best_algos
        except ValueError:
             print(f"Warning: No valid fitness values to compare for {problem_name} dim {dimension}.")
        except Exception as e:
             print(f"Error extracting best algorithms for {problem_name} dim {dimension}: {e}")

    return best_algorithms_per_problem


def kruskal_wallis_with_posthoc(pickle_files, perform_shapiro=False, perform_posthoc=True):
    """
    Perform Kruskal-Wallis and posthoc tests using dynamically extracted
    the best algorithms from combined data.
    """
    all_normal = False
    non_significant_dunn = []
    non_significant_tukey = []
    non_significant_vs_best_dunn = []
    non_significant_vs_best_tukey = []

    # 1. Load data from all pickle files
    print("Loading data from pickle files...")
    data_list = [load_data_from_pickle(file) for file in pickle_files]
    valid_data_list = [d for d in data_list if d is not None] # Filter out loading errors

    if not valid_data_list:
        print("Error: No valid data loaded from any pickle files.")
        return

    # 2. Combine data using your existing function
    print("Combining loaded data...")
    # combine_data handles the different list/dict structures internally
    combined_data_dict, total_runs = combine_data(valid_data_list)

    if not combined_data_dict:
        print("Error: Data combination resulted in empty dictionary.")
        return
    print(f"Data combined successfully. Total effective runs: {total_runs}")

    # 3. Extract best algorithms from the *combined* data
    print("Extracting best algorithms from combined data...")
    # Pass the dictionary output of combine_data
    best_algorithms = extract_best_algorithms_from_experiment_data(combined_data_dict)
    print("Best algorithms extracted:", best_algorithms)

    # 4. Iterate through the COMBINED data for statistical analysis
    print("\nStarting statistical analysis...")
    for problem_name, problem_data in combined_data_dict.items():
        dimension = problem_data.get('n_vars', 'Unknown')
        results = problem_data.get('results', {})

        # Get the best algorithms determined *after* combining all runs
        best_algos = best_algorithms.get(problem_name, {}).get(dimension, [])

        if not results:
            print(f"No results found for Problem: {problem_name}, Dimension: {dimension}. Skipping analysis.")
            continue

        print(f"\nPerforming analysis for Problem: {problem_name}, Dimension: {dimension}")
        if best_algos: print(f"Best algorithm(s) identified: {best_algos}")
        else: print("No best algorithm clearly identified (check extraction logic or data).")

        # Prepare data for statistical tests from the aggregated results
        final_fitness_values = {}
        algorithms_in_analysis = []
        data_for_test = []

        for algo, algo_data in results.items():
            if 'data' in algo_data and isinstance(algo_data['data'], np.ndarray) and algo_data['data'].size > 0:
                 if algo_data['data'].ndim == 2:
                      final_values = algo_data['data'][:, -1]
                 elif algo_data['data'].ndim == 1:
                      final_values = algo_data['data']
                 else: continue # Skip unexpected shape

                 finite_final_values = final_values[np.isfinite(final_values)]
                 if finite_final_values.size > 0:
                      final_fitness_values[algo] = finite_final_values
                      algorithms_in_analysis.append(algo)
                      data_for_test.append(finite_final_values)
                 else: print(f"Warning: No finite final fitness values for {algo}. Skipping.")
            else: print(f"Warning: Missing/invalid 'data' for {algo}. Skipping.")

        if len(final_fitness_values) < 2:
             print("\tNot enough algorithm groups (>1) with valid data for statistical comparison. Skipping tests.")
             continue

        if perform_shapiro:
            normality_results = {algo: shapiro(values) for algo, values in final_fitness_values.items()}
            all_normal = all(p > 0.05 for _, p in normality_results.values())
            print("\tShapiro-Wilk Test Results (p-values):")
            for algo, (stat, p_value) in normality_results.items():
                print(f"{algo}: Statistic={stat:.4f}, P-value={p_value:.4e}")
            if all_normal:
                print("\tAll groups are normally distributed. Performing ANOVA.")
                stat, p = f_oneway(*final_fitness_values.values())
            else:
                print("\tNot all groups are normally distributed. Performing Kruskal-Wallis test.")
                stat, p = kruskal(*final_fitness_values.values())
            print(f"Test Statistic: {stat}")
            print(f"P-value: {p}")
        else:
            stat, p = kruskal(*final_fitness_values.values())
            print(f"\tKruskal-Wallis Statistic: {stat}")
            print(f"P-value: {p}")

        if p < 0.05 and perform_posthoc:
            data = list(final_fitness_values.values())
            algo_names = list(final_fitness_values.keys())

            if perform_shapiro and all_normal:
                combined_data = pd.Series(data[0])
                groups = [algo_names[0]] * len(data[0])
                for i in range(1, len(data)):
                    combined_data = combined_data._append(pd.Series(data[i]))
                    groups.extend([algo_names[i]] * len(data[i]))

                tukey_results = pairwise_tukeyhsd(combined_data, groups)
                print("\nTukey's HSD test pairwise comparison results:")
                print(tukey_results.summary())

                for res in tukey_results._results_table.data[1:]:
                    group1, group2, meandiff, p_adj, lower, upper, reject = res
                    if not reject:
                        non_significant_tukey.append((problem_name, dimension, group1, group2, p_adj))
                        if group1 in best_algos or group2 in best_algos:
                            non_significant_vs_best_tukey.append((problem_name, dimension, group1, group2, p_adj))
            else:
                dunn_results = sp.posthoc_dunn(data, p_adjust='fdr_bh')
                dunn_results.index = algo_names
                dunn_results.columns = algo_names
                print("Dunn's test pairwise comparison p-values:")
                print(dunn_results)

                for i, algo1 in enumerate(algo_names):
                    for j, algo2 in enumerate(algo_names):
                        if i < j:
                            p_val = dunn_results.iloc[i, j]
                            if p_val >= 0.05:
                                non_significant_dunn.append((problem_name, dimension, algo1, algo2, p_val))
                                if algo1 in best_algos or algo2 in best_algos:
                                    non_significant_vs_best_dunn.append(
                                        (problem_name, dimension, algo1, algo2, p_val))

    # Output Results
    print("\nNon-Significant Results from Dunn's Test (LaTeX Format):")
    for result in non_significant_dunn:
        problem, dimension, algo1, algo2, p_val = result
        print(f"{problem} & {dimension} & {algo1} vs {algo2} & {p_val:.4f} \\\\")
    print(f"Size: {len(non_significant_dunn)}")

    print("\nNon-Significant Results from Tukey's HSD Test (LaTeX Format):")
    for result in non_significant_tukey:
        problem, dimension, group1, group2, p_adj = result
        print(f"{problem} & {dimension} & {group1} vs {group2} & {p_adj:.4f} \\\\")
    print(f"Size: {len(non_significant_tukey)}")

    print("\nNon-Significant Results vs Best Algorithms from Dunn's Test (LaTeX Format):")
    for result in non_significant_vs_best_dunn:
        problem, dimension, algo1, algo2, p_val = result
        print(f"{problem} & {dimension} & {algo1} vs {algo2} & {p_val:.4f} \\\\")
    print(f"Size: {len(non_significant_vs_best_dunn)}")

    print("\nNon-Significant Results vs Best Algorithms from Tukey's HSD Test (LaTeX Format):")
    for result in non_significant_vs_best_tukey:
        problem, dimension, group1, group2, p_adj = result
        print(f"{problem} & {dimension} & {group1} vs {group2} & {p_adj:.4f} \\\\")
    print(f"Size: {len(non_significant_vs_best_tukey)}")
