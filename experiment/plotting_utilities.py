from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt

# from experiment.setup import setup_experiment

# (_, _, no_of_runs, number_of_variables, solutions_size, max_evaluations, frequency,
#  algorithm_colors, results_dir) = setup_experiment()


def plot_results(data_dict, problem, results_dir, max_evaluations, no_of_runs, algorithm_colors, group_name="all"):
    plt.figure(figsize=(12, 6))
    for label, fitness_data in data_dict.items():
        average_fitness = np.mean(fitness_data['data'], axis=0)
        color = algorithm_colors.get(label, 'black')  # Use the global color dictionary
        plt.plot(average_fitness, label=label, color=color)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Average Best Fitness over {no_of_runs} runs')
    plt.legend(frameon=True, facecolor='white', framealpha=1)
    plt.grid()
    plt.tight_layout()
    safe_group_name = group_name.replace(' ', '_').replace('-', '_')
    filename = f"{results_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{problem.name()}_{safe_group_name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_results_with_std(data_dict, problem, results_dir, max_evaluations, no_of_runs, algorithm_colors,
                          group_name="all"):
    plt.figure(figsize=(12, 6))
    for label, fitness_data in data_dict.items():
        average_fitness = np.mean(fitness_data['data'], axis=0)
        std_dev_fitness = np.std(fitness_data['data'], axis=0)
        color = algorithm_colors.get(label, 'black')  # Use the global color dictionary
        plt.plot(average_fitness, label=label, color=color)
        plt.fill_between(range(len(average_fitness)),
                         average_fitness - std_dev_fitness,
                         average_fitness + std_dev_fitness,
                         color=color, alpha=0.2)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Average Best Fitness over {no_of_runs} runs')
    plt.legend(frameon=True, facecolor='white', framealpha=1)
    plt.grid()
    plt.tight_layout()
    safe_group_name = group_name.replace(' ', '_').replace('-', '_')
    filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}_{safe_group_name}_with_stddev.png'
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_box_at_intervals(data_dict, problem, interval=10, max_evaluations=25000, no_of_runs=10,
                          algorithms_to_compare=None, results_dir=None,
                          algorithm_colors=None,
                          group_name=None):
    if algorithms_to_compare is None:
        algorithms_to_compare = data_dict.keys()

    plt.figure(figsize=(12, 6))

    # Prepare legend entries
    legend_handles = []

    # Determine the maximum number of evaluations across all runs
    max_evaluations_index = max([len(fitness_data['data'][0]) for fitness_data in data_dict.values()])

    for label, fitness_data in data_dict.items():
        if label not in algorithms_to_compare:
            continue

        box_data = []
        positions = []
        for i in range(0, max_evaluations_index, interval):
            box_data.append([run_data[i] for run_data in fitness_data['data']])
            positions.append(i)

        # Add the final evaluation data to the box plot if not already included
        if max_evaluations_index - 1 not in positions:
            box_data.append([run_data[-1] for run_data in fitness_data['data']])
            positions.append(max_evaluations_index - 1)

        # Use the global color dictionary
        color = algorithm_colors.get(label, 'black')  # Default to black if not specified
        plt.boxplot(box_data,
                    positions=positions,
                    widths=5,
                    patch_artist=True,
                    boxprops=dict(facecolor=color, color=color),
                    whiskerprops=dict(color=color),
                    capprops=dict(color=color),
                    medianprops=dict(color='yellow'),
                    flierprops=dict(marker='o', color=color, markersize=5, alpha=0.5))

        # Append to legend handles
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

    # Add legend manually
    plt.legend(handles=legend_handles, frameon=True, facecolor='white', framealpha=1)

    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Fitness Distribution over {no_of_runs} runs')
    plt.grid()

    # Adjust x-axis range and labels
    plt.xlim([-5, max_evaluations_index + 5])  # Add padding to avoid cropping
    plt.xticks(np.arange(0, max_evaluations_index + 1, interval),
               labels=np.arange(0, max_evaluations_index + 1, interval))

    plt.tight_layout()

    # filename
    base_filename = f"{results_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{problem.name()}"
    algorithm_names = '_'.join(algo.replace(' ', '_').replace('-', '_') for algo in algorithms_to_compare)
    max_length = 255 - len(base_filename) - len("_etc") - len("_box_intervals.png")

    if len(algorithm_names) > max_length:
        truncated_algorithm_names = algorithm_names[:max_length] + "_etc"
    else:
        truncated_algorithm_names = algorithm_names

    safe_group_name = group_name.replace(' ', '_').replace('-', '_') if group_name else truncated_algorithm_names
    filename = f"{base_filename}_{safe_group_name}_box_intervals.png"

    plt.savefig(filename, dpi=300)
    plt.show()


def plot_final_box(data_dict, problem, results_dir, algorithm_colors, group_name="all"):
    plt.figure(figsize=(12, 6))
    box_data = [fitness_data['data'][:, -1] for fitness_data in data_dict.values()]
    labels = [label for label in data_dict.keys()]
    colors = [algorithm_colors.get(label, 'black') for label in labels]
    bp = plt.boxplot(box_data, labels=labels, patch_artist=True)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.ylabel('Final Fitness Distribution')
    plt.xticks(rotation=45, ha="right")

    # Remove the top, right, and bottom spines (the frame around the boxes)
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)

    plt.tight_layout()

    plt.tick_params(axis='x', which='both', bottom=False, top=False)
    safe_group_name = group_name.replace(' ', '_').replace('-', '_')
    filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{problem.name()}_{safe_group_name}_final_box.png'
    plt.savefig(filename, dpi=300)
    plt.show()
