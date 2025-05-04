from datetime import datetime

import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors


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
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    filename = f"{results_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_problem_name}_{safe_group_name}.png"
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_results_with_annotations(data_dict, problem, results_dir, max_evaluations, no_of_runs, algorithm_colors,
                                  group_name="all"):
    # Create the figure and axes
    fig, ax = plt.subplots(figsize=(12, 6))
    # Reserve space on the right side so annotations at x>1.0 in axes fraction are visible
    # Adjust 0.8 to a smaller or larger fraction if you have many algorithms or big boxes
    fig.subplots_adjust(right=0.8)

    # 1) Plot each algorithm's average curve. Collect final fitness info.
    annot_info = []
    for label, fitness_data in data_dict.items():
        avg_fit = np.mean(fitness_data['data'], axis=0)
        color = algorithm_colors.get(label, 'black')
        ax.plot(avg_fit, label=label, color=color)

        final_y = avg_fit[-1]  # final average fitness
        final_x = len(avg_fit) - 1  # last evaluation index
        annot_info.append((label, final_x, final_y, color))

    # Basic labeling & legend
    ax.set_title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    ax.set_xlabel(f'Evaluations ({max_evaluations})')
    ax.set_ylabel(f'Average Best Fitness over {no_of_runs} runs')
    ax.legend(frameon=True, facecolor='white', framealpha=1)
    ax.grid(True)

    # 2) Sort annotations by final fitness descending
    annot_info.sort(key=lambda x: x[2])

    # 3) Place annotations from the bottom (highest final fitness) to top (lowest final fitness)
    total_algs = len(annot_info)
    if total_algs <= 1:
        # trivial case => single annotation at about 0.1 fraction
        annotation_positions = [0.1]
    else:
        annotation_bottom = 0.05  # fraction from bottom
        annotation_top = 0.90  # fraction from top
        vertical_space = annotation_top - annotation_bottom
        if total_algs == 1:
            annotation_positions = [(annotation_bottom + annotation_top) / 2.0]
        else:
            annotation_spacing = 0.06
            annotation_positions = [
                annotation_bottom + i * annotation_spacing
                for i in range(total_algs)
            ]

    # 4) Annotate each curve
    #    We'll place the annotation box at x=1.02 in axes fraction,
    #    from bottom to top (descending final fitness).
    annotation_x = 1.02
    for (pos, (label, final_x, final_y, color)) in zip(annotation_positions, annot_info):
        arrow_color = color
        ax.annotate(
            text=f"{final_y:.2f}",
            xy=(final_x, final_y),  # arrow start in data coords
            xycoords='data',
            xytext=(annotation_x, pos),  # box in axes fraction coords
            textcoords='axes fraction',
            ha='left', va='center',
            bbox=dict(
                boxstyle="round,pad=0.3",
                edgecolor=arrow_color,
                facecolor="white"
            ),
            arrowprops=dict(
                arrowstyle="-",
                color="grey",
                connectionstyle="arc3,rad=0",  # makes the line straight
                relpos=(0., 0.5)  # arrow attaches at box's left-center
            ),
            fontsize=10,
            color="black",
        )

    # Final layout & saving
    plt.tight_layout()
    safe_group_name = group_name.replace(' ', '_').replace('-', '_')
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    filename = f"{results_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_problem_name}_{safe_group_name}.png"
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
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{safe_problem_name}_{safe_group_name}_with_stddev.png'
    plt.savefig(filename, dpi=300)
    plt.show()


def plot_box_at_intervals(data_dict, problem, interval=10, max_evaluations=25000, no_of_runs=10,
                          algorithms_to_compare=None, results_dir=None,
                          algorithm_colors=None,
                          group_name=None,
                          box_alpha=0.5):
    """
    Plot boxplots at specified evaluation intervals.
    Often, when there are many intervals the boxes overlap;
    here we add transparency (via an alpha value) so that overlapping areas are still visible.

    Parameters:
        data_dict (dict): Dictionary with keys as algorithm names and values being dicts containing key 'data'
                          whose rows are runs and columns evaluations.
        problem: An object with methods name() and number_of_variables() describing the problem.
        interval (int): Interval between evaluations to plot.
        max_evaluations (int): Maximum evaluation count (used for the label).
        no_of_runs (int): Number of runs.
        algorithms_to_compare (iterable): Which algorithms to plot (defaults to all keys in data_dict).
        results_dir (str): Directory where the result image is saved.
        algorithm_colors (dict): Mapping from algorithm names to colors.
        group_name (str): A label to incorporate in the filename.
        box_alpha (float): Transparency value for the box patches (between 0 and 1).
    """
    if algorithms_to_compare is None:
        algorithms_to_compare = data_dict.keys()

    plt.figure(figsize=(12, 6))

    # Prepare legend entries.
    legend_handles = []

    # Determine the maximum number of evaluations across all runs.
    max_evaluations_index = max([len(fitness_data['data'][0]) for fitness_data in data_dict.values()])

    for label, fitness_data in data_dict.items():
        if label not in algorithms_to_compare:
            continue

        box_data = []
        positions = []
        for i in range(0, max_evaluations_index, interval):
            box_data.append([run_data[i] for run_data in fitness_data['data']])
            positions.append(i)

        # Add the final evaluation data if not already included.
        if max_evaluations_index - 1 not in positions:
            box_data.append([run_data[-1] for run_data in fitness_data['data']])
            positions.append(max_evaluations_index - 1)

        # Use the global color dictionary; default to black if not specified.
        color = algorithm_colors.get(label, 'black')
        # Optionally, if you want to lighten the color:
        # color_for_box = lighten_color(color, amount=0.5)
        color_for_box = color  # or use the lightened version

        bp = plt.boxplot(box_data,
                         positions=positions,
                         widths=5,
                         patch_artist=True,
                         boxprops=dict(facecolor=color_for_box, color=color),
                         whiskerprops=dict(color=color),
                         capprops=dict(color=color),
                         medianprops=dict(color='yellow'),
                         flierprops=dict(marker='o', color=color, markersize=5, alpha=box_alpha))

        # Set transparency on each box patch.
        for patch in bp['boxes']:
            patch.set_alpha(box_alpha)

        # Append a legend handle.
        legend_handles.append(plt.Line2D([0], [0], color=color, lw=2, label=label))

    # Add legend manually.
    plt.legend(handles=legend_handles, frameon=True, facecolor='white', framealpha=1)
    plt.title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    plt.xlabel(f'Evaluations ({max_evaluations})')
    plt.ylabel(f'Fitness Distribution over {no_of_runs} runs')
    plt.grid()

    # Adjust x-axis range and labels.
    plt.xlim([-5, max_evaluations_index + 5])
    plt.xticks(np.arange(0, max_evaluations_index + 1, interval),
               labels=np.arange(0, max_evaluations_index + 1, interval))

    plt.tight_layout()

    # Build the filename.
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    base_filename = f"{results_dir}/{datetime.now().strftime('%Y%m%d_%H%M%S')}_{safe_problem_name}"
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
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{safe_problem_name}_{safe_group_name}_final_box.png'
    plt.savefig(filename, dpi=300)
    plt.show()


def lighten_color(color, amount=0.5):
    try:
        c = mcolors.cnames[color]
    except KeyError:
        c = color
    c = np.array(mcolors.to_rgb(c))
    white = np.array([1, 1, 1])
    return tuple(white - (white - c) * amount)


def plot_final_raincloud(data_dict, problem, results_dir, algorithm_colors,
                         group_name="all", scatter_mode="systematic_spread",
                         adaptive_height=False):
    """
    Create a raincloud plot (half-violin, box, and scatter) for the final fitness data.

    The scatter points are arranged with one of three modes:
      - "jitter": random vertical perturbation around a center line
      - "organized": all points share the same vertical position
      - "systematic_spread": points are evenly distributed in a vertical band

    The function can be used in two versions:
      1. A fixed-size version (adaptive_height=False; default 12 x 6 inches)
      2. An adaptive version (adaptive_height=True) in which the vertical dimension of the figure
         is computed based on the number of algorithms (data_dict keys). You can adjust the scaling
         factor as needed.

    Parameters:
        data_dict (dict): Dictionary with keys as algorithm names and each value being a dict
                          that includes a key 'data' containing an array whose last column is the
                          final fitness.
        problem: An object with methods .name() and .number_of_variables() to describe the problem.
        results_dir (str): Directory where the plot image should be saved.
        algorithm_colors (dict): Mapping from algorithm names to their respective colors.
        group_name (str): Label to help name the saved file.
        scatter_mode (str): How to arrange the scatter points; one of
                            "jitter", "organized", or "systematic_spread".
        adaptive_height (bool): If True, the figure height is computed from the number of algorithms.
                                Otherwise, a fixed height (6 inches) is used.

    """
    # Extract final fitness values for each algorithm
    rain_data = [fitness_data['data'][:, -1] for fitness_data in data_dict.values()]
    labels = list(data_dict.keys())
    num_groups = len(labels)

    # Determine colors: use provided colors for boxes and scatter; lighten for the violins.
    box_colors = [algorithm_colors.get(label, 'black') for label in labels]
    violin_colors = [lighten_color(color, 0.5) for color in box_colors]
    scatter_colors = box_colors

    # Define positions for each group on the categorical (vertical) axis.
    positions = np.arange(1, num_groups + 1)

    # Define parameters for the box/violin and scatter placements.
    box_width = 0.2       # thinner boxes
    scatter_offset = 0.25 # moves scatter points below the violin and box
    halfwidth = 0.05      # half the vertical range for systematic scatter distribution

    # Compute figure size:
    fig_width = 12
    # If adaptive, compute a new height based on number of groups (algorithms)
    if adaptive_height:
        # For example, use a formula: height = max(6, 0.5 * num_groups + 5)
        fig_height = max(6, 0.33 * num_groups + 5)
    else:
        fig_height = 6

    # Create figure and axis with the computed size.
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # ----------- Plot the boxplots (horizontal) -----------
    bp = ax.boxplot(rain_data, patch_artist=True, vert=False,
                    positions=positions, widths=box_width)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)

    # ----------- Plot the half-violin plots (horizontal) -----------
    vp = ax.violinplot(rain_data, positions=positions, points=500, showmeans=False,
                       showextrema=False, showmedians=False, vert=False)
    for idx, body in enumerate(vp['bodies']):
        vertices = body.get_paths()[0].vertices
        # Clip the violin to show only the upper half:
        lower_bound = positions[idx]
        upper_bound = positions[idx] + 0.5
        vertices[:, 1] = np.clip(vertices[:, 1], lower_bound, upper_bound)
        body.set_color(violin_colors[idx])
        body.set_alpha(0.7)

    # ----------- Plot the scatter points -----------
    # For each group, define a base vertical position.
    for idx, data in enumerate(rain_data):
        # Base y is computed from the group position and an offset to position the scatter below the main elements.
        base_y = positions[idx] - scatter_offset
        n_points = len(data)

        if scatter_mode == "jitter":
            # Add a small random noise (jitter) to the base_y.
            y_values = np.full(n_points, base_y) + np.random.uniform(-0.02, 0.02, size=n_points)
        elif scatter_mode == "organized":
            # All points share exactly the same y-coordinate.
            y_values = np.full(n_points, base_y)
        elif scatter_mode == "systematic_spread":
            # Evenly spread the points over a small vertical interval.
            # If more than one point exists, spread them evenly around base_y;
            # if a single point, keep it centered.
            if n_points > 1:
                offsets = np.linspace(-halfwidth, halfwidth, n_points)
            else:
                offsets = np.array([0.0])
            y_values = np.full(n_points, base_y) + offsets
        else:
            raise ValueError("scatter_mode must be 'jitter', 'organized', or 'systematic_spread'")

        ax.scatter(data, y_values, s=10, c=scatter_colors[idx], alpha=0.9, edgecolor='none')

    # ----------- Finishing touches -----------
    ax.set_yticks(positions)
    ax.set_yticklabels(labels)
    ax.set_title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    ax.set_xlabel('Final Fitness Distribution')

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()

    # Save the figure with a filename incorporating the current timestamp and group name.
    safe_group_name = group_name.replace(' ', '_').replace('-', '_')
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{safe_problem_name}_{safe_group_name}_raincloud.png'
    plt.savefig(filename, dpi=300)
    plt.show()


# ---------------------------------------------------------------------
# 2. Vertical “Petit Prince” RainCloud Plot
# ---------------------------------------------------------------------
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.colors as mcolors
from datetime import datetime # Make sure datetime is imported

# ...(Keep other plotting functions and helpers like lighten_color as they are)...

# ---------------------------------------------------------------------
# 2. Vertical “Petit Prince” RainCloud Plot (Corrected)
# ---------------------------------------------------------------------
def plot_final_petit_prince(data_dict, problem, results_dir, algorithm_colors,
                            group_name="all", scatter_mode="systematic_spread",
                            adaptive_width=False, side_split=True, side_offset=0.25):
    """
    Create a vertical raincloud plot, handling cases with zero variance in data.
    (Rest of docstring is the same)
    """
    # Extract final fitness values (used as y-values) for each algorithm.
    rain_data = [fitness_data['data'][:, -1] for fitness_data in data_dict.values()]
    labels = list(data_dict.keys())
    num_groups = len(labels)

    # Colors: Use given colors for box and scatter; lighten for violins.
    box_colors = [algorithm_colors.get(label, 'black') for label in labels]
    violin_colors = [lighten_color(color, 0.5) for color in box_colors]
    scatter_colors = box_colors

    # Positions along the x-axis for each algorithm.
    positions = np.arange(1, num_groups + 1)

    # Plotting parameters.
    box_width = 0.2
    halfwidth = 0.05 # For systematic scatter

    # Figure size: Fixed height (6 inches) and adaptive width if desired.
    fig_height = 8 # Increased default height slightly
    if adaptive_width:
        fig_width = max(8, 0.5 * num_groups + 5) # Adjusted adaptive formula
    else:
        fig_width = 12

    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    # --- Preprocess data for violinplot: Add jitter if variance is zero ---
    jittered_rain_data = []
    for i, data in enumerate(rain_data):
        # Ensure data is a numpy array for std calculation
        data_arr = np.asarray(data)
        if len(data_arr) > 1 and np.std(data_arr) < 1e-9: # Check if std dev is effectively zero
            print(f"Warning: Zero variance detected for algorithm '{labels[i]}'. Adding jitter for violin plot.")
            # Add small Gaussian noise. Scale noise based on data magnitude or use a small fixed value.
            data_mean = np.mean(data_arr)
            # Use a small fraction of the mean, or a minimum absolute value if mean is near zero
            noise_std = max(1e-6, abs(data_mean) * 1e-4)
            jittered_data = data_arr + np.random.normal(0, noise_std, size=data_arr.shape)
            jittered_rain_data.append(jittered_data)
        else:
            jittered_rain_data.append(data_arr) # Use original data if variance > 0 or only 1 point

    # ----- Plot the vertical boxplots (use ORIGINAL data) -----
    bp = ax.boxplot(rain_data, patch_artist=True, vert=True, showfliers=False, # Hide fliers, scatter shows points
                    positions=positions, widths=box_width)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.4)
    # Optionally style median lines etc.
    for median in bp['medians']:
         median.set_color('yellow')
         median.set_linewidth(1.5)


    # ----- Plot the vertical half-violin plots (use JITTERED data) -----
    try:
        vp = ax.violinplot(jittered_rain_data, positions=positions, points=500, showmeans=False,
                           showextrema=False, showmedians=False, vert=True)
        for idx, body in enumerate(vp['bodies']):
            vertices = body.get_paths()[0].vertices
            if side_split:
                lower_bound = positions[idx] - 0.5 # Draw violin to the left
                upper_bound = positions[idx]
            else:
                lower_bound = positions[idx] # Draw violin to the right
                upper_bound = positions[idx] + 0.5
            vertices[:, 0] = np.clip(vertices[:, 0], lower_bound, upper_bound)
            body.set_color(violin_colors[idx])
            body.set_alpha(0.7)
    except Exception as e: # Catch potential errors during violin plotting
        print(f"Warning: Violin plot failed. Skipping violins. Error: {e}")


    # ----- Plot the scatter points (use ORIGINAL data) -----
    for idx, data in enumerate(rain_data): # Iterate ORIGINAL data
        if side_split:
            base_x = positions[idx] + side_offset # Shift scatter to the right
        else:
            base_x = positions[idx] - side_offset # Shift scatter to the left
        n_points = len(data)

        if n_points == 0: continue # Skip if no data points

        if scatter_mode == "jitter":
            x_values = np.full(n_points, base_x) + np.random.uniform(-0.03, 0.03, size=n_points) # Slightly more jitter
        elif scatter_mode == "organized":
            x_values = np.full(n_points, base_x)
        elif scatter_mode == "systematic_spread":
            if n_points > 1:
                offsets = np.linspace(-halfwidth, halfwidth, n_points)
            else:
                offsets = np.array([0.0])
            x_values = np.full(n_points, base_x) + offsets
        else: # Fallback or raise error
             print(f"Warning: Unknown scatter_mode '{scatter_mode}'. Using organized.")
             x_values = np.full(n_points, base_x)

        ax.scatter(x_values, data, s=12, c=[scatter_colors[idx]], alpha=0.8, edgecolor='k', linewidth=0.3) # Added black edge

    # ----- Finishing touches -----
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=10) # Ensure readable font size
    ax.set_title(f'{problem.name()} ({problem.number_of_variables()} dimensions)')
    ax.set_ylabel('Final Fitness Distribution')
    ax.set_xlabel('Algorithms')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', linestyle=':', alpha=0.6) # Add light horizontal grid

    plt.tight_layout(rect=[0, 0.05, 1, 0.97]) # Adjust rect to give labels space

    # Save the figure
    safe_group_name = group_name.replace(' ', '_').replace('-', '_')
    safe_problem_name = problem.name().replace(' ', '_').replace('-', '_')
    filename = f'{results_dir}/{datetime.now().strftime("%Y%m%d_%H%M%S")}_{safe_problem_name}_{safe_group_name}_raincloud_vertical.png'
    try:
        plt.savefig(filename, dpi=300)
        print(f"Saved vertical raincloud plot to {filename}")
    except Exception as e:
        print(f"Error saving vertical raincloud plot: {e}")
    plt.show()