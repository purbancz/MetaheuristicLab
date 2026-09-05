import math

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.mplot3d import Axes3D
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin

from problem.n_variables.hgbat import HGBat
from experiment.setup import make_dir
from problem.fixed_varaibles.quantum_speed import QSLTimeBoundProblem
from problem.n_variables.CEC import RotatedHighConditionedElliptic, RotatedBentCigar, RotatedDiscus, \
    ShiftedRotatedRosenbrock, ShiftedRotatedAckley, ShiftedRastrigin, ShiftedRotatedRastrigin, ShiftedSchwefel, \
    ShiftedRotatedSchwefel, ShiftedRotatedKatsuura, ShiftedRotatedHappyCat, ShiftedRotatedHGBat, \
    ShiftedRotatedExpandedGriewankPlusRosenbrock, ShiftedRotatedExpandedScafferF6, HybridFunction1, HybridFunction2, \
    HybridFunction3, HybridFunction4, HybridFunction5, HybridFunction6, CompositionFunction1, CompositionFunction2, \
    CompositionFunction3, CompositionFunction4, CompositionFunction5, CompositionFunction6, CompositionFunction7, \
    CompositionFunction8, ShiftedRotatedSchafferF7
from problem.n_variables.ackley import Ackley
from problem.n_variables.alpine import AlpineN1, AlpineN2
from problem.n_variables.bent_cigar import BentCigar
from problem.n_variables.bird import Bird
from problem.n_variables.cross import GeneralizedCrossInTray, Cross, CrossLeggedTable
from problem.n_variables.discus import Discus
from problem.n_variables.dixon import DixonPrice, GeneralizedDixonPriceRosenbrock
from problem.n_variables.eggholder import EggHolder
from problem.n_variables.expanded_schaffer import ExpandedShaffer
from problem.n_variables.griewank import Griewank
from problem.n_variables.happy_cat import HappyCat
from problem.n_variables.holders import GeneralizedHolderTable, CarromTable, TestTubeHolder, PenHolder
from problem.n_variables.katsuura import Katsuura, ExpandedKatsuura
from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster
from problem.n_variables.levy import Levy
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.mishra import Mishra01, Mishra02, Mishra03, Mishra04, Mishra05, Mishra06, Mishra11
from problem.n_variables.penalized import GeneralizedPenalizedN1
from problem.n_variables.plateau import Plateau
from problem.n_variables.quantum_speed import QuantumSpeedLimit2D, GeneralizedQuantumSpeedLimit
from problem.n_variables.quartic import Quartic
from problem.n_variables.rosenbrock import Rosenbrock, RosenbrockModified01, RosenbrockModified02
from problem.n_variables.salomon import Salomon
from problem.n_variables.schaffer import GeneralizedSchafferN7, GeneralizedSchafferN1, GeneralizedSchafferN2, \
    GeneralizedSchafferN3, GeneralizedSchafferN4
from problem.n_variables.schmidt_vetters import GeneralizedSchmidtVetters
from problem.n_variables.schwefel import SchwefelN26, SchwefelN21, SchwefelN22, SchwefelN6, SchwefelN20, \
    SchwefelN36
from problem.n_variables.shubert import ShubertN1, ShubertN3, ShubertN4
from problem.n_variables.sine_envelope import SineEnvelope
from problem.n_variables.step import StepN1, StepN2, StepN3
from problem.n_variables.stochastic import Stochastic
from problem.n_variables.strechedv import StretchedV
from problem.n_variables.styblinski import StyblinskiTang
from problem.n_variables.weierstrass import ShiftedRotatedWeierstrass
from problem.n_variables.zakharov import Zakharov

#
# def plot_benchmark_function(func, resolution=100):
#     d = func.number_of_variables()
#     if d < 2:
#         raise ValueError("Funkcja musi mieć przynajmniej 2 zmienne do rysowania w 3D.")
#
#     x_min, x_max = func.lower_bound[0], func.upper_bound[0]
#     y_min, y_max = func.lower_bound[1], func.upper_bound[1]
#
#     x_vals = np.linspace(x_min, x_max, resolution)
#     y_vals = np.linspace(y_min, y_max, resolution)
#     x, y = np.meshgrid(x_vals, y_vals)
#     z = np.zeros_like(x)
#
#     fixed = []
#     for i in range(2, d):
#         midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
#         fixed.append(midpoint)
#
#     for i in range(resolution):
#         for j in range(resolution):
#             variables = [x[i, j], y[i, j]] + fixed
#             sol = FloatSolution(
#                 lower_bound=func.lower_bound,
#                 upper_bound=func.upper_bound,
#                 number_of_objectives=func.number_of_objectives(),
#                 number_of_constraints=func.number_of_constraints()
#             )
#             sol.variables = variables
#             func.evaluate(sol)
#             z[i, j] = sol.objectives[0]
#
#     fig = plt.figure()
#     ax = fig.add_subplot(111, projection='3d')
#     surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
#     ax.set_xlabel('x')
#     ax.set_ylabel('y')
#     ax.set_zlabel('f(x,y)', labelpad=5)
#     ax.set_title(func.name())
#     fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)
#
#     plt.tight_layout()
#     func_name = func.name().replace(' ', '_').replace('-', '_')
#
#     benchmark_dir = 'benchmark'
#     make_dir(benchmark_dir)
#     filename = f"{benchmark_dir}/{func_name}.png"
#
#     plt.savefig(filename, dpi=300)
#     plt.show()


# --- Improved Plotting Function ---
def plot_benchmark_function_maximized(func, resolution=100, fig_size=(8, 6)):
    d = func.number_of_variables()
    if d < 2:
        raise ValueError("Funkcja musi mieć przynajmniej 2 zmienne do rysowania w 3D.")

    x_min, x_max = func.lower_bound[0], func.upper_bound[0]
    y_min, y_max = func.lower_bound[1], func.upper_bound[1]

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals) # Renamed to x_mesh, y_mesh for clarity
    z_mesh = np.zeros_like(x_mesh)

    fixed_vars = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed_vars.append(midpoint)

    for i in range(resolution):
        for j in range(resolution):
            variables = [x_mesh[i, j], y_mesh[i, j]] + fixed_vars
            sol = FloatSolution(
                lower_bound=func.lower_bound,
                upper_bound=func.upper_bound,
                number_of_objectives=func.number_of_objectives(),
                number_of_constraints=func.number_of_constraints()
            )
            sol.variables = variables
            func.evaluate(sol)
            z_mesh[i, j] = sol.objectives[0]

    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111, projection='3d')

    # To make the plot take up more space within the figure window *before* saving
    # you can adjust subplot parameters. However, bbox_inches='tight' during save
    # is usually more effective for the final file.
    # fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # This can be aggressive

    surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', edgecolor='none', rstride=1, cstride=1) # rstride/cstride can affect appearance
    ax.set_xlabel('x', labelpad=10) # Adjusted labelpad
    ax.set_ylabel('y', labelpad=10) # Adjusted labelpad
    ax.set_zlabel('f(x,y)', labelpad=10) # Adjusted labelpad

    # ax.set_title(func.name()) # TITLE REMOVED

    # Optional: Make axis panes transparent for a cleaner look if desired
    # ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # Optional: Remove grid lines if they feel like clutter
    # ax.grid(False)

    # Adjust colorbar: consider its position and size carefully
    # For maximum plot space, sometimes a smaller or more carefully placed colorbar is needed.
    # Or, if the color scale is obvious or not critical, it could even be omitted.
    cbar = fig.colorbar(surf, shrink=0.5, aspect=10, pad=0.02, fraction=0.046)
    # cbar.ax.tick_params(labelsize=8) # Optionally reduce colorbar tick label size

    # plt.tight_layout() # Often not needed or can conflict when using bbox_inches='tight'.
                       # If used, call it *before* savefig. Can be useful for plt.show()
                       # but for the saved file, bbox_inches is key.

    func_name_safe = func.name().replace(' ', '_').replace('-', '_')

    benchmark_dir = 'benchmark_maximized' # Changed dir name for clarity
    make_dir(benchmark_dir)
    filename = f"{benchmark_dir}/{func_name_safe}_maximized.png"

    # Key changes for maximizing space in the saved file:
    # bbox_inches='tight' trims whitespace around the plot.
    # pad_inches controls the padding *after* trimming (can be 0 for no padding).
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0.01)
    print(f"Saved maximized plot to {filename}")
    plt.show()
    plt.close(fig) # Close the figure to free memory, especially if plotting many

# --- Improved Plotting Function ---
def plot_benchmark_function_truly_maximized(func, resolution=100, fig_size=(6, 5)):
    d = func.number_of_variables()
    if d < 2:
        raise ValueError("Funkcja musi mieć przynajmniej 2 zmienne do rysowania w 3D.")

    x_min, x_max = func.lower_bound[0], func.upper_bound[0]
    y_min, y_max = func.lower_bound[1], func.upper_bound[1]

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    x_mesh, y_mesh = np.meshgrid(x_vals, y_vals)
    z_mesh = np.zeros_like(x_mesh)

    fixed_vars = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed_vars.append(midpoint)

    for i in range(resolution):
        for j in range(resolution):
            variables = [x_mesh[i, j], y_mesh[i, j]] + fixed_vars
            sol = FloatSolution(
                lower_bound=func.lower_bound,
                upper_bound=func.upper_bound,
                number_of_objectives=func.number_of_objectives(),
                number_of_constraints=func.number_of_constraints()
            )
            sol.variables = variables
            func.evaluate(sol)
            z_mesh[i, j] = sol.objectives[0]

    # Create figure
    fig = plt.figure(figsize=fig_size)

    # --- KEY CHANGE: Use add_axes to make the axes fill the figure ---
    # [left, bottom, width, height] as fractions of figure dimensions.
    # Start with full coverage, then adjust slightly if labels are clipped.
    # ax_position = [0, 0, 1, 1] # Full coverage
    # If labels are clipped by the above, provide a small margin:
    ax_position = [0.01, 0.01, 0.98, 0.98]
    ax = fig.add_axes(ax_position, projection='3d')

    # --- Alternative using subplots_adjust (if you prefer add_subplot) ---
    # ax = fig.add_subplot(111, projection='3d')
    # fig.subplots_adjust(left=0.01, bottom=0.01, right=0.99, top=0.99) # Small margin

    surf = ax.plot_surface(x_mesh, y_mesh, z_mesh, cmap='viridis', edgecolor='none', rstride=1, cstride=1)

    # Adjust label padding - make it small
    ax.set_xlabel('x', labelpad=2, fontsize=8)  # Smaller fontsize, very small pad
    ax.set_ylabel('y', labelpad=2, fontsize=8)
    ax.set_zlabel('f(x,y)', labelpad=0, fontsize=8)  # Z label often needs less pad

    # Adjust tick parameters
    ax.tick_params(axis='x', which='major', labelsize=7, pad=1)
    ax.tick_params(axis='y', which='major', labelsize=7, pad=1)
    ax.tick_params(axis='z', which='major', labelsize=7, pad=-2)  # Pull Z ticks closer

    # To completely remove ticks and labels for absolute max plot:
    # ax.set_xticks([])
    # ax.set_yticks([])
    # ax.set_zticks([])
    # ax.set_xlabel('')
    # ax.set_ylabel('')
    # ax.set_zlabel('')

    # Colorbar: make it small and attach it well.
    # Using ax=ax is good. `fraction` and `pad` are key.
    # `anchor` and `panchor` can give even finer control over colorbar placement relative to axes.
    cbar = fig.colorbar(surf, ax=ax, shrink=0.6, aspect=20, pad=0.02, fraction=0.04)
    cbar.ax.tick_params(labelsize=6)  # Tiny colorbar ticks
    # To hide colorbar completely if not needed:
    # cbar.remove()

    func_name_safe = func.name().replace(' ', '_').replace('-', '_')
    benchmark_dir = 'benchmark_truly_maximized'
    make_dir(benchmark_dir)
    filename = f"{benchmark_dir}/{func_name_safe}_maximized.png"

    # Save the figure
    # bbox_inches='tight' will crop to the content.
    # pad_inches=0 will remove all padding around the cropped content.
    plt.savefig(filename, dpi=300, bbox_inches='tight', pad_inches=0)
    print(f"Saved truly maximized plot to {filename}")
    # plt.show() # Displaying it might look different than the saved file due to window chrome
    plt.close(fig)


def plot_benchmark_plots(func, resolution=100):
    """
    Generuje trzy wykresy dla funkcji benchmarkowej:
    1. Level sets (konturowy)
    2. Normalized rank heatmap
    3. Search space cut: lin-lin – dla d=2 wykres 3D powierzchni, dla d>2 przekrój funkcji przy stałym y.
    """
    d = func.number_of_variables()
    if d < 2:
        raise ValueError("Funkcja musi mieć przynajmniej 2 zmienne.")

    # Ustalamy zakresy dla pierwszych dwóch zmiennych
    x_min, x_max = func.lower_bound[0], func.upper_bound[0]
    y_min, y_max = func.lower_bound[1], func.upper_bound[1]
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)
    Z = np.zeros_like(X)

    # Dla d > 2 ustalamy pozostałe zmienne na ich wartość środkową
    fixed = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed.append(midpoint)

    # Ewaluujemy funkcję dla każdego punktu na siatce
    for i in range(resolution):
        for j in range(resolution):
            # Pierwsze dwie zmienne zmieniają się na siatce, pozostałe stałe
            variables = [X[i, j], Y[i, j]] + fixed
            sol = FloatSolution(
                lower_bound=func.lower_bound,
                upper_bound=func.upper_bound,
                number_of_objectives=func.number_of_objectives(),
                number_of_constraints=func.number_of_constraints()
            )
            sol.variables = variables
            func.evaluate(sol)
            Z[i, j] = sol.objectives[0]

    # Obliczamy normalized rank – porządkujemy wartości funkcji na siatce
    Z_flat = Z.flatten()
    # Używamy argsort dwukrotnie, aby uzyskać ranki
    ranks = np.argsort(np.argsort(Z_flat))
    norm_ranks = ranks / (len(Z_flat) - 1)
    norm_ranks = norm_ranks.reshape(Z.shape)

    # Tworzymy wykresy w jednej figurze (subploty) dla 2D
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Wykres 1: Level sets – konturowy
    cs = axs[0].contourf(X, Y, Z, cmap='viridis', levels=50)
    axs[0].set_title("Isovalue Heatmap")  # Scalar Field Map, Level Sets
    axs[0].set_xlabel("x")
    axs[0].set_ylabel("y")
    fig.colorbar(cs, ax=axs[0])

    # Wykres 2: Normalized rank heatmap
    hm = axs[1].imshow(norm_ranks, extent=[x_min, x_max, y_min, y_max],
                       origin='lower', cmap='viridis', aspect='auto')
    axs[1].set_title("Normalized Objective Heatmap")
    axs[1].set_xlabel("x")
    axs[1].set_ylabel("y")
    fig.colorbar(hm, ax=axs[1])

    plt.tight_layout()
    func_name = func.name().replace(' ', '_').replace('-', '_')

    benchmark_dir = 'benchmark'
    make_dir(benchmark_dir)
    filename = f"{benchmark_dir}/{func_name}_heatmaps.png"

    plt.savefig(filename, dpi=300)
    plt.show()

def plot_benchmark_combined(func, resolution=100):
    """
    Generuje trzy wykresy dla funkcji benchmarkowej:
    1. 3D surface plot
    2. Isovalue Heatmap (konturowy)
    3. Normalized Objective Heatmap
    """
    d = func.number_of_variables()
    if d < 2:
        raise ValueError("Funkcja musi mieć przynajmniej 2 zmienne do rysowania.")

    # Ustalamy zakresy dla pierwszych dwóch zmiennych
    x_min, x_max = func.lower_bound[0], func.upper_bound[0]
    y_min, y_max = func.lower_bound[1], func.upper_bound[1]
    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    X, Y = np.meshgrid(x_vals, y_vals)  # Use capital X, Y for meshgrid output
    Z = np.zeros_like(X)

    # Dla d > 2 ustalamy pozostałe zmienne na ich wartość środkową
    fixed_vars = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed_vars.append(midpoint)

    # Ewaluujemy funkcję dla każdego punktu na siatce
    for i in range(resolution):
        for j in range(resolution):
            # Pierwsze dwie zmienne zmieniają się na siatce, pozostałe stałe
            current_vars = [X[i, j], Y[i, j]] + fixed_vars
            sol = FloatSolution(
                lower_bound=func.lower_bound,
                upper_bound=func.upper_bound,
                number_of_objectives=func.number_of_objectives(),
                number_of_constraints=func.number_of_constraints()
            )
            sol.variables = current_vars
            func.evaluate(sol)
            Z[i, j] = sol.objectives[0]

    # Obliczamy normalized rank – porządkujemy wartości funkcji na siatce
    Z_flat = Z.flatten()
    # Używamy argsort dwukrotnie, aby uzyskać ranki
    ranks = np.argsort(np.argsort(Z_flat))
    norm_ranks = ranks / (len(Z_flat) - 1) if len(Z_flat) > 1 else np.zeros_like(Z_flat)  # Handle single point case
    norm_ranks_reshaped = norm_ranks.reshape(Z.shape)

    # Tworzymy wykresy w jednej figurze (1 wiersz, 3 kolumny)
    fig = plt.figure(figsize=(18, 6))  # Adjusted figsize for 3 plots

    # Wykres 1: 3D Surface Plot
    ax1 = fig.add_subplot(131, projection='3d')
    surf = ax1.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('f(x,y)', labelpad=10)  # Adjusted labelpad
    ax1.set_title(f"{func.name()} - 3D Surface")
    fig.colorbar(surf, ax=ax1, shrink=0.5, aspect=10, pad=0.1)  # Adjusted pad for 3D

    # Wykres 2: Isovalue Heatmap (konturowy)
    ax2 = fig.add_subplot(132)
    cs = ax2.contourf(X, Y, Z, cmap='viridis', levels=50)
    ax2.set_title("Isovalue Heatmap")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    fig.colorbar(cs, ax=ax2, shrink=0.7, aspect=10)  # Adjusted shrink/aspect

    # Wykres 3: Normalized Objective Heatmap
    ax3 = fig.add_subplot(133)
    hm = ax3.imshow(norm_ranks_reshaped, extent=[x_min, x_max, y_min, y_max],
                    origin='lower', cmap='viridis', aspect='auto')
    ax3.set_title("Normalized Objective Heatmap")
    ax3.set_xlabel("x")
    ax3.set_ylabel("y")
    fig.colorbar(hm, ax=ax3, shrink=0.7, aspect=10)  # Adjusted shrink/aspect

    plt.tight_layout(pad=1.5)  # Add some padding between subplots
    func_name_safe = func.name().replace(' ', '_').replace('-', '_')

    benchmark_dir = 'benchmark_plots'  # Changed directory name slightly to avoid confusion
    make_dir(benchmark_dir)
    filename = f"{benchmark_dir}/{func_name_safe}_combined_plots.png"

    plt.savefig(filename, dpi=300)
    print(f"Saved combined plot to {filename}")
    plt.show()


def plot_all_benchmarks():
    number_of_variables = 2

    functions_to_plot = [

        # RotatedHighConditionedElliptic(number_of_variables),
        # RotatedBentCigar(number_of_variables),
        # RotatedDiscus(number_of_variables),
        # ShiftedSchwefel(number_of_variables),
        # ShiftedRotatedHappyCat(number_of_variables),
        # ShiftedRotatedHGBat(number_of_variables),
        HappyCat(number_of_variables),
        HGBat(number_of_variables),
        # ShiftedRotatedWeierstrass(number_of_variables),
        # ShiftedRotatedExpandedScafferF6(number_of_variables),
        ##
        # AlpineN1(number_of_variables),
        # CrownedCross(number_of_variables),
        # EggHolder(number_of_variables),
        # ExpandedShaffer(number_of_variables),
        # GeneralizedSchafferN1(number_of_variables),
        # GeneralizedSchafferN2(number_of_variables),
        # GeneralizedSchafferN3(number_of_variables),
        # GeneralizedSchafferN4(number_of_variables),
        # GeneralizedSchmidtVetters(number_of_variables),
        # LennardJonesMinimumEnergyCluster(number_of_variables),
        # Michalewicz(number_of_variables),
        # Mishra03(number_of_variables),
        # Mishra04(number_of_variables),
        # RosenbrockModified02(number_of_variables),
        # Salomon(number_of_variables),
        # SchwefelN20(number_of_variables),
        # SchwefelN36(number_of_variables),
        # SchwefelN6(number_of_variables),
        # ShubertN3(number_of_variables),
        # ShubertN4(number_of_variables),
        # SineEnvelope(number_of_variables),
        # Stochastic(number_of_variables),
        # StretchedV(number_of_variables),
        # StyblinskiTang(number_of_variables),


    ]

    for func in functions_to_plot:
        plot_benchmark_function_truly_maximized(func)        # plot_benchmark_plots(func)
        # plot_benchmark_combined(func)


if __name__ == "__main__":
    plot_all_benchmarks()
