import math

import numpy as np
import matplotlib.pyplot as plt
from bs4.diagnose import benchmark_parsers
# from mpl_toolkits.mplot3d import Axes3D
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin

from algorithm.hgbat import HGBat
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
from problem.n_variables.discus import Discus
from problem.n_variables.dixon import DixonPrice, GeneralizedDixonPriceRosenbrock
from problem.n_variables.eggholder import EggHolder
from problem.n_variables.expanded_schaffer import ExpandedShaffer
from problem.n_variables.griewank import Griewank
from problem.n_variables.happy_cat import HappyCat
from problem.n_variables.katsuura import Katsuura, ExpandedKatsuura
from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster
from problem.n_variables.levy import Levy
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.penalized import GeneralizedPenalizedN1
from problem.n_variables.plateau import Plateau
from problem.n_variables.quantum_speed import QuantumSpeedLimit2D, GeneralizedQuantumSpeedLimit
from problem.n_variables.quartic import Quartic
from problem.n_variables.rosenbrock import Rosenbrock, RosenbrockModified01, RosenbrockModified02
from problem.n_variables.salomon import Salomon
from problem.n_variables.schaffer import GeneralizedSchafferN7, GeneralizedSchafferN1, GeneralizedSchafferN2, GeneralizedSchafferN3, GeneralizedSchafferN4
from problem.n_variables.schmidt_vetters import GeneralizedSchmidtVetters
from problem.n_variables.schwefel import SchwefelN26, SchwefelN21, SchwefelN22, SchwefelN6, SchwefelN20, \
    SchwefelN36
from problem.n_variables.step import StepN1, StepN2, StepN3
from problem.n_variables.stochastic import Stochastic
from problem.n_variables.styblinski import StyblinskiTang
from problem.n_variables.weierstrass import ShiftedRotatedWeierstrass
from problem.n_variables.zakharov import Zakharov


def plot_benchmark_function(func, resolution=100):
    d = func.number_of_variables()
    if d < 2:
        raise ValueError("Funkcja musi mieć przynajmniej 2 zmienne do rysowania w 3D.")

    x_min, x_max = func.lower_bound[0], func.upper_bound[0]
    y_min, y_max = func.lower_bound[1], func.upper_bound[1]

    x_vals = np.linspace(x_min, x_max, resolution)
    y_vals = np.linspace(y_min, y_max, resolution)
    x, y = np.meshgrid(x_vals, y_vals)
    z = np.zeros_like(x)

    fixed = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed.append(midpoint)

    for i in range(resolution):
        for j in range(resolution):
            variables = [x[i, j], y[i, j]] + fixed
            sol = FloatSolution(
                lower_bound=func.lower_bound,
                upper_bound=func.upper_bound,
                number_of_objectives=func.number_of_objectives(),
                number_of_constraints=func.number_of_constraints()
            )
            sol.variables = variables
            func.evaluate(sol)
            z[i, j] = sol.objectives[0]

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(x, y, z, cmap='viridis', edgecolor='none')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('f(x,y)', labelpad=5)
    ax.set_title(func.name())
    fig.colorbar(surf, shrink=0.5, aspect=5, pad=0.1)

    plt.tight_layout()
    func_name = func.name().replace(' ', '_').replace('-', '_')

    benchmark_dir = 'benchmark'
    make_dir(benchmark_dir)
    filename = f"{benchmark_dir}/{func_name}.png"

    plt.savefig(filename, dpi=300)
    plt.show()


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
    axs[0].set_title("Isovalue Heatmap") # Scalar Field Map, Level Sets
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




def plot_all_benchmarks():
    number_of_variables = 2

    functions_to_plot = [
        # Zakharov(number_of_variables),
        # Plateau(number_of_variables),
        # Rosenbrock(number_of_variables),
        # RosenbrockModified01(number_of_variables),
        # RosenbrockModified02(number_of_variables),
        # Rastrigin(number_of_variables),
        # Sphere(number_of_variables),
        # Quartic(number_of_variables),
        # AlpineN1(number_of_variables),
        # AlpineN2(number_of_variables),
        # EggHolder(number_of_variables),
        # DixonPrice(number_of_variables),
        # GeneralizedDixonPriceRosenbrock(number_of_variables),
        # Katsuura(number_of_variables),
        # ExpandedKatsuura(number_of_variables),
        # HappyCat(number_of_variables),
        # HGBat(number_of_variables),
        # GeneralizedSchafferN7(number_of_variables),
        # GeneralizedSchafferN1(number_of_variables),
        # GeneralizedSchafferN2(number_of_variables),
        # GeneralizedSchafferN3(number_of_variables),
        # GeneralizedSchafferN4(number_of_variables),
        # GeneralizedSchmidtVetters(number_of_variables),
        # Stochastic(number_of_variables),
        # Salomon(number_of_variables),
        # GeneralizedPenalizedN1(number_of_variables),
        # StepN1(number_of_variables),
        # StepN2(number_of_variables),
        # StepN3(number_of_variables),
        # StyblinskiTang(number_of_variables),
        # Ackley(number_of_variables),
        # Griewank(number_of_variables),
        # Levy(number_of_variables),
        # Michalewicz(number_of_variables),
        # SchwefelN26(number_of_variables),
        # SchwefelN21(number_of_variables),
        # SchwefelN22(number_of_variables),
        # SchwefelN6(number_of_variables),
        # SchwefelN20(number_of_variables),
        # SchwefelN36(number_of_variables),
        # LennardJonesMinimumEnergyCluster(number_of_variables),
        # BentCigar(number_of_variables),
        # ExpandedShaffer(number_of_variables),
        # Discus(number_of_variables),
        # ##
        # RotatedHighConditionedElliptic(number_of_variables),
        # RotatedBentCigar(number_of_variables),
        # RotatedDiscus(number_of_variables),
        # ShiftedRotatedRosenbrock(number_of_variables),
        # ShiftedRotatedAckley(number_of_variables),
        # ShiftedRastrigin(number_of_variables),
        # ShiftedRotatedRastrigin(number_of_variables),
        # ShiftedSchwefel(number_of_variables),
        # ShiftedRotatedSchwefel(number_of_variables),
        # ShiftedRotatedKatsuura(number_of_variables),
        # ShiftedRotatedHappyCat(number_of_variables),
        # ShiftedRotatedHGBat(number_of_variables),
        # ShiftedRotatedSchafferF7(number_of_variables),
        # ShiftedRotatedWeierstrass(number_of_variables),
        # ShiftedRotatedExpandedGriewankPlusRosenbrock(number_of_variables),
        # ShiftedRotatedExpandedScafferF6(number_of_variables),
        # ##
        #
        # # Group C: Hybrid Functions
        # HybridFunction1(number_of_variables),
        # HybridFunction2(4),
        # HybridFunction3(number_of_variables),
        # HybridFunction4(5),
        # HybridFunction5(10),
        # HybridFunction6(10),
        #
        # # Group D: Composition Functions
        # CompositionFunction1(30),
        # CompositionFunction2(30),
        # CompositionFunction3(30),
        # CompositionFunction4(30),
        # CompositionFunction5(30),
        # CompositionFunction6(30),
        # CompositionFunction7(30),
        # CompositionFunction8(30)
        # QSLTimeBoundProblem(),
        QuantumSpeedLimit2D(),
        # GeneralizedQuantumSpeedLimit(2)
    ]

    for func in functions_to_plot:
        plot_benchmark_function(func)
        plot_benchmark_plots(func)



if __name__ == "__main__":
    plot_all_benchmarks()