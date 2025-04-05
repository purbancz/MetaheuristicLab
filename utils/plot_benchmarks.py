import numpy as np
import matplotlib.pyplot as plt
from bs4.diagnose import benchmark_parsers
# from mpl_toolkits.mplot3d import Axes3D
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin

from experiment.setup import make_dir
from problem.n_variables.CEC import RotatedHighConditionedElliptic, RotatedBentCigar, RotatedDiscus, \
    ShiftedRotatedRosenbrock, ShiftedRotatedAckley, ShiftedRastrigin, ShiftedRotatedRastrigin, ShiftedSchwefel, \
    ShiftedRotatedSchwefel, ShiftedRotatedKatsuura, ShiftedRotatedHappyCat, ShiftedRotatedHGBat, \
    ShiftedRotatedExpandedGriewankPlusRosenbrock, ShiftedRotatedExpandedScafferF6, HybridFunction1, HybridFunction2, \
    HybridFunction3, HybridFunction4, HybridFunction5, HybridFunction6, CompositionFunction1, CompositionFunction2, \
    CompositionFunction3, CompositionFunction4, CompositionFunction5, CompositionFunction6, CompositionFunction7, \
    CompositionFunction8
from problem.n_variables.ackley import Ackley
from problem.n_variables.alpine import AlpineN1
from problem.n_variables.bent_cigar import BentCigar
from problem.n_variables.discus import Discus
from problem.n_variables.dixon import DixonPrice
from problem.n_variables.eggholder import EggHolder
from problem.n_variables.expanded_schaffer import ExpandedShaffer
from problem.n_variables.griewank import Griewank
from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster
from problem.n_variables.levy import Levy
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.penalized import GeneralizedPenalizedN1
from problem.n_variables.quartic import Quartic
from problem.n_variables.rosenbrock import Rosenbrock
from problem.n_variables.salomon import Salomon
from problem.n_variables.schwefel import Schwefel
from problem.n_variables.step import StepN1, StepN2, StepN3
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

    # Dla wymiarów >2, ustawiamy wartość środkową (midpoint) dla każdej zmiennej
    fixed = []
    for i in range(2, d):
        midpoint = (func.lower_bound[i] + func.upper_bound[i]) / 2.0
        fixed.append(midpoint)

    # Dla każdego punktu na siatce, budujemy pełny wektor zmiennych
    for i in range(resolution):
        for j in range(resolution):
            # Pierwsze dwie zmienne ze siatki, pozostałe stałe
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


def plot_all_benchmarks():
    number_of_variables = 2

    functions_to_plot = [
        Zakharov(number_of_variables),
        Rosenbrock(number_of_variables),
        Rastrigin(number_of_variables),
        Sphere(number_of_variables),
        Quartic(number_of_variables),
        AlpineN1(number_of_variables),
        EggHolder(number_of_variables),
        DixonPrice(number_of_variables),
        Salomon(number_of_variables),
        GeneralizedPenalizedN1(number_of_variables),
        StepN1(number_of_variables),
        # StepN2(number_of_variables),
        StepN3(number_of_variables),
        StyblinskiTang(number_of_variables),
        Ackley(number_of_variables),
        Griewank(number_of_variables),
        Levy(number_of_variables),
        Michalewicz(number_of_variables),
        Schwefel(number_of_variables),

        LennardJonesMinimumEnergyCluster(number_of_variables),
        BentCigar(number_of_variables),
        ExpandedShaffer(number_of_variables),
        Discus(number_of_variables),
        ShiftedRotatedWeierstrass(number_of_variables),

        RotatedHighConditionedElliptic(number_of_variables),
        RotatedBentCigar(number_of_variables),
        RotatedDiscus(number_of_variables),

        # Group B: Simple Multimodal Functions
        ShiftedRotatedRosenbrock(number_of_variables),
        ShiftedRotatedAckley(number_of_variables),
        ShiftedRastrigin(number_of_variables),
        ShiftedRotatedRastrigin(number_of_variables),
        ShiftedSchwefel(number_of_variables),
        ShiftedRotatedSchwefel(number_of_variables),
        ShiftedRotatedKatsuura(number_of_variables),
        ShiftedRotatedHappyCat(number_of_variables),
        ShiftedRotatedHGBat(number_of_variables),
        ShiftedRotatedExpandedGriewankPlusRosenbrock(number_of_variables),
        ShiftedRotatedExpandedScafferF6(number_of_variables),

        # Group C: Hybrid Functions
        HybridFunction1(number_of_variables),
        HybridFunction2(4),
        HybridFunction3(number_of_variables),
        HybridFunction4(5),
        HybridFunction5(10),
        HybridFunction6(10),

        # Group D: Composition Functions
        CompositionFunction1(30),
        CompositionFunction2(30),
        CompositionFunction3(30),
        CompositionFunction4(30),
        CompositionFunction5(30),
        CompositionFunction6(30),
        CompositionFunction7(30),
        CompositionFunction8(30)
    ]

    for func in functions_to_plot:
        plot_benchmark_function(func)


if __name__ == "__main__":
    plot_all_benchmarks()