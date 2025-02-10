import os

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.PGCHEA import PGCHEA
from algorithm.PGPHEA import PGPHEA
from algorithm.PGSHEA import PGSHEA
from algorithm.single_objective_PSO import SingleObjectivePSO, RebelPSO, EscapistPSO, EscapistRebelPSO

from problem.fixed_varaibles.branin import BraninRCOC
from problem.fixed_varaibles.de_joung import DeJoung
from problem.fixed_varaibles.easom import Easom
from problem.fixed_varaibles.goldstein_price import GoldsteinPrice
from problem.fixed_varaibles.hartmann import Hartmann
from problem.fixed_varaibles.schaffer import SchafferN2
from problem.fixed_varaibles.shekel import Shekel
from problem.fixed_varaibles.shubert import Shubert
from problem.n_variables.ackley import Ackley
from problem.n_variables.griewank import Griewank
from problem.n_variables.levy import Levy
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.rosenbrock import Rosenbrock
from problem.n_variables.schwefel import Schwefel
from problem.n_variables.weierstrass import ShiftedRotatedWeierstrass
from problem.n_variables.zakharov import Zakharov


def setup_experiment():
    no_of_runs = 10
    number_of_variables = 10
    solutions_size = 100
    max_evaluations = 25000
    frequency = solutions_size  # Snapshot each generation

    algorithm_colors = {
        'GA': 'blue',
        'PSO': 'orange',
        'PGPHEA': 'purple',
        'PGSHEA': 'green',
        'PGCHEA': 'red',
        'RebelPSO': 'cyan',
        'EscapistPSO': 'magenta',
        'EscapistRebelPSO': 'brown'
    }

    results_dir = 'experiment_results'
    make_dir(results_dir)

    # Define problems
    n_variables_problems = [
        # Zakharov(number_of_variables),
        # Rosenbrock(number_of_variables),
        # ##
        Rastrigin(number_of_variables),
        # Ackley(number_of_variables),
        # Griewank(number_of_variables),
        # Levy(number_of_variables),
        # Michalewicz(number_of_variables),
        # Schwefel(number_of_variables),
        # ShiftedRotatedWeierstrass(number_of_variables),
    ]

    fixed_variables_problems = [
        # BraninRCOC(),
        # DeJoung(),
        # GoldsteinPrice(),
        # Hartmann(),
        # Shubert(),
        # ##
        # SchafferN2(),
        # Shekel(),
        # Easom(),
    ]

    problems = n_variables_problems + fixed_variables_problems

    # Initialize the algorithms
    algorithms = {
        'GA': lambda p: GeneticAlgorithm(
            problem=p,
            population_size=solutions_size,
            offspring_population_size=solutions_size,
            mutation=PolynomialMutation(1.0 / p.number_of_variables(), 20.0),
            crossover=SBXCrossover(0.75, 5.0),
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        ),
        'PSO': lambda p: SingleObjectivePSO(
            problem=p,
            swarm_size=solutions_size,
            c1=1.97,
            c2=0.94,
            w=0.56,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'PGSHEA': lambda p: PGSHEA(
            problem=p,
            solutions_size=solutions_size,
            mutation=PolynomialMutation(0.38 / p.number_of_variables(), 20.0),
            crossover=SBXCrossover(1, 5.0),
            swap_interval=13,  # int(max_evaluations/(2 * solutions_size))
            c1=2.63,
            c2=0.21,
            w=0.01,
            starting_algorithm='PSO',
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'PGPHEA': lambda p: PGPHEA(
            problem=p,
            solutions_size=solutions_size,
            mutation=PolynomialMutation(0.37 / p.number_of_variables(), 20.0),
            crossover=SBXCrossover(1, 5.0),
            exchange_interval=13,
            exchange_number=7,  # 11
            c1=0.00001,
            c2=0.26,
            w=0.17,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'PGCHEA': lambda p: PGCHEA(
            problem=p,
            solutions_size=solutions_size,
            mutation=PolynomialMutation(0.61 / p.number_of_variables(), 20.0),
            crossover=SBXCrossover(1, 5.0),
            c1=1.85,
            c2=0.5,
            w=1.53,
            starting_algorithm='PSO',
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        # New PSO variants
        'RebelPSO': lambda p: RebelPSO(
            problem=p,
            swarm_size=solutions_size,
            c1=1.97,
            c2=0.94,
            w=0.56,
            rebel_fraction=0.1,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'EscapistPSO': lambda p: EscapistPSO(
            problem=p,
            swarm_size=solutions_size,
            c1=1.97,
            c2=0.94,
            w=0.56,
            escapist_fraction=0.1,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'EscapistRebelPSO': lambda p: EscapistRebelPSO(
            problem=p,
            swarm_size=solutions_size,
            c1=1.97,
            c2=0.94,
            w=0.56,
            rebel_fraction=0.1,
            escapist_fraction=0.1,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        )
    }

    return (algorithms, problems, no_of_runs, number_of_variables, solutions_size,
            max_evaluations, frequency, algorithm_colors, results_dir)


def initialize_algorithms(algorithms, problem):
    return {name: algo(problem) for name, algo in algorithms.items()}


def make_dir(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
