import os

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover, DifferentialEvolutionCrossover
from jmetal.problem import Sphere, Srinivas
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.AdaptivePSO import GlobalAdaptivePSO, LocalAdaptivePSO
from algorithm.DifferentialEvolution import DifferentialEvolution
from algorithm.FAPSO import FAPSO
from algorithm.GradientEnhancedPSO import GradientEnhancedPSO
from algorithm.HybridPSODE import HybridPSODE
from algorithm.LightningPSO import LightningPSO
from algorithm.NPSO import NPSO
from algorithm.PGCHEA import PGCHEA
from algorithm.PGPHEA import PGPHEA
from algorithm.PGSHEA import PGSHEA
from algorithm.QTPSO import QTPSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO
from algorithm.particles_with_roles import RebelPSO, EscapistPSO, RebelEscapistPSO, REAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO
from algorithm.single_objective_PSO import SingleObjectivePSO

from problem.fixed_varaibles.branin import BraninRCOC
from problem.fixed_varaibles.camel import SixHumpCamel, ThreeHumpCamel
from problem.fixed_varaibles.cross_in_tray import CrossInTray
from problem.fixed_varaibles.de_joung import DeJoung
from problem.fixed_varaibles.drop_wave import DropWave
from problem.fixed_varaibles.easom import Easom
from problem.fixed_varaibles.goldstein_price import GoldsteinPrice
from problem.fixed_varaibles.hartmann import Hartmann
from problem.fixed_varaibles.holder_table import HolderTable
from problem.fixed_varaibles.mccormick import McCormick
from problem.fixed_varaibles.schaffer import SchafferN2
from problem.fixed_varaibles.shekel import Shekel
from problem.fixed_varaibles.shubert import Shubert
from problem.n_variables.ackley import Ackley
from problem.n_variables.alpine import AlpineN1
from problem.n_variables.dixon import DixonPrice
from problem.n_variables.eggholder import EggHolder
from problem.n_variables.griewank import Griewank
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


def setup_experiment():
    no_of_runs = 10
    number_of_variables = 200
    solutions_size = 100
    max_evaluations = 25001
    frequency = solutions_size  # Snapshot each generation

    algorithm_colors = {
        'GA': 'blue',
        'PSO': 'orange',
        'PGPHEA': 'purple',
        'PGSHEA': 'green',
        'PGCHEA': 'red',
        'RebelPSO': 'cyan',
        'EscapistPSO': 'magenta',
        'RebelEscapistPSO': 'brown',
        'REAPSO': 'pink',
        'GradientEnhancedPSO': 'yellow',
        'LightningPSO': 'gray',
        'QTPSO': 'olive',
        'SPPPSO': 'black',
        'TDPSO': 'teal',
        'NPSO': 'maroon',
        'FAPSO': 'navy',
        'ReverseLearningGlobalAttractorPSO': 'lime',
        'ReverseLearningPersonalAttractorPSO': 'deepskyblue',
        'CombinedLearningPSO': 'lavender',
        'ContrarianPSO': 'darkgreen',
        'DefeatistPSO': 'darkcyan',
        'ContrarianDefeatistPSO': 'darkblue',
        'DE': 'gold',
        'HybridPSODE': 'turquoise',
        'GlobalAdaptivePSO': 'xkcd:lemon',
        'LocalAdaptivePSO': 'xkcd:camouflage green'
    }

    results_dir = 'experiment_results'
    make_dir(results_dir)

    # Define problems
    n_variables_problems = [
        # Zakharov(number_of_variables),
        # Rosenbrock(number_of_variables),
        ##
        Rastrigin(number_of_variables),
        # Sphere(number_of_variables),
        # Quartic(number_of_variables),
        # AlpineN1(number_of_variables),
        # EggHolder(number_of_variables),
        # DixonPrice(number_of_variables),
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
        # Schwefel(number_of_variables),
        # ShiftedRotatedWeierstrass(number_of_variables),
        ##
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
        # CrossInTray(),
        # DropWave(),
        # HolderTable(),
        # SixHumpCamel(),
        # ThreeHumpCamel(),
        # McCormick(),
    ]

    problems = n_variables_problems + fixed_variables_problems

    # Initialize the algorithms
    algorithms = {
        # 'GA': lambda p: GeneticAlgorithm(
        #     problem=p,
        #     population_size=solutions_size,
        #     offspring_population_size=solutions_size,
        #     mutation=PolynomialMutation(1.0 / p.number_of_variables(), 20.0),
        #     crossover=SBXCrossover(0.75, 5.0),
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        # ),
        'PSO': lambda p: SingleObjectivePSO(
            problem=p,
            swarm_size=solutions_size,
            c1=1.132464,
            c2=4.489647,
            w=0.110646,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        # 'DE': lambda p: DifferentialEvolution(
        #     problem=p,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        #     swarm_size = solutions_size,
        #     crossover_operator = DifferentialEvolutionCrossover(CR=0.9, F=0.5),
        # ),
        # 'HybridPSODE': lambda p: HybridPSODE(
        #     problem=p,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        #     swarm_size=solutions_size,
        #     b1=1.132464,
        #     b2=4.489647,
        #     w=0.110646,
        #     de_probability = 0.5,
        #     crossover_operator=DifferentialEvolutionCrossover(CR=0.9, F=0.5),
        # ),
        # 'PGSHEA': lambda p: PGSHEA(
        #     problem=p,
        #     solutions_size=solutions_size,
        #     mutation=PolynomialMutation(0.38 / p.number_of_variables(), 20.0),
        #     crossover=SBXCrossover(1, 5.0),
        #     swap_interval=13,  # int(max_evaluations/(2 * solutions_size))
        #     b1=2.63,
        #     b2=0.21,
        #     w=0.01,
        #     starting_algorithm='PSO',
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'PGPHEA': lambda p: PGPHEA(
        #     problem=p,
        #     solutions_size=solutions_size,
        #     mutation=PolynomialMutation(0.37 / p.number_of_variables(), 20.0),
        #     crossover=SBXCrossover(1, 5.0),
        #     exchange_interval=13,
        #     exchange_number=7,  # 11
        #     b1=0.00001,
        #     b2=0.26,
        #     w=0.17,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'PGCHEA': lambda p: PGCHEA(
        #     problem=p,
        #     solutions_size=solutions_size,
        #     mutation=PolynomialMutation(0.61 / p.number_of_variables(), 20.0),
        #     crossover=SBXCrossover(1, 5.0),
        #     b1=1.85,
        #     b2=0.5,
        #     w=1.53,
        #     starting_algorithm='PSO',
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # # New PSO variants
        # 'RebelPSO': lambda p: RebelPSO( # rough tuning
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=0.3077,
        #     c2=5.5310,
        #     ac2=3.9234,
        #     w=0.1308,
        #     rebel_fraction=0.18,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # 'EscapistPSO': lambda p: EscapistPSO( # rough tuning
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=1.5453,
        #     c2=5.9175,
        #     ac1=0.4661,
        #     w=0.0839,
        #     escapist_fraction=0.68,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # 'RebelEscapistPSO': lambda p: RebelEscapistPSO( # rough tuning
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=1.2917,
        #     c2=4.8326,
        #     ac1=1.4414,
        #     ac2=3.7445,
        #     w=0.1155,
        #     rebel_fraction=0.19,
        #     escapist_fraction=0.19,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # 'REAPSO': lambda p: REAPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=1.5510184332980186,
        #     c2=4.935325731217671,
        #     ac1=1.5510184332980186,
        #     ac2=4.935325731217671,
        #     base_inertia=0.214141581688782,
        #     min_inertia=0.11093829549932,
        #     max_inertia=0.935915518894973,
        #     rebel_fraction=0.2,
        #     escapist_fraction=0.43,
        #     window_size = 20,
        #     perturbation_probability = 0.460269994559271,
        #     perturbation_scale = 0.709878890732096,
        #     max_rebel_fraction= 0.79,
        #     max_escapist_fraction= 0.56,
        #     diversity_threshold = 0.058518962214864,
        #     improvement_threshold = 0.008656759607128,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # 'ReverseLearningGlobalAttractorPSO': lambda p: ReverseLearningGlobalAttractorPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     a = 1,
        #     b1=1.132464,
        #     b2=4.489647,
        #     w=0.110646,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'ReverseLearningPersonalAttractorPSO': lambda p: ReverseLearningPersonalAttractorPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     a=1,
        #     b1=1.132464,
        #     b2=4.489647,
        #     w=0.110646,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'CombinedLearningPSO': lambda p: CombinedLearningPSO(problem=p,
        #     swarm_size=solutions_size,
        #     c1=1.132464,
        #     c2=4.489647,
        #     b1=0.132464,
        #     b2=0.489647,
        #     w=0.110646,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)),
        # 'ContrarianPSO': lambda p: ContrarianPSO( # might be good
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=2.9291453027606287,
        #     c2=2.899888793334852,
        #     ac2=2.899888793334852,
        #     w=0.06742018332897903,
        #     contrarian_fraction=0.2,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # 'DefeatistPSO': lambda p: DefeatistPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=2.49,
        #     c2=2.47,
        #     ac1=2.49,
        #     w=0.15,
        #     defeatist_fraction=0.21,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # 'ContrarianDefeatistPSO': lambda p: ContrarianDefeatistPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=2.5,
        #     c2=2.5,
        #     ac1=2.5,
        #     ac2=2.5,
        #     w=0.1,
        #     contrarian_fraction=0.05,
        #     defeatist_fraction=0.05,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        # "GradientEnhancedPSO": lambda p: GradientEnhancedPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=1.97,
        #     b2=0.94,
        #     c3=1.97,
        #     w=0.56,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # "LightningPSO": lambda p: LightningPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=2.5,
        #     b2=2.5,
        #     c3=0.3,
        #     w=0.1,
        #     dim_sample=0.5,
        #     grad_sample=0.1,
        #     active_ratio=0.3,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
        # ),
        # 'QTPSO': lambda p: QTPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=0.9441842367886241,
        #     b2=5.4875414623505385,
        #     w=0.08830337945791762,
        #     quantum_prob=0.1,
        #     chaos_strength=0.05,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'SPPPSO': lambda p: SPPPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=0.9441842367886241,
        #     b2=5.4875414623505385,
        #     w=0.08830337945791762,
        #     predator_ratio=0.05,
        #     scavenger_ratio=0.2,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'TDPSO': lambda p: TDPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=0.9441842367886241,
        #     b2=5.4875414623505385,
        #     w=0.08830337945791762,
        #     temperature = 1.0,
        #     cooling_rate = 0.99,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'NPSO': lambda p: NPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=0.025515929851215,
        #     c2=2.251249703372387,
        #     w=0.056059595444136,
        #     spike_threshold = 0.963856605654984,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        # 'FAPSO': lambda p: FAPSO(
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=0.9441842367886241,
        #     b2=5.4875414623505385,
        #     w=0.08830337945791762,
        #     fractal_depth=3,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        'GlobalAdaptivePSO': lambda p: GlobalAdaptivePSO(
            problem=p,
            swarm_size=solutions_size,
            c1=0.025515929851215,
            c2=2.251249703372387,
            max_c1=7,
            max_c2=7,
            w=0.056059595444136,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'LocalAdaptivePSO': lambda p: LocalAdaptivePSO(
            problem=p,
            swarm_size=solutions_size,
            c1=0.025515929851215,
            c2=2.251249703372387,
            max_c1=7,
            max_c2=7,
            w=0.056059595444136,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
    }

    return (algorithms, problems, no_of_runs, number_of_variables, solutions_size,
            max_evaluations, frequency, algorithm_colors, results_dir)


def initialize_algorithms(algorithms, problem):
    return {name: algo(problem) for name, algo in algorithms.items()}


def make_dir(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
