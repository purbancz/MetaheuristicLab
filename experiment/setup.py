import os

from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.operator import PolynomialMutation, SBXCrossover, DifferentialEvolutionCrossover
from jmetal.problem import Sphere, Srinivas
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.AdaptivePSO import GlobalAdaptivePSO, PersonalAdaptivePSO
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
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO, \
    ReverseLearningPSO
from algorithm.SPPPSO import SPPPSO
from algorithm.TDPSO import TDPSO
from algorithm.hgbat import HGBat
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, CDAPSO, EEAPSO
from algorithm.ref_DCSPSO import DCSPSO
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
from problem.n_variables.cross import CrownedCross, CrossLeggedTable, Cross, GeneralizedCrossInTray
from problem.n_variables.discus import Discus
from problem.n_variables.dixon import DixonPrice, GeneralizedDixonPriceRosenbrock
from problem.n_variables.eggholder import EggHolder
from problem.n_variables.expanded_schaffer import ExpandedShaffer
from problem.n_variables.griewank import Griewank
from problem.n_variables.happy_cat import HappyCat
from problem.n_variables.holders import TestTubeHolder, CarromTable, PenHolder, GeneralizedHolderTable
from problem.n_variables.katsuura import Katsuura, ExpandedKatsuura
from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster
from problem.n_variables.levy import Levy
from problem.n_variables.michalewicz import Michalewicz
from problem.n_variables.mishra import Mishra01, Mishra02, Mishra03, Mishra04, Mishra05, Mishra06, Mishra11
from problem.n_variables.penalized import GeneralizedPenalizedN1
from problem.n_variables.plateau import Plateau
from problem.n_variables.quantum_speed import QuantumSpeedLimit2D
from problem.n_variables.quartic import Quartic
from problem.n_variables.rosenbrock import Rosenbrock, RosenbrockModified01, RosenbrockModified02
from problem.n_variables.salomon import Salomon
from problem.n_variables.schaffer import GeneralizedSchafferN7, GeneralizedSchafferN1, GeneralizedSchafferN3, \
    GeneralizedSchafferN4, GeneralizedSchafferN2
from problem.n_variables.schmidt_vetters import GeneralizedSchmidtVetters
from problem.n_variables.schwefel import SchwefelN26, SchwefelN21, SchwefelN22, SchwefelN6, SchwefelN20, SchwefelN36
from problem.n_variables.shubert import ShubertN1, ShubertN3, ShubertN4
from problem.n_variables.sine_envelope import SineEnvelope
from problem.n_variables.step import StepN1, StepN2, StepN3
from problem.n_variables.stochastic import Stochastic
from problem.n_variables.strechedv import StretchedV
from problem.n_variables.styblinski import StyblinskiTang
from problem.n_variables.weierstrass import ShiftedRotatedWeierstrass
from problem.n_variables.zakharov import Zakharov


def setup_experiment():
    no_of_runs = 50
    number_of_variables = 100
    solutions_size = 100
    max_evaluations = 25000
    frequency = solutions_size  # Snapshot each generation

    algorithm_colors = {
        # 'GA': 'xkcd:bright blue',
        'PSO': 'xkcd:bright red',
        # 'PGPHEA': 'xkcd:violet',
        # 'PGSHEA': 'xkcd:dark green',
        # 'PGCHEA': 'xkcd:rose',
        'RebelPSO': 'xkcd:green',
        'RejectorPSO': 'xkcd:dark indigo',
        'RebelRejectorPSO': 'xkcd:sienna',
        'RRAPSO': 'xkcd:violet',
        'CDAPSO': 'xkcd:burgundy',
        'EEAPSO': 'xkcd:orange',
        # 'GradientEnhancedPSO': 'xkcd:sunflower yellow',
        # 'LightningPSO': 'xkcd:grey blue',
        # 'QTPSO': 'xkcd:olive green',
        # 'SPPPSO': 'xkcd:charcoal',
        # 'TDPSO': 'xkcd:teal',
        # 'NPSO': 'xkcd:burgundy',
        'FAPSO': 'xkcd:teal',
        'ReverseLearningPSO': 'xkcd:yellow',
        'ReverseLearningPSO_with_bounce': 'xkcd:dirty yellow',
        'ReverseLearningGlobalAttractorPSO': 'xkcd:lime green',
        'ReverseLearningPersonalAttractorPSO': 'xkcd:cyan',
        'CombinedLearningPSO': 'xkcd:fuchsia',
        'ContrarianPSO': 'xkcd:cranberry',
        'DefeatistPSO': 'xkcd:sky blue',
        'ContrarianDefeatistPSO': 'xkcd:dark navy',
        # 'DE': 'xkcd:goldenrod',
        # 'HybridPSODE': 'xkcd:turquoise',
        'GlobalAdaptivePSO': 'xkcd:goldenrod',
        'PersonalAdaptivePSO': 'xkcd:charcoal',
        'EschewerPSO': 'xkcd:pea green',
        'EscapistPSO': 'xkcd:marine',
        'EschewerEscapistPSO': 'xkcd:dark magenta',
        ###
        'DCS-PSO': 'xkcd:rich purple',
    }

    results_dir = 'experiment_results'
    make_dir(results_dir)

    # Define problems
    n_variables_problems = [
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
        # ShiftedRotatedHappyCat(number_of_variables),
        # ShiftedRotatedHGBat(number_of_variables),
        # ShiftedRotatedSchafferF7(number_of_variables),
        # ShiftedRotatedWeierstrass(number_of_variables),
        # ShiftedRotatedExpandedGriewankPlusRosenbrock(number_of_variables),
        # ShiftedRotatedExpandedScafferF6(number_of_variables), #15
        # ##
        # AlpineN1(number_of_variables),
        # AlpineN2(number_of_variables),
        # BentCigar(number_of_variables),
        # Bird(number_of_variables),
        # CarromTable(number_of_variables),
        # Cross(number_of_variables),
        # CrossLeggedTable(number_of_variables),
        # CrownedCross(number_of_variables),
        # Discus(number_of_variables),
        # DixonPrice(number_of_variables),
        # EggHolder(number_of_variables),
        # ExpandedShaffer(number_of_variables), # 28
        # GeneralizedCrossInTray(number_of_variables), # 29 24h
        # GeneralizedDixonPriceRosenbrock(number_of_variables),
        # GeneralizedHolderTable(number_of_variables),
        # GeneralizedPenalizedN1(number_of_variables),
        # GeneralizedSchafferN1(number_of_variables),
        # GeneralizedSchafferN2(number_of_variables),
        # GeneralizedSchafferN3(number_of_variables),
        # GeneralizedSchafferN4(number_of_variables),
        # GeneralizedSchafferN7(number_of_variables),
        # GeneralizedSchmidtVetters(number_of_variables),
        # Griewank(number_of_variables),
        # HappyCat(number_of_variables),
        # HGBat(number_of_variables),
        # LennardJonesMinimumEnergyCluster(number_of_variables),
        # Levy(number_of_variables),
        # Michalewicz(number_of_variables),
        # Mishra01(number_of_variables),
        # Mishra02(number_of_variables),
        # Mishra03(number_of_variables),
        # Mishra04(number_of_variables),
        # Mishra05(number_of_variables),
        # Mishra06(number_of_variables),
        Mishra11(number_of_variables),
        PenHolder(number_of_variables),
        Plateau(number_of_variables),
        Quartic(number_of_variables),
        Rosenbrock(number_of_variables),
        RosenbrockModified01(number_of_variables),
        RosenbrockModified02(number_of_variables),
        Salomon(number_of_variables),
        SchwefelN20(number_of_variables),
        SchwefelN21(number_of_variables),
        SchwefelN22(number_of_variables),
        SchwefelN26(number_of_variables),
        SchwefelN36(number_of_variables),
        SchwefelN6(number_of_variables),
        ShubertN1(number_of_variables),
        ShubertN3(number_of_variables),
        ShubertN4(number_of_variables),
        SineEnvelope(number_of_variables),
        StepN1(number_of_variables),
        StepN2(number_of_variables),
        StepN3(number_of_variables),
        Stochastic(number_of_variables),
        StretchedV(number_of_variables),
        StyblinskiTang(number_of_variables),
        TestTubeHolder(number_of_variables),
        Zakharov(number_of_variables),

        # HybridFunction1(number_of_variables),
        # HybridFunction2(number_of_variables),
        # HybridFunction3(number_of_variables),
        # HybridFunction4(number_of_variables),
        # HybridFunction5(number_of_variables),
        # HybridFunction6(number_of_variables),
        # CompositionFunction1(number_of_variables),
        # CompositionFunction2(number_of_variables),
        # CompositionFunction3(number_of_variables),
        # CompositionFunction4(number_of_variables),
        # CompositionFunction5(number_of_variables),
        # CompositionFunction6(number_of_variables),
        # CompositionFunction7(number_of_variables),
        # CompositionFunction8(number_of_variables),

        # ShiftedRotatedKatsuura(number_of_variables), # too long
        # ExpandedKatsuura(number_of_variables), # too long
        # Katsuura(number_of_variables),  # too long
        # Ackley(number_of_variables), # irace
        # Rastrigin(number_of_variables), # irace
        # Sphere(number_of_variables), # irace

    ]

    fixed_variables_problems = [
        # QuantumSpeedLimit2D(),
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
        ## baseline
        'PSO': lambda p: SingleObjectivePSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=4.3732,
            c2=2.7552,
            w=0.0632,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ## Rebel algorithms
        'RebelPSO': lambda p: RebelPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.3077,
            c2=5.5310,
            ac2=3.9234,
            w=0.1308,
            rebel_fraction=0.18,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'RejectorPSO': lambda p: RejectorPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=1.5453,
            c2=5.9175,
            ac1=0.4661,
            w=0.0839,
            rejector_fraction=0.68,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'RebelRejectorPSO': lambda p: RebelRejectorPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=1.2917,
            c2=4.8326,
            ac1=1.4414,
            ac2=3.7445,
            w=0.1155,
            rebel_fraction=0.19,
            rejector_fraction=0.19,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        # 'RRAPSO': lambda p: RRAPSO(  # clip rough tuning 1st and slighteky better
        #     problem=p,
        #     swarm_size=solutions_size,
        #     c1=0.5695,
        #     c2=5.4892,
        #     ac1=4.2746,
        #     ac2=0.5587,
        #     base_inertia=0.1050,
        #     min_inertia=0.1220,
        #     max_inertia=0.3423,
        #     rebel_fraction=0.0533,
        #     rejector_fraction=0.0799,
        #     window_size=30,
        #     # perturbation_probability=0.4752,
        #     # perturbation_scale=0.9648,
        #     max_rebel_fraction=0.9201,
        #     max_rejector_fraction=0.5860,
        #     diversity_threshold=0.1205,
        #     improvement_threshold=0.0567,
        #     termination_criterion=StoppingByEvaluations(max_evaluations)
        # ),
        'RRAPSO': lambda p: RRAPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.5917,
            c2=5.4661,
            ac1=5.4117,
            ac2=4.3553,
            base_inertia=0.1171,
            min_inertia=0.0592,
            max_inertia=0.4416,
            rebel_fraction=0.1074,
            rejector_fraction=0.2688,
            window_size=47,
            max_rebel_fraction=0.1975,
            max_rejector_fraction=0.8094,
            diversity_threshold=0.2107,
            improvement_threshold=0.0722,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        ## Worse aware algorithms
        'ReverseLearningPSO': lambda p: ReverseLearningPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            b1=4.3732,
            b2=2.7552,
            w=0.0632,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        # 'ReverseLearningPSO_with_bounce': lambda p: ReverseLearningPSO(  # bounce rough tuning
        #     problem=p,
        #     swarm_size=solutions_size,
        #     b1=2.2427,
        #     b2=4.8108,
        #     w=1.6864,
        #     constraint_handling_mode="bounce",
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
        'ReverseLearningGlobalAttractorPSO': lambda p: ReverseLearningGlobalAttractorPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            a=3.0859,
            b1=0.1083,
            b2=0.8373,
            w=0.0243,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'ReverseLearningPersonalAttractorPSO': lambda p: ReverseLearningPersonalAttractorPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            a=3.2241,
            b1=1.6475,
            b2=0.0101,
            w=0.0355,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'CombinedLearningPSO': lambda p: CombinedLearningPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.6799,
            c2=3.2484,
            b1=0.0445,
            b2=0.3843,
            w=0.2530,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ### Contrarian algorithms
        'ContrarianPSO': lambda p: ContrarianPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.4049,
            c2=5.7292,
            ac2=4.5595,
            w=0.1015,
            contrarian_fraction=0.4802,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'DefeatistPSO': lambda p: DefeatistPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.4045,
            c2=5.2960,
            ac1=0.9531,
            w=0.0687,
            defeatist_fraction=0.53,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'ContrarianDefeatistPSO': lambda p: ContrarianDefeatistPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.9052,
            c2=5.0633,
            ac1=1.7460,
            ac2=4.3333,
            w=0.0665,
            contrarian_fraction=0.12,
            defeatist_fraction=0.19,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'CDAPSO': lambda p: CDAPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=5.8826,
            c2=0.8914,
            ac1=5.5753,
            ac2=3.1205,
            base_inertia=0.0953,
            min_inertia=0.0882,
            max_inertia=0.4882,
            contrarian_fraction=0.0732,
            defeatist_fraction=0.1181,
            window_size=42,
            max_contrarian_fraction=0.8457,
            max_defeatist_fraction=0.4605,
            diversity_threshold=0.0745,
            improvement_threshold=0.0414,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        ### Eschewer algortihms
        'EschewerPSO': lambda p: EschewerPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=2.1774,
            c2=3.6490,
            ac2=1.6576,
            w=0.0908,
            eschewer_fraction=0.22,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'EscapistPSO': lambda p: EscapistPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=2.5767,
            c2=5.8653,
            ac1=0.0594,
            w=0.0470,
            escapist_fraction=0.48,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'EschewerEscapistPSO': lambda p: EschewerEscapistPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.8995,
            c2=5.2708,
            ac1=1.5816,
            ac2=4.8526,
            w=0.0879,
            eschewer_fraction=0.16,
            escapist_fraction=0.23,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        'EEAPSO': lambda p: EEAPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=1.2897,
            c2=4.9065,
            ac1=0.3558,
            ac2=3.6638,
            base_inertia=0.1142,
            min_inertia=0.0681,
            max_inertia=0.4696,
            eschewer_fraction=0.1033,
            escapist_fraction=0.1078,
            window_size=37,
            max_eschewer_fraction=0.4509,
            max_escapist_fraction=0.4807,
            diversity_threshold=0.0888,
            improvement_threshold=0.0733,
            termination_criterion=StoppingByEvaluations(max_evaluations)
        ),
        ## Adaptive algorithms
        'GlobalAdaptivePSO': lambda p: GlobalAdaptivePSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.2853,
            c2=4.6565,
            max_c1=5.8463,
            max_c2=9.5588,
            w=0.0355,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        'PersonalAdaptivePSO': lambda p: PersonalAdaptivePSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.2412,
            c2=3.9367,
            max_c1=4.8349,
            max_c2=6.4719,
            w=0.0983,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ### Other
        'FAPSO': lambda p: FAPSO(  # clip rough tuning
            problem=p,
            swarm_size=solutions_size,
            c1=0.2400,
            c2=5.3089,
            w=0.0818,
            fractal_depth=4,
            convergence_threshold=0.0266,
            termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        ),
        ## Not mine
        # 'DCS-PSO': lambda p: DCSPSO(  # clip rough tuning
        #     problem=p,
        #     swarm_size=solutions_size,
        #     termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations)
        # ),
    }

    group_of_algorithms = {
        # 'Algorithms with roles': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO',
        #                           'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                           'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO'],
        # 'Algorithms with roles without RRAPSO': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO',
        #                                          'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                                          'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO', 'RRAPSO'],
        # 'Rebel algorithms': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO', 'RRAPSO'],
        # 'Rebel algorithms without RRAPSO': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO'],
        # 'Contrarian algorithms without CDAPSO': ['ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO'],
        # 'Eschewer algorithms without EEAPSO': ['EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO'],
        # 'Worse aware algorithms all': ['ReverseLearningPSO', 'ReverseLearningGlobalAttractorPSO',
        #                                'ReverseLearningPersonalAttractorPSO',
        #                                'CombinedLearningPSO', 'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                                'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO'],
        # 'Reverse learning algorithms': ['ReverseLearningPSO', 'ReverseLearningGlobalAttractorPSO',
        #                                 'ReverseLearningPersonalAttractorPSO',
        #                                 'CombinedLearningPSO'],
        # 'Worse aware algorithms without reverse learning': ['ReverseLearningGlobalAttractorPSO',
        #                                                     'ReverseLearningPersonalAttractorPSO',
        #                                                     'CombinedLearningPSO', 'ContrarianPSO', 'DefeatistPSO',
        #                                                     'ContrarianDefeatistPSO',
        #                                                     'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO'],
        # 'Worse aware algorithms without all reverse learning': ['ContrarianPSO', 'DefeatistPSO',
        #                                                         'ContrarianDefeatistPSO',
        #                                                         'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO'],
        # 'Reverse learning algorithms without reverse learning': ['ReverseLearningGlobalAttractorPSO',
        #                                                          'ReverseLearningPersonalAttractorPSO',
        #                                                          'CombinedLearningPSO'],
        # 'Worse aware algorithms negative': ['ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO'],
        # 'Worse aware algorithms positive': ['EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO'],
        # 'Adaptive algorithms': ['GlobalAdaptivePSO', 'PersonalAdaptivePSO', 'FAPSO'],
        # 'Adaptive algorithms without FAPSO': ['GlobalAdaptivePSO', 'PersonalAdaptivePSO'],
        # 'All without reverse learning': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO', 'RRAPSO',
        #                                  'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO', 'CDAPSO',
        #                                  'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO', 'EEAPSO',
        #                                  'ReverseLearningGlobalAttractorPSO', 'ReverseLearningPersonalAttractorPSO',
        #                                  'CombinedLearningPSO', 'GlobalAdaptivePSO', 'PersonalAdaptivePSO', 'FAPSO'],
        # 'All without reverse learning, FAPSO and RRAPSO': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO',
        #                                             'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                                             'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO',
        #                                             'ReverseLearningGlobalAttractorPSO',
        #                                             'ReverseLearningPersonalAttractorPSO',
        #                                             'CombinedLearningPSO', 'GlobalAdaptivePSO', 'PersonalAdaptivePSO'],
        'All without all reverse learning': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO', 'RRAPSO',
                                             'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO', 'CDAPSO',
                                             'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO', 'EEAPSO',
                                             'GlobalAdaptivePSO', 'PersonalAdaptivePSO', 'FAPSO'],
        # 'All without all reverse learning, FAPSO and RRAPSO': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO',
        #                                                 'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                                                 'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO',
        #                                                 'GlobalAdaptivePSO', 'PersonalAdaptivePSO'],
    }
    group_of_algorithms = {
        group_name: ['PSO'] + [algo for algo in algorithm_list if algo != 'PSO']
        for group_name, algorithm_list in group_of_algorithms.items()
    }

    return (algorithms, group_of_algorithms, problems, no_of_runs, number_of_variables, solutions_size,
            max_evaluations, frequency, algorithm_colors, results_dir)


def initialize_algorithms(algorithms, problem):
    return {name: algo(problem) for name, algo in algorithms.items()}


def make_dir(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
