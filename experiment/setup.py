import colorsys
import os
from itertools import chain
import matplotlib.colors as mcolors
from jmetal.problem.singleobjective.unconstrained import Rastrigin

from experiment.factories import factory_PSO, factory_RebelPSO, factory_RejectorPSO, factory_RebelRejectorPSO, \
    factory_RRAPSO, factory_ContrarianPSO, factory_DefeatistPSO, factory_ContrarianDefeatistPSO, factory_CDAPSO, \
    factory_EschewerPSO, factory_EscapistPSO, factory_EschewerEscapistPSO, factory_EEAPSO, factory_ReverseLearningPSO, \
    factory_ReverseLearningGlobalAttractorPSO, factory_ReverseLearningPersonalAttractorPSO, factory_CombinedLearningPSO, \
    factory_CLAPSO, factory_AnarchicPSO, factory_AmnesiacPSO, factory_WandererPSO, factory_AAAPSO, factory_NoisyPSO, \
    factory_PerturbationPSO, factory_PartialResetPSO, factory_CollectiveResetPSO, factory_FRAPSO, \
    factory_HybridFullDisjointPSO, factory_HybridPartialDisjointPSO, factory_HybridAdditivePSO, \
    factory_HybridFullDisjointRestarterPSO, factory_HybridPartialDisjointRestarterPSO, \
    factory_HybridAdditiveRestarterPSO, factory_CAPSO, factory_IAPSO, factory_DrifterPSO, factory_DAPSO, factory_CMAES, \
    factory_LSHADE, factory_AnarchicAmnesiacPSO, factory_HybridDisjointPSO_WithWanderer_NonVar, \
    factory_HybridAdditivePSO_WithWanderer_NonVar
from experiment.globals import NO_OF_RUNS, NUMBER_OF_VARIABLES, G_SOLUTIONS_SIZE, G_MAX_EVALUATIONS, RESULTS_DIR

from algorithm.AdaptivePSO import CoAdaptativePSO, IndividualAdaptivePSO
from algorithm.reinitialized_PSO import FRAPSO, CollectiveResetPSO, PartialResetPSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO, \
    ReverseLearningPSO
from algorithm.hybrid_diverse import HybridPartialDisjointPSO, HybridFullDisjointPSO, HybridAdditivePSO, \
    HybridFullDisjointPSO_WithRandom, HybridPartialDisjointPSO_WithRandom, HybridAdditivePSO_WithRandom, \
    HybridFullDisjointRestarterPSO, HybridPartialDisjointRestarterPSO, HybridAdditiveRestarterPSO
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, CDAPSO, EEAPSO, AnarchicPSO, AmnesiacPSO, \
    ErraticPSO, WandererPSO, AAAPSO, NAPSO, CLAPSO
from algorithm.single_objective_PSO import SingleObjectivePSO, PerturbationPSO

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
from problem.n_variables.hgbat import HGBat
from problem.n_variables.CEC import RotatedHighConditionedElliptic, RotatedBentCigar, RotatedDiscus, \
    ShiftedRotatedRosenbrock, ShiftedRotatedAckley, ShiftedRastrigin, ShiftedRotatedRastrigin, ShiftedSchwefel, \
    ShiftedRotatedSchwefel, ShiftedRotatedKatsuura, ShiftedRotatedHappyCat, ShiftedRotatedHGBat, \
    ShiftedRotatedExpandedGriewankPlusRosenbrock, ShiftedRotatedExpandedScafferF6, HybridFunction1, HybridFunction2, \
    HybridFunction3, HybridFunction4, HybridFunction5, HybridFunction6, CompositionFunction1, CompositionFunction2, \
    CompositionFunction3, CompositionFunction4, CompositionFunction5, CompositionFunction6, CompositionFunction7, \
    CompositionFunction8, ShiftedRotatedSchafferF7
from problem.n_variables.ackley import Ackley
from problem.n_variables.alpine import AlpineN1, AlpineN2, AlpineN2Max, AlpineN1Max
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

BASE_GROUP_COLORS = {
    'Baseline': '#000000',  # dark grey
    'Rebel': '#2ca02c',  # green
    'Contrarian': '#bcbd22',  # olive‐green
    'Eschewer': '#1f77b4',  # blue
    'Combined learning': '#9edae5',  # light‐blue (paler than Eschewer)
    'Anarchic': '#9467bd',  # purple
    'Noisy': '#c20078',  # light‐pink (paler than Anarchic)
    'Erratic': '#c20078',  # light‐pink (paler than Anarchic)
    'Reset': '#ff7f0e',  # orange
    'Hybrid': '#d62728',  # red
    'RandomComplex': '#653700', #brown
    'Variable coefficient': '#8c564b',
    'Reverse learning': '#826d8c',
    'SOTA': '#ff7f0e',  # orange
}

LINE_STYLES = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)),
               # (0, (5, 1)), (0, (1, 1))
               ]


def setup_experiment():
    no_of_runs = NO_OF_RUNS
    number_of_variables = NUMBER_OF_VARIABLES
    solutions_size = G_SOLUTIONS_SIZE
    max_evaluations = G_MAX_EVALUATIONS
    frequency = G_SOLUTIONS_SIZE  # Snapshot each generation

    groups = {
        'Baseline': [
            'PSO',
            'PerturbationPSO',
            'DrifterPSO',
            'DAPSO',
        ],
        'Rebel': [
            'RebelPSO',
            'RejectorPSO',
            'RebelRejectorPSO',
            'RRAPSO',
        ],
        'Contrarian': [
            'ContrarianPSO',
            'DefeatistPSO',
            'ContrarianDefeatistPSO',
            'CDAPSO',
        ],
        'Eschewer': [
            'EschewerPSO',
            'EscapistPSO',
            'EschewerEscapistPSO',
            'EEAPSO',
        ],
        'Combined learning': [
            'CombinedLearningPSO',
            'CLAPSO',
        ],
        'Anarchic': [
            'AnarchicPSO',
            'AmnesiacPSO',
            'WandererPSO',
            'AnarchicAmnesiacPSO'
            'AAAPSO',
            'NoisyPSO',
            'NAPSO',
            'ErraticPSO'
        ],
        'Noisy': [
            'NoisyPSO',
            'NAPSO',
        ],
        'Reset': [
            'PartialResetPSO',
            'CollectiveResetPSO',
            'FRAPSO',
        ],
        'Hybrid': [
            'HybridFullDisjointPSO',
            'HybridPartialDisjointPSO',
            'HybridAdditivePSO',
        ],
        'RandomComplex': [
            'HybridDisjointPSO_WithWanderer',
            'HybridAdditivePSO_WithWanderer',
        ],
        'Variable coefficient': [
            'CAPSO',
            'IAPSO',
        ],
        'Reverse learning': [
            'ReverseLearningPSO',
            'ReverseLearningGlobalAttractorPSO',
            'ReverseLearningPersonalAttractorPSO',
        ],
        'SOTA': [
            'CMAES',
            'LSHADE',
        ]
    }

    # base_group_colors = {
    #     'Baseline': '#000000',  # dark grey
    #     'Rebel': '#2ca02c',  # green
    #     'Contrarian': '#bcbd22',  # olive‐green
    #     'Eschewer': '#1f77b4',  # blue
    #     'Combined learning': '#9edae5',  # light‐blue (paler than Eschewer)
    #     'Anarchic': '#9467bd',  # purple
    #     'Noisy': '#c20078',  # light‐pink (paler than Anarchic)
    #     'Reset': '#ff7f0e',  # orange
    #     'Hybrid': '#d62728',  # red
    #     'Variable coefficient': '#8c564b',
    #     'Reverse learning': '#826d8c',
    # }

    lighten_amt = 0  # 0 = unchanged, 1 = full white

    algorithm_colors = {}

    for group_name, algos in groups.items():
        base_hex = BASE_GROUP_COLORS[group_name]
        r, g, b = mcolors.to_rgb(base_hex)
        h, l, s = colorsys.rgb_to_hls(r, g, b)

        n = len(algos)
        # existing L ramp—whatever you had before
        l_min, l_max = 0.3, 0.7
        if n > 1:
            Ls = [l_min + (l_max - l_min) * (i / (n - 1)) for i in range(n)]
        elif n == 1:
            Ls = [l_min]
        else:
            Ls = []  # Handles n=0 or negative
        Ls = list(reversed(Ls))

        for algo, L in zip(algos, Ls):
            # blend that L toward pure white lightness=1.0
            L2 = L + (1.0 - L) * lighten_amt
            r2, g2, b2 = colorsys.hls_to_rgb(h, L2, s)
            algorithm_colors[algo] = mcolors.to_hex((r2, g2, b2))

    # leave your two baseline colors exactly as before:
    algorithm_colors['PSO'] = 'xkcd:black'
    algorithm_colors['PerturbationPSO'] = 'xkcd:charcoal'
    algorithm_colors['DrifterPSO'] = 'xkcd:blue grey'
    algorithm_colors['DAPSO'] = 'xkcd:greenish grey'

    results_dir = RESULTS_DIR
    make_dir(results_dir)

    # Define problems
    n_variables_problems = [
        ##
        RotatedBentCigar(number_of_variables),
        RotatedDiscus(number_of_variables),
        RotatedHighConditionedElliptic(number_of_variables),
        ShiftedSchwefel(number_of_variables),
        ShiftedRotatedHappyCat(number_of_variables),
        ShiftedRotatedHGBat(number_of_variables),
        ShiftedRotatedWeierstrass(number_of_variables),
        ##
        AlpineN1(number_of_variables),
        CrownedCross(number_of_variables),
        EggHolder(number_of_variables),
        ExpandedShaffer(number_of_variables),
        GeneralizedSchafferN1(number_of_variables),
        GeneralizedSchafferN2(number_of_variables),
        GeneralizedSchafferN3(number_of_variables),
        GeneralizedSchafferN4(number_of_variables),
        GeneralizedSchmidtVetters(number_of_variables),
        LennardJonesMinimumEnergyCluster(number_of_variables),
        Michalewicz(number_of_variables),
        Mishra03(number_of_variables),
        Mishra04(number_of_variables),
        RosenbrockModified02(number_of_variables),
        Salomon(number_of_variables),
        SchwefelN20(number_of_variables),
        SchwefelN36(number_of_variables),
        SchwefelN6(number_of_variables),
        ShubertN3(number_of_variables),
        ShubertN4(number_of_variables),
        SineEnvelope(number_of_variables),
        Stochastic(number_of_variables),
        StretchedV(number_of_variables),
        StyblinskiTang(number_of_variables),
        ShiftedRotatedSchafferF7(number_of_variables),

        ## Rejected after experiment

        # ShiftedRotatedAckley(number_of_variables),
        # ShiftedRastrigin(number_of_variables),
        # ShiftedRotatedRastrigin(number_of_variables),
        # CrossLeggedTable(number_of_variables),
        # GeneralizedHolderTable(number_of_variables),
        # Levy(number_of_variables),
        # Mishra06(number_of_variables),
        # SchwefelN21(number_of_variables),

        # ## Rejected

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
        Rastrigin(number_of_variables),  # irace
        # Sphere(number_of_variables), # irace

        # ## uninterested results
        # BentCigar(number_of_variables),
        # Bird(number_of_variables),
        # CarromTable(number_of_variables),
        # Cross(number_of_variables),
        # Discus(number_of_variables),
        # DixonPrice(number_of_variables),
        # GeneralizedCrossInTray(number_of_variables),
        # GeneralizedDixonPriceRosenbrock(number_of_variables),
        # GeneralizedPenalizedN1(number_of_variables),
        # GeneralizedSchafferN7(number_of_variables),
        # Griewank(number_of_variables),
        # HappyCat(number_of_variables),
        # HGBat(number_of_variables),
        # Mishra01(number_of_variables),
        # Mishra02(number_of_variables),
        # Mishra11(number_of_variables),
        # PenHolder(number_of_variables),
        # Plateau(number_of_variables),
        # Quartic(number_of_variables),
        # Rosenbrock(number_of_variables),
        # RosenbrockModified01(number_of_variables),
        # SchwefelN22(number_of_variables),
        # StepN1(number_of_variables),
        # StepN2(number_of_variables),
        # StepN3(number_of_variables),
        # TestTubeHolder(number_of_variables),
        # Zakharov(number_of_variables),

        ## redundant results
        # AlpineN1Max(number_of_variables),
        # AlpineN2Max(number_of_variables),

        ## inf/nan results
        # ShubertN1(number_of_variables),
        # AlpineN2(number_of_variables),

        ## final rejection
        # ShiftedRotatedRosenbrock(number_of_variables),
        # ShiftedRotatedSchwefel(number_of_variables),
        # ShiftedRotatedExpandedGriewankPlusRosenbrock(number_of_variables),
        # SchwefelN26(number_of_variables),
        # Mishra05(number_of_variables),

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

    algorithms = {
        'PSO': factory_PSO,
        # 'PerturbationPSO': factory_PerturbationPSO,
        # 'DrifterPSO': factory_DrifterPSO,
        # 'DAPSO': factory_DAPSO,
        # 'RebelPSO': factory_RebelPSO,
        # 'RejectorPSO': factory_RejectorPSO,
        'RebelRejectorPSO': factory_RebelRejectorPSO,
        # 'RRAPSO': factory_RRAPSO,
        # 'ContrarianPSO': factory_ContrarianPSO,
        # 'DefeatistPSO': factory_DefeatistPSO,
        'ContrarianDefeatistPSO': factory_ContrarianDefeatistPSO,
        # 'CDAPSO': factory_CDAPSO,
        # 'EschewerPSO': factory_EschewerPSO,
        # 'EscapistPSO': factory_EscapistPSO,
        'EschewerEscapistPSO': factory_EschewerEscapistPSO,
        # 'EEAPSO': factory_EEAPSO,
        # 'ReverseLearningPSO': factory_ReverseLearningPSO,
        # 'ReverseLearningGlobalAttractorPSO': factory_ReverseLearningGlobalAttractorPSO,
        # 'ReverseLearningPersonalAttractorPSO': factory_ReverseLearningPersonalAttractorPSO,
        # 'CombinedLearningPSO': factory_CombinedLearningPSO,
        # 'CLAPSO': factory_CLAPSO,
        # 'AnarchicPSO': factory_AnarchicPSO,
        # 'AmnesiacPSO': factory_AmnesiacPSO,
        # 'AnarchicAmnesiacPSO': factory_AnarchicAmnesiacPSO,
        # 'WandererPSO': factory_WandererPSO,
        # 'AAAPSO': factory_AAAPSO,
        # 'NoisyPSO': factory_NoisyPSO,
        # 'NAPSO': factory_NoisyPSO,
        # 'PartialResetPSO': factory_PartialResetPSO,
        # 'CollectiveResetPSO': factory_CollectiveResetPSO,
        # 'FRAPSO': factory_FRAPSO,
        'HybridFullDisjointPSO': factory_HybridFullDisjointPSO,
        'HybridPartialDisjointPSO': factory_HybridPartialDisjointPSO,
        'HybridAdditivePSO': factory_HybridAdditivePSO,
        # 'HybridFullDisjointPSO_WithRandom': factory_HybridFullDisjointPSO,
        # 'HybridPartialDisjointPSO_WithRandom': factory_HybridPartialDisjointPSO,
        # 'HybridAdditivePSO_WithRandom': factory_HybridAdditivePSO,
        # 'HybridDisjointPSO_WithWanderer': factory_HybridDisjointPSO_WithWanderer,
        # 'HybridAdditivePSO_WithWanderer': factory_HybridAdditivePSO_WithWanderer,
        # 'HybridFullDisjointRestarterPSO': factory_HybridFullDisjointRestarterPSO,
        # 'HybridPartialDisjointRestarterPSO': factory_HybridPartialDisjointRestarterPSO,
        # 'HybridAdditiveRestarterPSO': factory_HybridAdditiveRestarterPSO,
        # 'CAPSO': factory_CAPSO,
        # 'IAPSO': factory_IAPSO,
        'CMAES': factory_CMAES,
        'LSHADE': factory_LSHADE,
    }

    # helper to flatten some groups
    def flatten_group_keys(keys):
        return list(chain.from_iterable(groups[k] for k in keys))

    exclude_groups = ['Reverse learning', 'Variable coefficient', 'Combined learning', 'Reset',
                      # 'Noisy',
                      # 'Anarchic',
                      # 'SOTA', 'Hybrid'
                      ]
    adaptive_algorithms = ['RRAPSO', 'CDAPSO', 'EEAPSO', 'AAAPSO', 'CLAPSO', 'NAPSO', 'DAPSO']
    all_without_additive = flatten_group_keys(
        [g for g in groups if g not in exclude_groups]
    )

    additional_groups = {
        # 'Adaptive algorithms': ['RRAPSO', 'CDAPSO', 'EEAPSO', 'AAAPSO', 'CLAPSO', 'NAPSO', 'DAPSO'],
        # 'Social change': ['RebelPSO', 'ContrarianPSO', 'EschewerPSO', 'AnarchicPSO'],
        # 'Cognitive change': ['RejectorPSO', 'DefeatistPSO', 'EscapistPSO', 'AmnesiacPSO'],
        # 'Social and cognitive change': ['RebelRejectorPSO', 'ContrarianDefeatistPSO', 'EschewerEscapistPSO', 'WandererPSO'],

        # 'Rebel': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO', ],
        # 'Contrarian': ['ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO', ],
        # 'Eschewer': ['EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO', ],
        # 'Anarchic': ['AnarchicPSO', 'AmnesiacPSO', 'WandererPSO', 'NoisyPSO', ],
        # 'Reset': ['PartialResetPSO', 'CollectiveResetPSO', ],
        # 'Hybrid': ['HybridFullDisjointPSO', 'HybridPartialDisjointPSO', 'HybridAdditivePSO', ],

        # 'All without all reverse learning':
        #     ['PerturbationPSO'] + flatten_group_keys([k for k in groups if k not in ['Reverse learning', 'Variable coefficient', 'Adaptive algorithms']]),
        #
        # 'All without all additive':
        # ['PerturbationPSO'] + all_without_additive,

        'All selected algorithms':
            ['PerturbationPSO'] + [alg for alg in all_without_additive if
                                   alg not in [
                                       # 'DrifterPSO',
                                               'DAPSO', 'FRAPSO'] + adaptive_algorithms],
    }

    group_of_algorithms = {
        # **groups,
        **additional_groups
    }

    group_of_algorithms = {
        group_name: ['PSO'] + [algo for algo in algorithm_list if algo not in ['PSO']]
        for group_name, algorithm_list in group_of_algorithms.items()
    }

    return (algorithms, group_of_algorithms, problems, no_of_runs, number_of_variables, solutions_size,
            max_evaluations, frequency, algorithm_colors, results_dir)


def initialize_algorithms(algorithms, problem):
    return {name: algo(problem) for name, algo in algorithms.items()}


def make_dir(results_dir):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
