import os

from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.AdaptivePSO import CoAdaptativePSO, IndividualAdaptivePSO
from algorithm.reinitialized_PSO import FRAPSO, CollectiveResetPSO, PartialResetPSO
from algorithm.WAPSO import ReverseLearningGlobalAttractorPSO, CombinedLearningPSO, ReverseLearningPersonalAttractorPSO, \
    ReverseLearningPSO
from algorithm.hybrid_diverse import HybridPartialDisjointPSO, HybridFullDisjointPSO, HybridAdditivePSO, \
    HybridFullDisjointPSO_WithRandom, HybridPartialDisjointPSO_WithRandom, HybridAdditivePSO_WithRandom, \
    HybridFullDisjointRestarterPSO, HybridPartialDisjointRestarterPSO, HybridAdditiveRestarterPSO
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, CDAPSO, EEAPSO, AnarchicPSO, AmnesiacPSO, \
    WandererPSO, NoisyPSO, AAAPSO, NAPSO, CLAPSO
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

NO_OF_RUNS = 5
NUMBER_OF_VARIABLES = 100
###
G_SOLUTIONS_SIZE = 100
G_MAX_EVALUATIONS = 25000
###
RESULTS_DIR = 'experiment_results'


def factory_PSO(p):
    return SingleObjectivePSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=4.373186623347942,
        c2=2.7550764085992134,
        w=0.063200081558323,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_RebelPSO(p):
    return RebelPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.307688161908015,
        c2=5.530968353849125,
        ac2=3.923395769404532,
        w=0.130799692121227,
        rebel_fraction=0.179677247325933,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_RejectorPSO(p):
    return RejectorPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=1.5453302933766626,
        c2=5.917520827367081,
        ac1=0.466119854194634,
        w=0.083852078943052,
        rejector_fraction=0.67871106256468,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_RebelRejectorPSO(p):
    return RebelRejectorPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=1.2917073232273844,
        c2=4.832642324745723,
        ac1=1.441428786452163,
        ac2=3.7444907132412686,
        w=0.115458375538175,
        rebel_fraction=0.18928109655799,
        rejector_fraction=0.193683456340565,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_RRAPSO(p):
    return RRAPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.548691568141336,
        c2=5.907597506402747,
        ac1=3.2624960653318853,
        ac2=0.376736442803273,
        base_inertia=0.109099990738931,
        min_inertia=0.019422700188969,
        max_inertia=0.467675367006755,
        rebel_fraction=0.089454737167723,
        rejector_fraction=0.10863150880624,
        window_size=27,
        max_rebel_fraction=0.148842504973759,
        max_rejector_fraction=0.679700999528298,
        diversity_threshold=0.027890990404388,
        improvement_threshold=0.038668316988009,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_ContrarianPSO(p):
    return ContrarianPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.404857785667593,
        c2=5.729240158735593,
        ac2=4.559505440999292,
        w=0.101459174094392,
        contrarian_fraction=0.480183592836892,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_DefeatistPSO(p):
    return DefeatistPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.404528715091359,
        c2=5.296019337408482,
        ac1=0.953051777442312,
        w=0.068745315537278,
        defeatist_fraction=0.5278118065626,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_ContrarianDefeatistPSO(p):
    return ContrarianDefeatistPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.905168830733747,
        c2=5.06331501616116,
        ac1=1.7459907261121894,
        ac2=4.33333920937383,
        w=0.066483551613745,
        contrarian_fraction=0.121672489612598,
        defeatist_fraction=0.192437869591703,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_CDAPSO(p):
    return CDAPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=5.88261581482204,
        c2=0.89143474600207,
        ac1=5.575331382236984,
        ac2=3.120455009391698,
        base_inertia=0.09531358967731,
        min_inertia=0.088214243704632,
        max_inertia=0.48823616972219,
        contrarian_fraction=0.073214849836237,
        defeatist_fraction=0.118111938088691,
        window_size=42,
        max_contrarian_fraction=0.845656397131062,
        max_defeatist_fraction=0.460539231789005,
        diversity_threshold=0.074527032625495,
        improvement_threshold=0.04143143069345,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_EschewerPSO(p):
    return EschewerPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=2.1774001982532,
        c2=3.64900348772787,
        ac2=1.6575999241449189,
        w=0.090811852472692,
        eschewer_fraction=0.224166555700455,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_EscapistPSO(p):
    return EscapistPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=2.5766896424288985,
        c2=5.86528111647695,
        ac1=0.059418776819967,
        w=0.047051309145994,
        escapist_fraction=0.475506405577598,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_EschewerEscapistPSO(p):
    return EschewerEscapistPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.899537417149102,
        c2=5.270847500954289,
        ac1=1.5815717964732474,
        ac2=4.852560033371143,
        w=0.087900392693393,
        eschewer_fraction=0.162502902369604,
        escapist_fraction=0.230072580176876,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_EEAPSO(p):
    return EEAPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=1.289749876747761,
        c2=4.9065094037838595,
        ac1=0.355816068002949,
        ac2=3.663822316963646,
        base_inertia=0.114157882091415,
        min_inertia=0.068050062417605,
        max_inertia=0.469635443616177,
        eschewer_fraction=0.103342982453466,
        escapist_fraction=0.107812491024632,
        window_size=37,
        max_eschewer_fraction=0.450930240404732,
        max_escapist_fraction=0.480703305275815,
        diversity_threshold=0.088794875683937,
        improvement_threshold=0.073341119131162,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_ReverseLearningPSO(p):
    return ReverseLearningPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        b1=4.3732,
        b2=2.7552,
        w=0.0632,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_ReverseLearningGlobalAttractorPSO(p):
    return ReverseLearningGlobalAttractorPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        a=3.0859235064821076,
        b1=0.108341716970255,
        b2=0.837282738204714,
        w=0.024291813233742,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_ReverseLearningPersonalAttractorPSO(p):
    return ReverseLearningPersonalAttractorPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        a=3.224127815247041,
        b1=1.647589032939392,
        b2=0.010053245887266,
        w=0.035467244014656,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_CombinedLearningPSO(p):
    return CombinedLearningPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.679870939188416,
        c2=3.2484486786755418,
        b1=0.04445122849381,
        b2=0.384275983213548,
        w=0.252967959141965,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_CLAPSO(p):
    return CLAPSO(
        problem=p,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        swarm_size=G_SOLUTIONS_SIZE,
        c1=2.0,
        c2=2.0,
        cl_c1=2.0,
        cl_c2=2.0,
        b1=0.5,
        b2=0.5,
        base_inertia=0.9,
        min_inertia=0.4,
        max_inertia=0.9,
        cl_fraction=0.5,
        max_cl_fraction=0.9,
        window_size=10,
        diversity_threshold=0.1,
        improvement_threshold=0.01,
        constraint_handling_mode="clip"
    )


def factory_AnarchicPSO(p):
    return AnarchicPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.096322171404143,
        c1=2.7290090858714087,
        c2=4.024235943238622,
        random_strength=2.2234020741689235,
        anarchic_fraction=0.075244394488655,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_AmnesiacPSO(p):
    return AmnesiacPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.124625650821204,
        c1=0.455051164336021,
        c2=4.829397363998921,
        random_strength=1.8006726666109796,
        amnesiac_fraction=0.059087361415095,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_WandererPSO(p):
    return WandererPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.104550478300695,
        c1=3.4360763756439807,
        c2=2.55880405172936,
        random_strength=2.3357014415796105,
        wanderer_fraction=0.380225883274424,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_AAAPSO(p):
    return AAAPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.790774912488224,
        c2=5.552410713161935,
        base_inertia=0.091445359341472,
        min_inertia=0.075912859093488,
        max_inertia=0.628221210339724,
        anarchic_fraction=0.066778649836667,
        amnesiac_fraction=0.089620440131471,
        max_anarchic_fraction=0.088838865076925,
        max_amnesiac_fraction=0.728022693617781,
        diversity_threshold=0.02391452073353,
        improvement_threshold=0.006969807524771,
        random_strength=0.469952346008692,
        window_size=44,
        constraint_handling_mode="clip",
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_NoisyPSO(p):
    return NoisyPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.115020987604261,
        c1=4.8318858911420115,
        c2=1.614568939944967,
        noise_strength=0.290212372970094,
        noisy_fraction=0.112113689734281,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )

def factory_NAPSO(p):
    return NAPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=2.913687469154542,
        c2=2.909493273550172,
        base_inertia=0.1012660398563355,
        min_inertia=0.086888696778086,
        max_inertia=0.218280616377528,
        noise_strength=0.570005691598192,
        noisy_fraction=0.078288888821339,
        max_noisy_fraction=0.797594775981736,
        window_size=47,
        diversity_threshold=0.239934036427403,
        improvement_threshold=0.039205662121256,
        constraint_handling_mode="clip",
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_PerturbationPSO(p):
    return PerturbationPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.637125109727299,
        c1=2.5453875960820325,
        c2=0.789097207083248,
        perturbation_scale=0.008133747975949,
        perturbation_method="gaussian",
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_PartialResetPSO(p):
    return PartialResetPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.679134138911766,
        c2=5.272727181825268,
        w=0.104836252102958,
        convergence_threshold=0.096846050413171,
        restarter_fraction=0.645623785582219,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_CollectiveResetPSO(p):
    return CollectiveResetPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.110057824204724,
        c2=5.689570413149101,
        w=0.090016392772831,
        convergence_threshold=0.042520351566281,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_FRAPSO(p):
    return FRAPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.240010838523818,
        c2=5.308903636820276,
        w=0.081799781060863,
        fractal_depth=4,
        convergence_threshold=0.026560651344854,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_HybridFullDisjointPSO(p):
    return HybridFullDisjointPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.010042627554697,
        c1=2.5194453242490207,
        c2=1.9589638292411289,
        rejector_c=0.804714320807866,
        defeatist_c=0.352715323277293,
        escapist_c=2.2573850297429416,
        rebel_c=1.5231697428821178,
        contrarian_c=4.525433432281106,
        eschewer_c=3.545177039917037,
        rejector_fraction=0.3531898577274737,
        defeatist_fraction=0.3882793197090974,
        escapist_fraction=0.0160671847325481,
        rebel_fraction=0.0710350179078033,
        contrarian_fraction=0.0353096524802138,
        eschewer_fraction=0.1361189674428635,
        assign_roles_every_iteration=True,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_HybridPartialDisjointPSO(p):
    return HybridPartialDisjointPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.054225916981733,
        c1=5.250704530511119,
        c2=1.0439559585735996,
        rejector_c=0.108339599181597,
        defeatist_c=0.068260451909027,
        escapist_c=5.557822739605506,
        rebel_c=0.444554248210547,
        contrarian_c=0.791514380319767,
        eschewer_c=3.7320856632713113,
        rejector_fraction=0.39268952755304,
        defeatist_fraction=0.5694900458309657,
        escapist_fraction=0.0378204266159942,
        rebel_fraction=0.114291889757625,
        contrarian_fraction=0.364494663773207,
        eschewer_fraction=0.171609898957229,
        assign_roles_every_iteration=True,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_HybridAdditivePSO(p):
    return HybridAdditivePSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.042646428822865,
        c1=0.253645685149182,
        c2=2.307040589069303,
        rejector_c=1.3034967863175075,
        defeatist_c=0.137324173204785,
        escapist_c=0.023227418883015,
        rebel_c=0.040886380920212,
        contrarian_c=2.2432839871932138,
        eschewer_c=2.769447241515806,
        std_cognitive_prob=0.879673273767784,
        rejector_prob=0.535682485751952,
        defeatist_prob=0.538526878006812,
        escapist_prob=0.462726801113222,
        std_social_prob=0.764228739850829,
        rebel_prob=0.475650793954077,
        contrarian_prob=0.46010305052283,
        eschewer_prob=0.31480023964547,
        assign_flags_every_iteration=True,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_HybridFullDisjointPSO_WithRandom(p):
    return HybridFullDisjointPSO_WithRandom(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.045871171887412,
        c1=0.04186456823808,
        c2=3.3512603430983687,
        rejector_c=1.6818015055543236,
        defeatist_c=0.073048466909476,
        escapist_c=4.295140040496995,
        amnesiac_c=0.123202191476923,
        rebel_c=1.046999188101443,
        contrarian_c=1.0201315371765118,
        eschewer_c=2.781363171836315,
        anarchic_c=2.4146755146545096,
        rejector_fraction=0.058218603922034,
        defeatist_fraction=0.287604619139506,
        escapist_fraction=0.150589334690724,
        amnesiac_fraction=0.039872169395881,
        rebel_fraction=0.536254030121648,
        contrarian_fraction=0.692369927851304,
        eschewer_fraction=0.160533277039762,
        anarchic_fraction=0.201068690599398,
        constraint_handling_mode="clip",
        assign_roles_every_iteration=True,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_HybridPartialDisjointPSO_WithRandom(p):
    return HybridPartialDisjointPSO_WithRandom(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.032920866248212,
        c1=3.471681455837407,
        c2=1.2151276229152843,
        rejector_c=0.376657552954615,
        defeatist_c=0.055180455774964,
        escapist_c=0.62896031560951,
        rebel_c=0.427986984634051,
        contrarian_c=1.699577316604754,
        eschewer_c=3.6928779079599714,
        amnesiac_c=4.98124490255147,
        anarchic_c=5.342026988789787,
        rejector_fraction=0.018857441494762,
        defeatist_fraction=0.714529547920285,
        escapist_fraction=0.346943732309136,
        amnesiac_fraction=0.023634819035895,
        rebel_fraction=0.373606012229191,
        contrarian_fraction=0.175060180505544,
        eschewer_fraction=0.241785605848191,
        anarchic_fraction=0.045603668001108,
        constraint_handling_mode="clip",
        assign_roles_every_iteration=True,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_HybridAdditivePSO_WithRandom(p):
    return HybridAdditivePSO_WithRandom(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.029152442728975,
        c1=0.312354103872271,
        c2=2.537451806033552,
        rejector_c=0.435837108869733,
        defeatist_c=4.27456352808514,
        escapist_c=5.300136874325789,
        rebel_c=4.626164727708544,
        contrarian_c=1.8358205978065425,
        eschewer_c=0.565041947981466,
        anarchic_c=2.7414545378092856,
        amnesiac_c=0.554749781031361,
        std_cognitive_prob=0.921591357935905,
        rejector_prob=0.967998524023063,
        defeatist_prob=0.02678740199382,
        escapist_prob=0.054471591028952,
        amnesiac_prob=0.045923766989247,
        std_social_prob=0.178282780413502,
        rebel_prob=0.226833864067438,
        contrarian_prob=0.420293816550519,
        eschewer_prob=0.152088881967566,
        anarchic_prob=0.102623833459703,
        constraint_handling_mode="clip",
        assign_flags_every_iteration=True,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )



def factory_HybridFullDisjointRestarterPSO(p):
    return HybridFullDisjointRestarterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.65,
        c1=2.10,
        rejector_c=0.90,
        defeatist_c=1.00,
        escapist_c=0.85,
        amnesiac_c=0.70,
        c2=2.00,
        rebel_c=1.20,
        contrarian_c=1.10,
        eschewer_c=0.80,
        anarchic_c=0.60,
        rejector_fraction=0.05,
        defeatist_fraction=0.10,
        escapist_fraction=0.05,
        amnesiac_fraction=0.05,
        rebel_fraction=0.10,
        contrarian_fraction=0.10,
        eschewer_fraction=0.05,
        anarchic_fraction=0.05,
        assign_roles_every_iteration=False,
        convergence_threshold=1e-3,
        restarter_fraction=0.20,
        constraint_handling_mode="clip"
    )


def factory_HybridPartialDisjointRestarterPSO(p):
    return HybridPartialDisjointRestarterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.70,
        c1=2.00,
        c2=1.80,
        rejector_c=0.90,
        defeatist_c=1.10,
        escapist_c=0.80,
        amnesiac_c=0.50,
        rebel_c=1.20,
        contrarian_c=1.00,
        eschewer_c=0.70,
        anarchic_c=0.60,
        restarter_fraction=0.15,
        rejector_fraction=0.10,
        defeatist_fraction=0.10,
        escapist_fraction=0.05,
        amnesiac_fraction=0.05,
        rebel_fraction=0.10,
        contrarian_fraction=0.10,
        eschewer_fraction=0.05,
        anarchic_fraction=0.05,
        convergence_threshold=1e-3,
        assign_roles_every_iteration=True,
        constraint_handling_mode="clip"
    )


def factory_HybridAdditiveRestarterPSO(p):
    return HybridAdditiveRestarterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.60,
        c1=2.00,
        rejector_c=0.90,
        defeatist_c=1.10,
        escapist_c=0.80,
        amnesiac_c=0.70,
        c2=1.90,
        rebel_c=1.20,
        contrarian_c=1.00,
        eschewer_c=0.75,
        anarchic_c=0.65,
        std_cognitive_prob=1.00,
        rejector_prob=0.10,
        defeatist_prob=0.10,
        escapist_prob=0.05,
        amnesiac_prob=0.05,
        std_social_prob=1.00,
        rebel_prob=0.10,
        contrarian_prob=0.10,
        eschewer_prob=0.05,
        anarchic_prob=0.05,
        assign_flags_every_iteration=True,
        convergence_threshold=2e-3,
        restarter_fraction=0.15,
        constraint_handling_mode="clip"
    )


def factory_CAPSO(p):
    return CoAdaptativePSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.087918923584117,
        c2=4.53542462540025,
        max_c1=5.7549791553377325,
        max_c2=9.587776575299161,
        w=0.066438426371153,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def factory_IAPSO(p):
    return IndividualAdaptivePSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.364802068763345,
        c2=5.855423966603137,
        max_c1=10.814269144312599,
        max_c2=7.34769908479714,
        w=0.076308243863222,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


def setup_experiment():
    no_of_runs = NO_OF_RUNS
    number_of_variables = NUMBER_OF_VARIABLES
    solutions_size = G_SOLUTIONS_SIZE
    max_evaluations = G_MAX_EVALUATIONS
    frequency = G_SOLUTIONS_SIZE  # Snapshot each generation

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
        'FRAPSO': 'xkcd:teal',
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
        'CAPSO': 'xkcd:goldenrod',
        'IAPSO': 'xkcd:charcoal',
        'EschewerPSO': 'xkcd:pea green',
        'EscapistPSO': 'xkcd:marine',
        'EschewerEscapistPSO': 'xkcd:dark magenta',
        'AnarchicPSO': 'xkcd:dark purple',
        'AmnesiacPSO': 'xkcd:yellow green',
        'WandererPSO': 'xkcd:pale blue',
        'HybridFullDisjointPSO': 'xkcd:reddy brown',
        'HybridPartialDisjointPSO': 'xkcd:strong blue',
        'HybridAdditivePSO': "xkcd:forrest green",
        ###
        'DCS-PSO': 'xkcd:rich purple',
    }

    results_dir = RESULTS_DIR
    make_dir(results_dir)

    # Define problems
    n_variables_problems = [
        # # ##
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
        # ShiftedRotatedExpandedScafferF6(number_of_variables),
        # # ##
        # AlpineN1(number_of_variables),
        # AlpineN1Max(number_of_variables),
        # AlpineN2(number_of_variables),
        # AlpineN2Max(number_of_variables),
        # CrossLeggedTable(number_of_variables),
        # CrownedCross(number_of_variables),
        # EggHolder(number_of_variables),
        # ExpandedShaffer(number_of_variables),
        # GeneralizedHolderTable(number_of_variables),
        # GeneralizedSchafferN1(number_of_variables),
        # GeneralizedSchafferN2(number_of_variables),
        # GeneralizedSchafferN3(number_of_variables),
        # GeneralizedSchafferN4(number_of_variables),
        # GeneralizedSchmidtVetters(number_of_variables),
        # LennardJonesMinimumEnergyCluster(number_of_variables),
        # Levy(number_of_variables),
        # Michalewicz(number_of_variables),
        # Mishra03(number_of_variables),
        # Mishra04(number_of_variables),
        # Mishra05(number_of_variables),
        # Mishra06(number_of_variables),
        # RosenbrockModified02(number_of_variables),
        # Salomon(number_of_variables),
        # SchwefelN20(number_of_variables),
        # SchwefelN21(number_of_variables),
        # SchwefelN26(number_of_variables),
        # SchwefelN36(number_of_variables),
        # SchwefelN6(number_of_variables),
        # ShubertN1(number_of_variables),
        # ShubertN3(number_of_variables),
        # ShubertN4(number_of_variables),
        # SineEnvelope(number_of_variables),
        # Stochastic(number_of_variables),
        # StretchedV(number_of_variables),
        # StyblinskiTang(number_of_variables),

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
        # 'PSO': factory_PSO,
        # 'RebelPSO': factory_RebelPSO,
        # 'RejectorPSO': factory_RejectorPSO,
        # 'RebelRejectorPSO': factory_RebelRejectorPSO,
        # 'RRAPSO': factory_RRAPSO,
        # 'ContrarianPSO': factory_ContrarianPSO,
        # 'DefeatistPSO': factory_DefeatistPSO,
        # 'ContrarianDefeatistPSO': factory_ContrarianDefeatistPSO,
        # 'CDAPSO': factory_CDAPSO,
        # 'EschewerPSO': factory_EschewerPSO,
        # 'EscapistPSO': factory_EscapistPSO,
        # 'EschewerEscapistPSO': factory_EschewerEscapistPSO,
        # 'EEAPSO': factory_EEAPSO,
        # 'ReverseLearningPSO': factory_ReverseLearningPSO,
        # 'ReverseLearningGlobalAttractorPSO': factory_ReverseLearningGlobalAttractorPSO,
        # 'ReverseLearningPersonalAttractorPSO': factory_ReverseLearningPersonalAttractorPSO,
        # 'CombinedLearningPSO': factory_CombinedLearningPSO,
        'CLAPSO': factory_CLAPSO,
        # 'AnarchicPSO': factory_AnarchicPSO,
        # 'AmnesiacPSO': factory_AmnesiacPSO,
        # 'WandererPSO': factory_WandererPSO,
        # 'AAAPSO': factory_AAAPSO,
        # 'NoisyPSO': factory_NoisyPSO,
        # 'NAPSO': factory_NoisyPSO,
        # 'PerturbationPSO': factory_PerturbationPSO,
        # 'PartialResetPSO': factory_PartialResetPSO,
        # 'CollectiveResetPSO': factory_CollectiveResetPSO,
        # 'FRAPSO': factory_FRAPSO,
        # 'HybridFullDisjointPSO': factory_HybridFullDisjointPSO,
        # 'HybridPartialDisjointPSO': factory_HybridPartialDisjointPSO,
        # 'HybridAdditivePSO': factory_HybridAdditivePSO,
        # 'HybridFullDisjointPSO_WithRandom': factory_HybridFullDisjointPSO,
        # 'HybridPartialDisjointPSO_WithRandom': factory_HybridPartialDisjointPSO,
        # 'HybridAdditivePSO_WithRandom': factory_HybridAdditivePSO,
        'HybridFullDisjointRestarterPSO': factory_HybridFullDisjointRestarterPSO,
        'HybridPartialDisjointRestarterPSO': factory_HybridPartialDisjointRestarterPSO,
        'HybridAdditiveRestarterPSO': factory_HybridAdditiveRestarterPSO,
        # 'CAPSO': factory_CAPSO,
        # 'IAPSO': factory_IAPSO
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
        # 'Adaptive algorithms': ['CAPSO', 'IAPSO', 'FRAPSO'],
        # 'Adaptive algorithms without FRAPSO': ['CAPSO', 'IAPSO'],
        # 'All without reverse learning': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO', 'RRAPSO',
        #                                  'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO', 'CDAPSO',
        #                                  'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO', 'EEAPSO',
        #                                  'ReverseLearningGlobalAttractorPSO', 'ReverseLearningPersonalAttractorPSO',
        #                                  'CombinedLearningPSO', 'CAPSO', 'IAPSO', 'FRAPSO'],
        # 'All without reverse learning, FRAPSO and RRAPSO': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO',
        #                                             'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                                             'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO',
        #                                             'ReverseLearningGlobalAttractorPSO',
        #                                             'ReverseLearningPersonalAttractorPSO',
        #                                             'CombinedLearningPSO', 'CAPSO', 'IAPSO'],
        'All without all reverse learning': [
            'RebelPSO', 'RejectorPSO', 'RebelRejectorPSO', 'RRAPSO',
            'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO', 'CDAPSO',
            'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO', 'EEAPSO',
            'AnarchicPSO', 'AmnesiacPSO', 'WandererPSO', 'NoisyPSO', 'PerturbationPSO',
            'PartialResetPSO', 'CollectiveResetPSO', 'FRAPSO',
            'HybridFullDisjointPSO_WithRandom', 'HybridPartialDisjointPSO_WithRandom', 'HybridAdditivePSO_WithRandom',
            'CAPSO', 'IAPSO',
        ],
        # 'All without all reverse learning, FRAPSO and RRAPSO': ['RebelPSO', 'RejectorPSO', 'RebelRejectorPSO',
        #                                                 'ContrarianPSO', 'DefeatistPSO', 'ContrarianDefeatistPSO',
        #                                                 'EschewerPSO', 'EscapistPSO', 'EschewerEscapistPSO',
        #                                                 'CAPSO', 'IAPSO'],
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
