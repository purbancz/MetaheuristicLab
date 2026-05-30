from jmetal.algorithm.singleobjective import EvolutionStrategy
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.AdaptivePSO import CoAdaptativePSO, IndividualAdaptivePSO
from algorithm.CMAES import CMAES
from algorithm.LSHADE import LSHADE
from algorithm.WAPSO import ReverseLearningPSO, ReverseLearningGlobalAttractorPSO, ReverseLearningPersonalAttractorPSO, \
    CombinedLearningPSO
from algorithm.hybrid_diverse import HybridFullDisjointPSO, HybridPartialDisjointPSO, HybridAdditivePSO, \
    HybridFullDisjointPSO_WithRandom, HybridPartialDisjointPSO_WithRandom, HybridAdditivePSO_WithRandom, \
    HybridFullDisjointRestarterPSO, HybridPartialDisjointRestarterPSO, HybridAdditiveRestarterPSO, \
    HybridDisjointPSO_WithWanderer, HybridAdditivePSO_WithWanderer
from algorithm.particles_with_roles import RebelPSO, RejectorPSO, RebelRejectorPSO, RRAPSO, ContrarianPSO, DefeatistPSO, \
    ContrarianDefeatistPSO, CDAPSO, EschewerPSO, EscapistPSO, EschewerEscapistPSO, EEAPSO, CLAPSO, AnarchicPSO, \
    AmnesiacPSO, WandererPSO, AAAPSO, NoisyPSO, NAPSO, DrifterPSO, DAPSO, AnarchicAmnesiacPSO
from algorithm.reinitialized_PSO import PartialResetPSO, CollectiveResetPSO, FRAPSO
from algorithm.single_objective_PSO import SingleObjectivePSO, PerturbationPSO
from experiment.globals import G_SOLUTIONS_SIZE, G_MAX_EVALUATIONS, NUMBER_OF_VARIABLES


def factory_LSHADE(p):
    return LSHADE(
        problem=p,
        initial_population_size=G_SOLUTIONS_SIZE,  # Or problem.number_of_variables * 18 as suggested in some contexts
        # pop_size_factor= 18,  # Or problem.number_of_variables * 18 as suggested in some contexts
        termination_criterion=StoppingByEvaluations(G_MAX_EVALUATIONS), # Or max_evaluations=p.number_of_variables * 10000
        memory_size=25,
        p_best_rate=0.158,
        archive_size_rate=2.237
    )


def factory_CMAES(p):
    return CMAES(
        problem=p,
        mu=41,
        lambda_=52,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )


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
        c1=0.578412438913527,
        c2=5.156399293441848,
        cl_c1=4.906086393866017,
        cl_c2=0.766015764236342,
        b1=0.717953770228799,
        b2=0.929150663432141,
        base_inertia=0.112589149253055,
        min_inertia=0.087102617677445,
        max_inertia=0.11377021036723,
        cl_fraction=0.067908113199937,
        max_cl_fraction=0.458810763170724,
        window_size=19,
        diversity_threshold=0.161780593366814,
        improvement_threshold=0.032948084493759,
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

def factory_AnarchicAmnesiacPSO(p):
    return AnarchicAmnesiacPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.091236956472647,
        c1=4.522803205994748,
        c2=1.755962797723315,
        random_strength_social=0.763185707895226,
        random_strength_cognitive=4.2799235012880095,
        anarchic_fraction=0.084672896214155,
        amnesiac_fraction=0.289013979624548,
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


def factory_DrifterPSO(p):
    return DrifterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.088711856756825,
        c1=1.8564843990447324,
        c2=4.154167631958623,
        drifter_fraction=0.182009454005769,
        perturbation_scale=0.098183963203318,
        perturbation_method="gaussian",
        constraint_handling_mode="clip"
    )


def factory_DAPSO(p):
    return DAPSO(
        problem=p,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        swarm_size=G_SOLUTIONS_SIZE,
        c1=0.849918319132478,
        c2=4.685357255835426,
        base_inertia=0.09887748582866825,
        min_inertia=0.091426794132834,
        max_inertia=0.100739664967222,
        perturbation_scale=0.037160322082221,
        drifter_fraction=0.219271367175397,
        max_drifter_fraction=0.404505117115725,
        window_size=28,
        diversity_threshold=0.260324821841396,
        improvement_threshold=0.05957390748379,
        perturbation_method="gaussian",
        constraint_handling_mode="clip"
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

def factory_HybridFullDisjointPSO_WithRandom_Var(p):
    return HybridFullDisjointPSO_WithRandom(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.070613324161097,
        c1=1.2648841133643252,
        c2=5.139031539268885,

        rejector_c=0,
        defeatist_c=0,
        escapist_c=0,
        amnesiac_c=3.727243242431227,
        rebel_c=0,
        contrarian_c=0,
        eschewer_c=0,
        anarchic_c=3.4265350576587106,

        rejector_fraction=0,
        defeatist_fraction=0,
        escapist_fraction=0,
        amnesiac_fraction=0.534157673588983,
        rebel_fraction=0,
        contrarian_fraction=0,
        eschewer_fraction=0,
        anarchic_fraction=0.116381850790778,

        constraint_handling_mode="clip",
        assign_roles_every_iteration=False,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )

def factory_HybridPartialDisjointPSO_WithRandom_Var(p):
    return HybridPartialDisjointPSO_WithRandom(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.054655302723429,
        c1=0.429282943785279,
        c2=5.76062395431257,

        rejector_c=0,
        defeatist_c=0,
        escapist_c=0,
        rebel_c=0,
        contrarian_c=0,
        eschewer_c=0,
        amnesiac_c=4.2311484550184915,
        anarchic_c=0.51542902758087,

        rejector_fraction=0,
        defeatist_fraction=0,
        escapist_fraction=0,
        rebel_fraction=0,
        contrarian_fraction=0,
        eschewer_fraction=0,
        amnesiac_fraction=0.172142057161717,
        anarchic_fraction=0.262647267385497,

        constraint_handling_mode="clip",
        assign_roles_every_iteration=False,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )

def factory_HybridAdditivePSO_WithRandom_Var(p):
    return HybridAdditivePSO_WithRandom(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.105078908098416,
        c1=5.922638446294212,
        c2=1.853544278625566,

        rejector_c=0,
        defeatist_c=0,
        escapist_c=0,
        rebel_c=0,
        contrarian_c=0,
        eschewer_c=0,
        anarchic_c=2.803031943603203,
        amnesiac_c=5.298190812498183,

        std_cognitive_prob=0.468507292758563,
        rejector_prob=0,
        defeatist_prob=0,
        escapist_prob=0,
        amnesiac_prob=0.026503169610302,

        std_social_prob=0.322113944701517,
        rebel_prob=0,
        contrarian_prob=0,
        eschewer_prob=0,
        anarchic_prob=0.012611293683424,

        constraint_handling_mode="clip",
        assign_flags_every_iteration=False,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS)
    )

def factory_HybridDisjointPSO_WithWanderer_NonVar(p):
    return HybridDisjointPSO_WithWanderer(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.132778503791544,
        c1=1.0194514621712447,
        c2=5.787507933504707,
        wanderer_c=4.266136447445009,
        wanderer_fraction=0.116240284310713,
        constraint_handling_mode="clip",
        assign_roles_every_iteration=True,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=G_MAX_EVALUATIONS
        )
    )


def factory_HybridAdditivePSO_WithWanderer_NonVar(p):
    return HybridAdditivePSO_WithWanderer(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        w=0.193300939674606,
        c1=0.530432854954883,
        c2=5.581592153666334,
        wanderer_c=2.1815272230776914,
        std_cognitive_prob=0.111307913121892,
        std_social_prob=0.733331292108552,
        wanderer_prob=0.115546222298038,
        constraint_handling_mode="clip",
        assign_flags_every_iteration=True,
        termination_criterion=StoppingByEvaluations(
            max_evaluations=G_MAX_EVALUATIONS
        )
    )

def factory_HybridFullDisjointRestarterPSO(p):
    return HybridFullDisjointRestarterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.095688266897959,
        c1=0.885299337265381,
        rejector_c=3.8911996337136565,
        defeatist_c=4.455181777391979,
        escapist_c=1.863890245263307,
        amnesiac_c=1.798996861868356,
        c2=5.048325894802074,
        rebel_c=1.4419974833320057,
        contrarian_c=0.379686254333889,
        eschewer_c=1.595700282851451,
        anarchic_c=1.3368057123780848,
        rejector_fraction=0.08370367413855925,
        defeatist_fraction=0.12600686216191112,
        escapist_fraction=0.1678420726289181,
        amnesiac_fraction=0.203805610036889,
        rebel_fraction=0.02871986663213928,
        contrarian_fraction=0.1525982821646841,
        eschewer_fraction=0.06806892649672112,
        anarchic_fraction=0.169254705740178,
        assign_roles_every_iteration=True,
        convergence_threshold=0.094339345070986,
        restarter_fraction=0.44786082281561,
        constraint_handling_mode="clip"
    )


def factory_HybridPartialDisjointRestarterPSO(p):
    return HybridPartialDisjointRestarterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.122750919509418,
        c1=0.434133398554887,
        c2=5.134176452508831,
        rejector_c=3.002871625054691,
        defeatist_c=5.62242970100857,
        escapist_c=1.5628289377239988,
        amnesiac_c=2.09212009749896,
        rebel_c=2.8599644934461064,
        contrarian_c=1.869958125619992,
        eschewer_c=2.535012045494592,
        anarchic_c=4.808400462888208,
        restarter_fraction=0.69354141959732,
        rejector_fraction=0.1068265272458872,
        defeatist_fraction=0.3250594759554392,
        escapist_fraction=0.2174558802632997,
        amnesiac_fraction=0.3506581165353737,
        rebel_fraction=0.2917510430393993,
        contrarian_fraction=0.2118763668702528,
        eschewer_fraction=0.2718605739376922,
        anarchic_fraction=0.2245120161526556,
        convergence_threshold=0.098495489044966,
        assign_roles_every_iteration=True,
        constraint_handling_mode="clip"
    )


def factory_HybridAdditiveRestarterPSO(p):
    return HybridAdditiveRestarterPSO(
        problem=p,
        swarm_size=G_SOLUTIONS_SIZE,
        termination_criterion=StoppingByEvaluations(max_evaluations=G_MAX_EVALUATIONS),
        w=0.049578276621761,
        c1=0.297566305360359,
        rejector_c=0.424865165604231,
        defeatist_c=0.252764182412948,
        escapist_c=1.6537414981181828,
        amnesiac_c=1.0883350406296681,
        c2=5.251605657010062,
        rebel_c=4.011028900194063,
        contrarian_c=3.6266712123543567,
        eschewer_c=0.760555256014262,
        anarchic_c=4.223278214695644,
        std_cognitive_prob=0.022185207040309,
        rejector_prob=0.152194882893105,
        defeatist_prob=0.29939118062107,
        escapist_prob=0.292262123107002,
        amnesiac_prob=0.038128900661213,
        std_social_prob=0.375140666221316,
        rebel_prob=0.038566485365754,
        contrarian_prob=0.865700842397552,
        eschewer_prob=0.076615631507984,
        anarchic_prob=0.049409962971306,
        assign_flags_every_iteration=True,
        convergence_threshold=0.032681324928114,
        restarter_fraction=0.399034124683013,
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
