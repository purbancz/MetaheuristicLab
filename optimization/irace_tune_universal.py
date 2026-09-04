import hashlib
import os
import json
import random
import traceback

import numpy as np
import inspect

from jmetal.problem import Sphere
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.reinitialization.reinitialized_pso import FRAPSO, PartialResetPSO, CollectiveResetPSO
from algorithm.reinitialization.boundary_reinitialized_pso import BoundaryReinitializedPSO

from algorithm.role_based.role_hybrids import (
    HybridPartialDisjointRestarterPSO,
    HybridFullDisjointRestarterPSO,
    HybridAdditiveRestarterPSO,
)
from algorithm.pso_ga_hybrids.pgchea import PGCHEA
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation

from algorithm.sparse_roles.sparse_hybrid import (
    SparseHybridAdditivePSO,
    SparseHybridFullDisjointPSO,
    SparseHybridPartialDisjointPSO,
)
from algorithm.sparse_roles.sparse_role_based import (
    SparseAmnesiacPSO,
    SparseAnarchicAmnesiacPSO,
    SparseAnarchicPSO,
    SparseContrarianDefeatistPSO,
    SparseContrarianPSO,
    SparseDefeatistPSO,
    SparseDrifterPSO,
    SparseErraticPSO,
    SparseEscapistPSO,
    SparseEschewerEscapistPSO,
    SparseEschewerPSO,
    SparseRebelPSO,
    SparseRebelRejectorPSO,
    SparseRejectorPSO,
    SparseWandererPSO,
)
from irace import irace, ParameterSpace, Scenario, Experiment, Real, Integer, Bool, Categorical
import rpy2.robjects as robjects

from problem.n_variables.ackley import Ackley

os.environ["LANG"] = "en_US.UTF-8"
os.environ["LC_ALL"] = "en_US.UTF-8"
os.environ["R_DEFAULT_ENCODING"] = "UTF-8"
robjects.r('Sys.setlocale("LC_ALL", "en_US.UTF-8")')
# robjects.r('library(iraceplot)')


# number_of_variables = 10
# solutions_size = 10
# max_evaluations = 1000
# num_runs = 2
# budget = 60


number_of_variables = 100
solutions_size = 100
max_evaluations = 25000
num_runs = 5  # Number of independent runs per problem
budget = 1000  # Total number of configurations to try per parameter

problems = [
    Sphere(number_of_variables),
    Rastrigin(number_of_variables),
    Ackley(number_of_variables),
]


def base_pso_params():
    return [
        Real("w", 0.01, 1.0),
        Real("c1", 0.01, 6.0),
        Real("c2", 0.01, 6.0),
    ]


def single_sparse_mask_params():
    return [
        Categorical("coordinate_mode", ["fraction"]),
        Real("coordinate_fraction", 0.0, 1.0),
    ]


def component_sparse_mask_params():
    return [
        Categorical("social_coordinate_mode", ["fraction"]),
        Real("social_coordinate_fraction", 0.0, 1.0),
        Categorical("cognitive_coordinate_mode", ["fraction"]),
        Real("cognitive_coordinate_fraction", 0.0, 1.0),
    ]


def sparse_single_role_params(coefficient_name: str, fraction_name: str):
    return [
        *base_pso_params(),
        Real(coefficient_name, 0.01, 6.0),
        Real(fraction_name, 0.01, 0.98),
        *single_sparse_mask_params(),
    ]


parameter_spaces = {

        # 'BoundaryReinitializedPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("pbest_gbest_epsilon", 1e-4, 0.5),
        #         Categorical("distance_metric", ["normalized_rms", "normalized_linf", "fraction_close"]),
        #         Categorical("boundary_strategy", ["random_face", "near_boundary", "mixed_boundary"]),
        #         Real("boundary_margin", 0.01, 0.3),
        #         Categorical("velocity_reset_strategy", ["zero", "random", "away_from_gbest"]),
        #         Real("velocity_scale", 0.01, 0.5),
        #         Bool("reset_personal_best_on_reinit"),
        #     ],
        # },

        'HybridPartialDisjointRestarterPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("amnesiac_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("anarchic_c", 0.01, 6.0),
                Real("rejector_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                Real("amnesiac_fraction", 0.01, 0.99),
                Real("rebel_fraction", 0.01, 0.99),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("eschewer_fraction", 0.01, 0.99),
                Real("anarchic_fraction", 0.01, 0.99),
                Bool("assign_roles_every_iteration"),
                Real("restarter_fraction", 0.01, 0.99),
                Real("convergence_threshold", 1e-4, 1.0),
            ],
        },

        'HybridFullDisjointRestarterPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("amnesiac_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("anarchic_c", 0.01, 6.0),
                Real("rejector_fraction", 0.01, 0.99),
                Real("defeatist_fraction", 0.01, 0.99),
                Real("escapist_fraction", 0.01, 0.99),
                Real("amnesiac_fraction", 0.01, 0.99),
                Real("rebel_fraction", 0.01, 0.99),
                Real("contrarian_fraction", 0.01, 0.99),
                Real("eschewer_fraction", 0.01, 0.99),
                Real("anarchic_fraction", 0.01, 0.99),
                Bool("assign_roles_every_iteration"),
                Real("restarter_fraction", 0.01, 0.99),
                Real("convergence_threshold", 1e-4, 1.0),
            ],
        },

        'HybridAdditiveRestarterPSO': {
            'params': [
                *base_pso_params(),
                Real("rejector_c", 0.01, 6.0),
                Real("defeatist_c", 0.01, 6.0),
                Real("escapist_c", 0.01, 6.0),
                Real("amnesiac_c", 0.01, 6.0),
                Real("rebel_c", 0.01, 6.0),
                Real("contrarian_c", 0.01, 6.0),
                Real("eschewer_c", 0.01, 6.0),
                Real("anarchic_c", 0.01, 6.0),
                Real("std_cognitive_prob", 0.01, 0.99),
                Real("rejector_prob", 0.01, 0.99),
                Real("defeatist_prob", 0.01, 0.99),
                Real("escapist_prob", 0.01, 0.99),
                Real("amnesiac_prob", 0.01, 0.99),
                Real("std_social_prob", 0.01, 0.99),
                Real("rebel_prob", 0.01, 0.99),
                Real("contrarian_prob", 0.01, 0.99),
                Real("eschewer_prob", 0.01, 0.99),
                Real("anarchic_prob", 0.01, 0.99),
                Bool("assign_flags_every_iteration"),
                Real("restarter_fraction", 0.01, 0.99),
                Real("convergence_threshold", 1e-4, 1.0),
            ],
        },

        # 'PGCHEA': {
        #     'params': [
        #         *base_pso_params(),
        #         Categorical("starting_algorithm", ["PSO", "GA"]),
        #         Bool("inherit_best"),
        #         Real("sbx_distribution_index", 2.0, 30.0),
        #         Real("mutation_distribution_index", 5.0, 100.0),
        #     ],
        # },

        'FRAPSO': {
            'params': [
                *base_pso_params(),
                Integer("fractal_depth", 1, 6),
                Real("convergence_threshold", 1e-4, 1.0),
            ],
        },

        'PartialResetPSO': {
            'params': [
                *base_pso_params(),
                Real("convergence_threshold", 1e-4, 1.0),
                Real("restarter_fraction", 0.01, 0.99),
            ],
        },

        'CollectiveResetPSO': {
            'params': [
                *base_pso_params(),
                Real("convergence_threshold", 1e-4, 1.0),
            ],
        },

        # 'SparseWandererPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("noise_strength", 0.01, 3.0),
        #         Real("wanderer_fraction", 0.01, 0.98),
        #         *single_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseDefeatistPSO': {
        #     'params': sparse_single_role_params("defeatist_c", "defeatist_fraction"),
        # },
        #
        # 'SparseRebelPSO': {
        #     'params': sparse_single_role_params("rebel_c", "rebel_fraction"),
        # },
        #
        # 'SparseRejectorPSO': {
        #     'params': sparse_single_role_params("rejector_c", "rejector_fraction"),
        # },
        #
        # 'SparseContrarianPSO': {
        #     'params': sparse_single_role_params("contrarian_c", "contrarian_fraction"),
        # },
        #
        # 'SparseEschewerPSO': {
        #     'params': sparse_single_role_params("eschewer_c", "eschewer_fraction"),
        # },
        #
        # 'SparseEscapistPSO': {
        #     'params': sparse_single_role_params("escapist_c", "escapist_fraction"),
        # },
        #
        # 'SparseAnarchicPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("random_strength", 0.01, 6.0),
        #         Real("anarchic_fraction", 0.01, 0.98),
        #         *single_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseAmnesiacPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("random_strength", 0.01, 6.0),
        #         Real("amnesiac_fraction", 0.01, 0.98),
        #         *single_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseErraticPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("random_strength", 0.01, 6.0),
        #         Real("erratic_fraction", 0.01, 0.98),
        #         *single_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseDrifterPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("drifter_fraction", 0.01, 0.98),
        #         Real("perturbation_scale", 0.0001, 0.10),
        #         Categorical("perturbation_method", ["gaussian", "cauchy"]),
        #         *single_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseContrarianDefeatistPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("defeatist_c", 0.01, 6.0),
        #         Real("contrarian_c", 0.01, 6.0),
        #         Real("contrarian_fraction", 0.01, 0.98),
        #         Real("defeatist_fraction", 0.01, 0.98),
        #         *component_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseRebelRejectorPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("rejector_c", 0.01, 6.0),
        #         Real("rebel_c", 0.01, 6.0),
        #         Real("rebel_fraction", 0.01, 0.98),
        #         Real("rejector_fraction", 0.01, 0.98),
        #         *component_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseEschewerEscapistPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("escapist_c", 0.01, 6.0),
        #         Real("eschewer_c", 0.01, 6.0),
        #         Real("eschewer_fraction", 0.01, 0.98),
        #         Real("escapist_fraction", 0.01, 0.98),
        #         *component_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseAnarchicAmnesiacPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("random_strength_social", 0.01, 6.0),
        #         Real("random_strength_cognitive", 0.01, 6.0),
        #         Real("anarchic_fraction", 0.01, 0.98),
        #         Real("amnesiac_fraction", 0.01, 0.98),
        #         *component_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseHybridPartialDisjointPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("rejector_c", 0.01, 6.0),
        #         Real("defeatist_c", 0.01, 6.0),
        #         Real("escapist_c", 0.01, 6.0),
        #         Real("rebel_c", 0.01, 6.0),
        #         Real("contrarian_c", 0.01, 6.0),
        #         Real("eschewer_c", 0.01, 6.0),
        #         Real("rejector_fraction", 0.01, 0.78),
        #         Real("defeatist_fraction", 0.01, 0.78),
        #         Real("escapist_fraction", 0.01, 0.78),
        #         Real("rebel_fraction", 0.01, 0.78),
        #         Real("contrarian_fraction", 0.01, 0.78),
        #         Real("eschewer_fraction", 0.01, 0.78),
        #         Bool("assign_roles_every_iteration"),
        #         *component_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseHybridFullDisjointPSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("rejector_c", 0.01, 6.0),
        #         Real("defeatist_c", 0.01, 6.0),
        #         Real("escapist_c", 0.01, 6.0),
        #         Real("rebel_c", 0.01, 6.0),
        #         Real("contrarian_c", 0.01, 6.0),
        #         Real("eschewer_c", 0.01, 6.0),
        #         Real("rejector_fraction", 0.01, 0.75),
        #         Real("defeatist_fraction", 0.01, 0.75),
        #         Real("escapist_fraction", 0.01, 0.75),
        #         Real("rebel_fraction", 0.01, 0.75),
        #         Real("contrarian_fraction", 0.01, 0.75),
        #         Real("eschewer_fraction", 0.01, 0.75),
        #         Bool("assign_roles_every_iteration"),
        #         *component_sparse_mask_params(),
        #     ],
        # },
        #
        # 'SparseHybridAdditivePSO': {
        #     'params': [
        #         *base_pso_params(),
        #         Real("rejector_c", 0.01, 6.0),
        #         Real("defeatist_c", 0.01, 6.0),
        #         Real("escapist_c", 0.01, 6.0),
        #         Real("rebel_c", 0.01, 6.0),
        #         Real("contrarian_c", 0.01, 6.0),
        #         Real("eschewer_c", 0.01, 6.0),
        #         Real("std_cognitive_prob", 0.01, 1.0),
        #         Real("rejector_prob", 0.01, 1.0),
        #         Real("defeatist_prob", 0.01, 1.0),
        #         Real("escapist_prob", 0.01, 1.0),
        #         Real("std_social_prob", 0.01, 1.0),
        #         Real("rebel_prob", 0.01, 1.0),
        #         Real("contrarian_prob", 0.01, 1.0),
        #         Real("eschewer_prob", 0.01, 1.0),
        #         Bool("assign_flags_every_iteration"),
        #         *component_sparse_mask_params(),
        #     ],
        # },

    # 'AnarchicAmnesiacPSO': {
    #     'params': [
    #         Real("c1", 0.01, 6.0),
    #         Real("c2", 0.01, 6.0),
    #         Real("w", 0.01, 1.0),
    #         Real("anarchic_fraction", 0.01, 0.98),
    #         Real("amnesiac_fraction", 0.01, 0.98),
    #         Real("random_strength_social", 0.01, 6.0),
    #         Real("random_strength_cognitive", 0.01, 6.0),
    #     ],
    # }

    'LSHADE': {
        'params': [
            Integer("initial_population_size", 20, 500),
            Integer("memory_size", 2, 50),
            Real("p_best_rate", 0.05, 0.25),
            Real("archive_size_rate", 1.0, 4.0),
        ]
    },

    # 'CMAES': {
    #     'params': [
    #         Integer("mu", 2, 100),
    #         Integer("lambda_", 10, 200)
    #     ],
    #     'forbidden': ["mu >= lambda_"]
    # },

    # 'AAAPSO': [
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("base_inertia", 0.01, 1.0),
    #     Real("min_inertia", 0.01, 1.0),
    #     Real("max_inertia", 0.01, 1.0),
    #     Real("random_strength", 0.01, 1.0),
    #     Real("anarchic_fraction", 0.01, 0.8),
    #     Real("amnesiac_fraction", 0.01, 0.8),
    #     Integer("window_size", 10, 50),
    #     Real("max_anarchic_fraction", 0.01, 0.98),
    #     Real("max_amnesiac_fraction", 0.01, 0.98),
    #     Real("diversity_threshold", 0.001, 0.3),
    #     Real("improvement_threshold", 0.0001, 0.1),
    # ],
    # 'NAPSO': [
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("base_inertia", 0.01, 1.0),
    #     Real("min_inertia", 0.01, 1.0),
    #     Real("max_inertia", 0.01, 1.0),
    #     Real("noise_strength", 0.01, 1.0),
    #     Real("noisy_fraction", 0.01, 0.8),
    #     Real("max_noisy_fraction", 0.01, 0.98),
    #     Integer("window_size", 10, 50),
    #     Real("diversity_threshold", 0.001, 0.3),
    #     Real("improvement_threshold", 0.0001, 0.1),
    # ],
    # 'HybridFullDisjointPSO_WithRandom': [
    #     Real("w", 0.01, 1.0),
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     # Real("rejector_c", 0.01, 6.0),
    #     # Real("defeatist_c", 0.01, 6.0),
    #     # Real("escapist_c", 0.01, 6.0),
    #     Real("amnesiac_c", 0.01, 6.0),
    #     # Real("rebel_c", 0.01, 6.0),
    #     # Real("contrarian_c", 0.01, 6.0),
    #     # Real("eschewer_c", 0.01, 6.0),
    #     Real("anarchic_c", 0.01, 6.0),
    #     # Real("rejector_fraction", 0.01, 0.73),
    #     # Real("defeatist_fraction", 0.01, 0.73),
    #     # Real("escapist_fraction", 0.01, 0.73),
    #     Real("amnesiac_fraction", 0.01, 0.98),
    #     # Real("rebel_fraction", 0.01, 0.73),
    #     # Real("contrarian_fraction", 0.01, 0.73),
    #     # Real("eschewer_fraction", 0.01, 0.73),
    #     Real("anarchic_fraction", 0.01, 0.98),
    #     Bool("assign_roles_every_iteration"),
    # ],
    # 'HybridPartialDisjointPSO_WithRandom': [
    #     Real("w", 0.01, 1.0),
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     # Real("rejector_c", 0.01, 6.0),
    #     # Real("defeatist_c", 0.01, 6.0),
    #     # Real("escapist_c", 0.01, 6.0),
    #     # Real("rebel_c", 0.01, 6.0),
    #     # Real("contrarian_c", 0.01, 6.0),
    #     # Real("eschewer_c", 0.01, 6.0),
    #     Real("amnesiac_c", 0.01, 6.0),
    #     Real("anarchic_c", 0.01, 6.0),
    #     # Real("rejector_fraction", 0.01, 0.77),
    #     # Real("defeatist_fraction", 0.01, 0.77),
    #     # Real("escapist_fraction", 0.01, 0.77),
    #     Real("amnesiac_fraction", 0.01, 0.98),
    #     # Real("rebel_fraction", 0.01, 0.77),
    #     # Real("contrarian_fraction", 0.01, 0.77),
    #     # Real("eschewer_fraction", 0.01, 0.77),
    #     Real("anarchic_fraction", 0.01, 0.98),
    #     Bool("assign_roles_every_iteration"),
    # ],
    # 'HybridAdditivePSO_WithRandom': [
    #     Real("w", 0.01, 1.0),
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     # Real("rejector_c", 0.01, 6.0),
    #     # Real("defeatist_c", 0.01, 6.0),
    #     # Real("escapist_c", 0.01, 6.0),
    #     # Real("rebel_c", 0.01, 6.0),
    #     # Real("contrarian_c", 0.01, 6.0),
    #     # Real("eschewer_c", 0.01, 6.0),
    #     Real("anarchic_c", 0.01, 6.0),
    #     Real("amnesiac_c", 0.01, 6.0),
    #     Real("std_cognitive_prob", 0.01, 1.0),
    #     # Real("rejector_prob", 0.01, 1.0),
    #     # Real("defeatist_prob", 0.01, 1.0),
    #     # Real("escapist_prob", 0.01, 1.0),
    #     Real("amnesiac_prob", 0.01, 1.0),
    #     Real("std_social_prob", 0.01, 1.0),
    #     # Real("rebel_prob", 0.01, 1.0),
    #     # Real("contrarian_prob", 0.01, 1.0),
    #     # Real("eschewer_prob", 0.01, 1.0),
    #     Real("anarchic_prob", 0.01, 1.0),
    #     Bool("assign_flags_every_iteration"),
    # ],
    #
    # 'DrifterPSO': [
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("w", 0.01, 1.0),
    #     Real("drifter_fraction", 0.01, 0.98),
    #     Real("perturbation_scale", 0.0001, 0.10),
    # ],
    #
    # 'DAPSO': [
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("base_inertia", 0.01, 1.0),
    #     Real("min_inertia", 0.01, 1.0),
    #     Real("max_inertia", 0.01, 1.0),
    #     Real("perturbation_scale", 0.0001, 0.1),
    #     Real("drifter_fraction", 0.01, 0.98),
    #     Real("max_drifter_fraction", 0.01, 0.98),
    #     Integer("window_size", 5, 50),
    #     Real("diversity_threshold", 0.001, 0.30),
    #     Real("improvement_threshold", 0.0001, 0.10),
    # ],
    #
    # 'CLAPSO': [
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("cl_c1", 0.01, 6.0),
    #     Real("cl_c2", 0.01, 6.0),
    #     Real("b1", 0.0, 1.0),
    #     Real("b2", 0.0, 1.0),
    #     Real("base_inertia", 0.01, 1.0),
    #     Real("min_inertia", 0.01, 1.0),
    #     Real("max_inertia", 0.01, 1.0),
    #     Real("cl_fraction", 0.01, 0.98),
    #     Real("max_cl_fraction", 0.01, 0.98),
    #     Integer("window_size", 5, 50),
    #     Real("diversity_threshold", 0.001, 0.3),
    #     Real("improvement_threshold", 0.0001, 0.1),
    # ],
    #
    # 'HybridFullDisjointRestarterPSO': [
    #     Real("w", 0.01, 1.0),
    #     Real("c1", 0.01, 6.0),
    #     Real("rejector_c", 0.01, 6.0),
    #     Real("defeatist_c", 0.01, 6.0),
    #     Real("escapist_c", 0.01, 6.0),
    #     Real("amnesiac_c", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("rebel_c", 0.01, 6.0),
    #     Real("contrarian_c", 0.01, 6.0),
    #     Real("eschewer_c", 0.01, 6.0),
    #     Real("anarchic_c", 0.01, 6.0),
    #     Real("rejector_fraction", 0.01, 0.71),
    #     Real("defeatist_fraction", 0.01, 0.71),
    #     Real("escapist_fraction", 0.01, 0.71),
    #     Real("amnesiac_fraction", 0.01, 0.71),
    #     Real("rebel_fraction", 0.01, 0.71),
    #     Real("contrarian_fraction", 0.01, 0.71),
    #     Real("eschewer_fraction", 0.01, 0.71),
    #     Real("anarchic_fraction", 0.01, 0.71),
    #     Bool("assign_roles_every_iteration"),
    #     Real("restarter_fraction", 0.01, 0.71),
    #     Real("convergence_threshold", 0.0001, 0.1),
    # ],
    #
    # 'HybridPartialDisjointRestarterPSO': [
    #     Real("w", 0.01, 1.0),
    #     Real("c1", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("rejector_c", 0.01, 6.0),
    #     Real("defeatist_c", 0.01, 6.0),
    #     Real("escapist_c", 0.01, 6.0),
    #     Real("amnesiac_c", 0.01, 6.0),
    #     Real("rebel_c", 0.01, 6.0),
    #     Real("contrarian_c", 0.01, 6.0),
    #     Real("eschewer_c", 0.01, 6.0),
    #     Real("anarchic_c", 0.01, 6.0),
    #     Real("restarter_fraction", 0.01, 0.76),
    #     Real("rejector_fraction", 0.01, 0.76),
    #     Real("defeatist_fraction", 0.01, 0.76),
    #     Real("escapist_fraction", 0.01, 0.76),
    #     Real("amnesiac_fraction", 0.01, 0.76),
    #     Real("rebel_fraction", 0.01, 0.76),
    #     Real("contrarian_fraction", 0.01, 0.76),
    #     Real("eschewer_fraction", 0.01, 0.76),
    #     Real("anarchic_fraction", 0.01, 0.76),
    #     Bool("assign_roles_every_iteration"),
    #     Real("convergence_threshold", 0.0001, 0.1),
    # ],
    #
    # 'HybridAdditiveRestarterPSO': [
    #     Real("w", 0.01, 1.0),
    #     Real("c1", 0.01, 6.0),
    #     Real("rejector_c", 0.01, 6.0),
    #     Real("defeatist_c", 0.01, 6.0),
    #     Real("escapist_c", 0.01, 6.0),
    #     Real("amnesiac_c", 0.01, 6.0),
    #     Real("c2", 0.01, 6.0),
    #     Real("rebel_c", 0.01, 6.0),
    #     Real("contrarian_c", 0.01, 6.0),
    #     Real("eschewer_c", 0.01, 6.0),
    #     Real("anarchic_c", 0.01, 6.0),
    #     Real("std_cognitive_prob", 0.01, 1.0),
    #     Real("rejector_prob", 0.01, 1.0),
    #     Real("defeatist_prob", 0.01, 1.0),
    #     Real("escapist_prob", 0.01, 1.0),
    #     Real("amnesiac_prob", 0.01, 1.0),
    #     Real("std_social_prob", 0.01, 1.0),
    #     Real("rebel_prob", 0.01, 1.0),
    #     Real("contrarian_prob", 0.01, 1.0),
    #     Real("eschewer_prob", 0.01, 1.0),
    #     Real("anarchic_prob", 0.01, 1.0),
    #     Bool("assign_flags_every_iteration"),
    #     Real("restarter_fraction", 0.01, 0.8),
    #     Real("convergence_threshold", 0.0001, 0.1),
    # ],
}

current_algorithm = None


# ==============================================================================
# Constraint Handling Functions
# ==============================================================================
def normalize_fraction_sum(fractions_dict: dict, max_sum: float = 1.0) -> dict:
    """Normalizes fractions in a dictionary if their sum exceeds max_sum."""
    current_sum = sum(v for v in fractions_dict.values() if isinstance(v, (int, float)))
    numeric_fractions = {k:v for k,v in fractions_dict.items() if isinstance(v, (int, float))}
    normalized_fractions = numeric_fractions.copy() # Work with numeric copy

    if current_sum > max_sum + 1e-9:
        print(f"Normalizing fractions (Sum: {current_sum:.4f} > {max_sum})")
        if current_sum > 1e-9:
            factor = max_sum / current_sum
            for role in normalized_fractions:
                normalized_fractions[role] *= factor
        else:
            for role in normalized_fractions: normalized_fractions[role] = 0.0
    return normalized_fractions


def repair_max_param_constraints_random(config: dict) -> dict:
    """
    Repairs constraints like max_param >= param and min <= base <= max.
    Uses swapping for invalid min/max ranges, random repair for base, clamping for simple max >= param.
    """
    repaired_config = config.copy()
    # Max >= Param constraints (clamp max = param)
    constraints_to_check = [
        ("c1", "max_c1"), ("c2", "max_c2"),
        ("eschewer_fraction", "max_eschewer_fraction"), ("escapist_fraction", "max_escapist_fraction"),
        ("contrarian_fraction", "max_contrarian_fraction"), ("defeatist_fraction", "max_defeatist_fraction"),
        ("rebel_fraction", "max_rebel_fraction"), ("rejector_fraction", "max_rejector_fraction"),
        ("anarchic_fraction", "max_anarchic_fraction"), ("amnesiac_fraction", "max_amnesiac_fraction"),
        ("noisy_fraction", "max_noisy_fraction"), ("cl_fraction", "max_cl_fraction"),
        ("drifter_fraction", "max_drifter_fraction")
    ]
    for param, max_param in constraints_to_check:
        if param in repaired_config and max_param in repaired_config:
            param_val = repaired_config[param]; max_param_val = repaired_config[max_param]
            if isinstance(param_val, (int, float)) and isinstance(max_param_val, (int, float)):
                if max_param_val < param_val:
                    print(f"Repair constraint: {max_param}<{param}. Clamp {max_param}={param_val:.4f}")
                    repaired_config[max_param] = param_val

    # Min <= Base <= Max constraints (swap min/max, random repair base)
    min_base_max_triplets = [("min_inertia", "base_inertia", "max_inertia")]
    for min_key, base_key, max_key in min_base_max_triplets:
        if min_key in repaired_config and base_key in repaired_config and max_key in repaired_config:
            min_val = repaired_config[min_key]; base_val = repaired_config[base_key]; max_val = repaired_config[max_key]
            if not (isinstance(min_val, (int, float)) and isinstance(base_val, (int, float)) and isinstance(max_val, (int, float))): continue
            if max_val < min_val:
                print(f"Repair bounds: {max_key}<{min_key}. Swap.")
                repaired_config[min_key], repaired_config[max_key] = max_val, min_val
                min_val, max_val = max_val, min_val # Update local vars
            # Step 2: Repair base_val if outside [min_val, max_val]
            if base_val < min_val or base_val > max_val:
                 if max_val >= min_val:
                      new_base = random.uniform(min_val, max_val)
                      print(f"Repair value: {base_key} ({base_val:.4f}) outside [{min_val:.4f}, {max_val:.4f}]. Set to random: {new_base:.4f}")
                      repaired_config[base_key] = new_base
                 else: # Should not happen after swap, but safety check
                      print(f"Warning: Invalid range [{min_val:.4f}, {max_val:.4f}] after repair for {base_key}. Clamping to min.")
                      repaired_config[base_key] = min_val

    return repaired_config

# ==============================================================================
# Main Target Runner Function
# ==============================================================================

def derive_run_seed(base_seed: int, problem_name: str, run_index: int) -> int:
    identity = f"{base_seed}:{problem_name}:{run_index}"
    digest = hashlib.sha256(identity.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], byteorder="little", signed=False)


def target_runner(experiment: Experiment, scenario: Scenario) -> float:
    """
    Universal target runner. Repairs max/min constraints.
    Performs fraction normalization *within the runner* based on algorithm type.
    """
    global current_algorithm, problems, num_runs, solutions_size, max_evaluations # Access globals

    if current_algorithm is None: return float('inf')
    config = experiment.configuration
    # print(f"\n--- Evaluating Algorithm: {current_algorithm} ---") # Verbose
    # print(f"Received Config: {config}") # Verbose

    # --- Step 1: Repair Dependent Max/Min Parameter Constraints ---
    repaired_config = repair_max_param_constraints_random(config)
    # if repaired_config != config: print(f"Config after repair: {repaired_config}") # Verbose

    # --- Step 2: Handle Fraction Normalization (Algorithm Specific) ---
    final_params = repaired_config.copy() # Start with repaired params

    # Define parameter groups
    cognitive_fraction_keys = ["rejector_fraction", "defeatist_fraction", "escapist_fraction", "amnesiac_fraction"]
    social_fraction_keys = ["rebel_fraction", "contrarian_fraction", "eschewer_fraction", "anarchic_fraction"]
    all_special_fraction_keys = cognitive_fraction_keys + social_fraction_keys + ["wanderer_fraction"] # Assuming wanderer might exist

    # --- PGCHEA special handling ---
    if current_algorithm == 'PGCHEA':
        final_params['solutions_size'] = solutions_size
        final_params['crossover'] = SBXCrossover(
            probability=1.0,
            distribution_index=final_params.pop('sbx_distribution_index', 5.0),
        )
        final_params['mutation'] = PolynomialMutation(
            probability=1.0 / number_of_variables,
            distribution_index=final_params.pop('mutation_distribution_index', 20.0),
        )

    # --- Normalization for Partial Disjoint Variants ---
    if current_algorithm in [
        'HybridPartialDisjointPSO',
        'HybridPartialDisjointPSO_WithRandom',
        'HybridPartialDisjointRestarterPSO',
        'SparseHybridPartialDisjointPSO',
    ]:
        # Normalize cognitive group
        cog_fractions = {k: repaired_config.get(k, 0.0) for k in cognitive_fraction_keys if k in repaired_config}
        if cog_fractions:
             norm_cog = normalize_fraction_sum(cog_fractions, 1.0)
             final_params.update(norm_cog) # Update final_params dict

        # Normalize social group
        soc_fractions = {k: repaired_config.get(k, 0.0) for k in social_fraction_keys if k in repaired_config}
        if soc_fractions:
             norm_soc = normalize_fraction_sum(soc_fractions, 1.0)
             final_params.update(norm_soc)

        # Optional logging of final fractions passed
        # cog_final_str = ", ".join([f"{k.split('_')[0][:3]}={final_params.get(k, 0.0):.2f}" for k in cognitive_fraction_keys if k in final_params])
        # soc_final_str = ", ".join([f"{k.split('_')[0][:3]}={final_params.get(k, 0.0):.2f}" for k in social_fraction_keys if k in final_params])
        # print(f"  Using PartialDisjoint Fractions: Cog=[{cog_final_str}] Soc=[{soc_final_str}]")

    # --- Normalization for Full Disjoint Variants ---
    elif current_algorithm in [
        'HybridFullDisjointPSO',
        'HybridFullDisjointPSO_WithRandom',
        'HybridFullDisjointRestarterPSO',
        'HybridDisjointPSO_WithWanderer',
        'SparseHybridFullDisjointPSO',
    ]:
        # Normalize sum of ALL special fractions to be <= 1.0
        all_special_fractions = {k: repaired_config.get(k, 0.0) for k in all_special_fraction_keys if k in repaired_config}
        if all_special_fractions:
            norm_special = normalize_fraction_sum(all_special_fractions, 1.0)
            final_params.update(norm_special) # Update final_params dict

        # Optional logging
        # frac_final_str = ", ".join([f"{k.split('_')[0][:3]}={final_params.get(k, 0.0):.2f}" for k in all_special_fraction_keys if k in final_params])
        # print(f"  Using FullDisjoint Fractions: [{frac_final_str}]")

    # if current_algorithm == 'CMAES':
    #     if 'mu' in repaired_config and 'lambda_' in repaired_config:
    #         if repaired_config['mu'] >= repaired_config['lambda_']:
    #             repaired_config['mu'] = max(2, int(repaired_config['lambda_'] / 2))


    # --- No fraction normalization needed for Additive or other types ---

    # --- Step 3: Get Algorithm Class ---
    try:
        AlgorithmClass = globals()[current_algorithm]
    except KeyError:
        print(f"ERROR: Algorithm class '{current_algorithm}' not found.")
        return float('inf')

    # --- Step 4: Run Experiments ---
    results = []
    problem_obj_for_run = None # To hold the problem instance for this set of runs
    for problem in problems:
        problem_name = problem.get_name() if hasattr(problem, 'get_name') else problem.__class__.__name__
        problem_obj_for_run = problem # Use the selected problem
        # print(f"  Problem: {problem_name}") # Verbose
        for run_index in range(num_runs):
            try:
                # Seed the algorithm RNGs for this run (irace supplies a
                # per-experiment seed; runs and problems get distinct streams).
                run_seed = derive_run_seed(experiment.seed, problem_name, run_index)
                random.seed(run_seed)
                np.random.seed(run_seed)
                if hasattr(problem_obj_for_run, "set_seed"):
                    problem_obj_for_run.set_seed(run_seed)

                # Prepare parameters, FILTERING based on the specific AlgorithmClass constructor
                constructor_params = {"problem": problem_obj_for_run, # Use the problem instance
                                      "swarm_size": solutions_size,
                                      "termination_criterion": StoppingByEvaluations(max_evaluations)}
                constructor_params.update(final_params) # Add repaired/normalized params

                sig = inspect.signature(AlgorithmClass.__init__)
                allowed_params = {k for k in sig.parameters if k != 'self'}
                use_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

                if use_kwargs:
                    filtered_constructor_params = constructor_params # Pass all if **kwargs allowed
                else:
                    filtered_constructor_params = {k: v for k, v in constructor_params.items() if k in allowed_params}

                # Instantiate
                algorithm = AlgorithmClass(**filtered_constructor_params)
                algorithm.run()
                result = algorithm.result()

                if result is None or not hasattr(result, 'objectives') or not result.objectives:
                     results.append(float('inf'))
                else:
                    results.append(result.objectives[0])

            except Exception as e:
                print(f"      ERROR during run {run_index + 1} on {problem_name} for {current_algorithm}: {e}")
                # Print config for debugging, NOT filtered params as they vary per algo
                print(f"      Config (Original): {config}")
                print(f"      Config (Repaired): {repaired_config}")
                print(f"      Final Params Passed (Subset): {{k:v for k,v in filtered_constructor_params.items() if k not in ['problem','termination_criterion']}}") # Show relevant params
                traceback.print_exc()
                results.append(float('inf'))
        # Break after first problem if only tuning on one for speed, otherwise loops through all
        # break # Uncomment for faster testing on one problem

    # --- Step 5: Process Results ---
    # print(f"--- Finished runs for config. Num results: {len(results)} ---")
    if not results: return float('inf')
    try:
        numeric_results = [r for r in results if isinstance(r, (int, float)) and np.isfinite(r)]
        if not numeric_results: return float('inf')
        avg_result = np.mean(numeric_results)
    except Exception as e: print(f"Error calculating mean: {e}"); return float('inf')
    cost = float(avg_result) if np.isfinite(avg_result) else float('inf')
    # Log original config for irace, but final cost
    print(f"Evaluated config ({final_params}) -> Final Avg Cost: {cost:.4f}")
    return cost


if __name__ == "__main__":
    best_configurations = {}
    output_file = "irace_best_configurations.json"

    for algo_name, space_config in parameter_spaces.items():
        current_algorithm = algo_name
        print(f"Optimizing parameters for {algo_name} ...")

        # Unpack the parameters list and the forbidden expression from the config
        params_list = space_config['params']
        forbidden_expression = space_config.get('forbidden', None)

        # Create the ParameterSpace using the extracted components
        parameter_space = ParameterSpace(params=params_list, forbidden=forbidden_expression)

        scenario = Scenario(max_experiments=budget * len(params_list), seed=42, n_jobs=48)

        result = irace(target_runner, parameter_space, scenario, return_df=True, remove_metadata=True)
        best_configurations[algo_name] = result

        # 2. Load that RData into R’s global env (it creates `iraceResults`)
        # robjects.r['load']("irace.log")

        # 3. Tell IRACE to dump the human‑readable log to a .txt file
        # robjects.r['save_irace_logfile'](robjects.r['iraceResults'], "irace.txt")

        # Save results after each algorithm
        with open(output_file, "w") as f:
            json.dump({k: v.to_json() for k, v in best_configurations.items()}, f, indent=4)
