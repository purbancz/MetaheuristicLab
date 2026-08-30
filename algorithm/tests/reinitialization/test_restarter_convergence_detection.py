import pytest
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.role_based.role_hybrids import (
    HybridAdditiveRestarterPSO,
    HybridFullDisjointRestarterPSO,
    HybridPartialDisjointRestarterPSO,
)

_N_VARS = 3
_SWARM = 4


def _collapsed_restarters(n, n_vars):
    solutions = []
    for _ in range(n):
        s = FloatSolution([-5.0] * n_vars, [5.0] * n_vars, 1, 0)
        s.variables = [1.0] * n_vars
        s.objectives = [0.0]
        s.attributes = {
            "is_restarter": True,
            "velocity": [0.0] * n_vars,
            "best_position": [1.0] * n_vars,
            "best_objective": 0.0,
        }
        solutions.append(s)
    return solutions


@pytest.mark.parametrize(
    "cls",
    [
        HybridPartialDisjointRestarterPSO,
        HybridFullDisjointRestarterPSO,
        HybridAdditiveRestarterPSO,
    ],
)
def test_restarter_detects_collapsed_swarm(cls):
    problem = Sphere(_N_VARS)
    alg = cls(
        problem=problem,
        swarm_size=_SWARM,
        termination_criterion=StoppingByEvaluations(max_evaluations=_SWARM),
        w=0.5,
    )
    alg.solutions = _collapsed_restarters(_SWARM, _N_VARS)
    alg.convergence_threshold = 1e10
    assert alg.check_convergence()
