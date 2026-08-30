from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.reinitialization.reinitialized_pso import CollectiveResetPSO, PartialResetPSO
from algorithm.role_based.role_hybrids import HybridPartialDisjointRestarterPSO

_STALE = 99.9
_N = 3


def _particle(n_vars, with_worst=False):
    s = FloatSolution([-5.0] * n_vars, [5.0] * n_vars, 1, 0)
    s.variables = [1.0] * n_vars
    s.objectives = [_STALE]
    s.attributes = {
        'velocity': [0.0] * n_vars,
        'best_position': [1.0] * n_vars,
        'best_objective': _STALE,
        'is_restarter': True,
    }
    if with_worst:
        s.attributes['worst_position'] = [1.0] * n_vars
        s.attributes['worst_objective'] = _STALE
    return s


def _criterion():
    return StoppingByEvaluations(max_evaluations=4)


def test_partial_reset_pso_clears_stale_best_objective():
    problem = Sphere(_N)
    alg = PartialResetPSO(
        problem=problem, swarm_size=4, w=0.5, c1=1.0, c2=1.0,
        termination_criterion=_criterion(),
    )
    alg.solutions = [_particle(_N) for _ in range(4)]
    alg.selective_reinitialization()
    for p in alg.solutions:
        assert p.attributes['best_objective'] != _STALE


def test_collective_reset_pso_clears_stale_best_objective():
    problem = Sphere(_N)
    alg = CollectiveResetPSO(
        problem=problem, swarm_size=4, w=0.5, c1=1.0, c2=1.0,
        termination_criterion=_criterion(),
    )
    alg.solutions = [_particle(_N) for _ in range(4)]
    alg.reinitialize_swarm()
    for p in alg.solutions:
        assert p.attributes['best_objective'] != _STALE


def test_hybrid_restarter_pso_clears_stale_best_and_worst_objective():
    problem = Sphere(_N)
    alg = HybridPartialDisjointRestarterPSO(
        problem=problem, swarm_size=4,
        termination_criterion=_criterion(),
        w=0.5,
    )
    alg.solutions = [_particle(_N, with_worst=True) for _ in range(4)]
    alg.selective_reinitialization()
    for p in alg.solutions:
        assert p.attributes['best_objective'] != _STALE
        assert p.attributes['worst_objective'] != _STALE
