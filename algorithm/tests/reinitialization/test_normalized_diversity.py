import numpy as np
import pytest
from jmetal.problem import Sphere
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.diversity import normalized_swarm_diversity
from algorithm.reinitialization.reinitialized_pso import PartialResetPSO
from algorithm.role_based.role_hybrids import HybridPartialDisjointRestarterPSO


def test_normalized_diversity_is_scale_invariant():
    rng = np.random.default_rng(7)
    base = rng.uniform(0.0, 1.0, size=(30, 20))

    small = normalized_swarm_diversity(base, [0.0] * 20, [1.0] * 20)
    scaled = normalized_swarm_diversity(base * 1000.0, [0.0] * 20, [1000.0] * 20)
    shifted = normalized_swarm_diversity(base * 1000.0 - 500.0, [-500.0] * 20, [500.0] * 20)

    assert small == pytest.approx(scaled)
    assert small == pytest.approx(shifted)
    # A uniform random swarm sits near 0.28 regardless of bounds/dimension.
    assert 0.2 < small < 0.35


def test_normalized_diversity_is_dimension_stable():
    rng = np.random.default_rng(11)
    low_dim = normalized_swarm_diversity(
        rng.uniform(-5.0, 5.0, size=(50, 10)), [-5.0] * 10, [5.0] * 10
    )
    high_dim = normalized_swarm_diversity(
        rng.uniform(-5.0, 5.0, size=(50, 1000)), [-5.0] * 1000, [5.0] * 1000
    )
    assert low_dim == pytest.approx(high_dim, abs=0.05)


def _make_algorithm(cls, threshold, **kwargs):
    return cls(
        problem=Sphere(3),
        swarm_size=4,
        w=0.5,
        c1=1.5,
        c2=1.5,
        convergence_threshold=threshold,
        termination_criterion=StoppingByEvaluations(max_evaluations=12),
        **kwargs,
    )


def test_partial_reset_counts_restarts_and_reports_them():
    # Threshold 0.9 exceeds any real swarm diversity (~0.28 for a uniform
    # swarm), so the trigger fires on every step.
    algorithm = _make_algorithm(PartialResetPSO, 0.9, restarter_fraction=0.5)
    algorithm.run()

    assert algorithm.total_restarts >= 1
    assert algorithm.observable_data()["TOTAL_RESTARTS"] == algorithm.total_restarts


def test_partial_reset_zero_threshold_never_restarts():
    algorithm = _make_algorithm(PartialResetPSO, 0.0, restarter_fraction=0.5)
    algorithm.run()

    assert algorithm.total_restarts == 0
    assert algorithm.observable_data()["TOTAL_RESTARTS"] == 0


def _solution(variables, lower, upper, is_restarter):
    s = FloatSolution(lower, upper, 1, 0)
    s.objectives = [0.0]
    s.variables = list(variables)
    s.attributes["is_restarter"] = is_restarter
    return s


def test_hybrid_restarter_trigger_uses_normalized_restarter_subset():
    algorithm = HybridPartialDisjointRestarterPSO(
        problem=Sphere(2),
        swarm_size=4,
        w=0.5,
        c1=1.5,
        c2=1.5,
        convergence_threshold=0.05,
        restarter_fraction=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=8),
    )
    lower, upper = [-5.12, -5.12], [5.12, 5.12]

    # Restarter subset collapsed into a tiny cluster (~0.1% of the diagonal);
    # non-restarters far apart must not mask the collapse.
    algorithm.solutions = [
        _solution([0.0, 0.0], lower, upper, True),
        _solution([0.01, 0.0], lower, upper, True),
        _solution([-5.0, -5.0], lower, upper, False),
        _solution([5.0, 5.0], lower, upper, False),
    ]
    assert algorithm.check_convergence() is True

    # Restarter subset spread across the domain: no trigger.
    algorithm.solutions = [
        _solution([-5.0, -5.0], lower, upper, True),
        _solution([5.0, 5.0], lower, upper, True),
        _solution([0.0, 0.0], lower, upper, False),
        _solution([0.01, 0.0], lower, upper, False),
    ]
    assert algorithm.check_convergence() is False


def test_hybrid_partial_disjoint_restarter_reports_its_own_name():
    algorithm = HybridPartialDisjointRestarterPSO(
        problem=Sphere(2),
        swarm_size=4,
        w=0.5,
        c1=1.5,
        c2=1.5,
        convergence_threshold=0.05,
        restarter_fraction=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=8),
    )
    assert algorithm.get_name() == "HybridPartialDisjointRestarterPSO"
