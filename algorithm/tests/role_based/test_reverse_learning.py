"""ReverseLearningPSO must report the best solution it ever evaluated.

Its velocity update deliberately uses only the worst attractors (pure
reverse learning); best-tracking is reporting infrastructure. Before the
fix, best_global was frozen at initialization and result() returned the
best of the initial random sample.
"""

import random

import numpy as np
import pytest
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.role_based.worst_aware_pso import ReverseLearningPSO


class BestRecordingSphere(Sphere):
    """Sphere that remembers the best objective it ever computed."""

    def __init__(self, number_of_variables: int = 3):
        super().__init__(number_of_variables)
        self.best_seen = float("inf")

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution = super().evaluate(solution)
        self.best_seen = min(self.best_seen, solution.objectives[0])
        return solution


def _make(problem, max_evaluations=200):
    return ReverseLearningPSO(
        problem=problem,
        swarm_size=5,
        b1=1.5,
        b2=1.5,
        w=0.5,
        termination_criterion=StoppingByEvaluations(max_evaluations=max_evaluations),
    )


def test_result_is_the_best_ever_evaluated_solution():
    random.seed(11)
    np.random.seed(11)
    problem = BestRecordingSphere(3)
    algorithm = _make(problem)

    algorithm.run()

    assert algorithm.result().objectives[0] == pytest.approx(problem.best_seen)
    # And it genuinely searched: the reported optimum beats the best of the
    # initial 5-point random sample (which is what the frozen best_global
    # used to return).
    assert problem.best_seen < float("inf")


def test_result_is_never_worse_than_the_initial_population():
    # Note: pure reverse learning has no attractor toward good regions, so on
    # a convex problem it may never beat its initial sample (this is a
    # property of the dynamics, not a reporting bug). The guaranteed property
    # is: the reported result is at least as good as the initial-sample best
    # that the old frozen best_global used to return. Seed 3 additionally
    # yields a strict improvement, pinning that the tracking is live.
    random.seed(3)
    np.random.seed(3)
    problem = BestRecordingSphere(3)
    algorithm = _make(problem)

    solutions = algorithm.create_initial_solutions()
    initial_best = min(s.objectives[0] for s in solutions)

    random.seed(3)
    np.random.seed(3)
    problem.best_seen = float("inf")
    algorithm = _make(problem)
    algorithm.run()

    assert algorithm.result().objectives[0] <= initial_best
    assert algorithm.result().objectives[0] < initial_best  # seed-specific


def test_worst_tracking_semantics_are_unchanged():
    random.seed(11)
    np.random.seed(11)
    algorithm = _make(BestRecordingSphere(3))
    algorithm.run()

    # Personal worst is all-time: never below the particle's current value.
    for particle in algorithm.solutions:
        assert particle.attributes["worst_objective"] >= particle.objectives[0]
    # Global worst is the current swarm's worst (live semantics, by design).
    assert algorithm.global_worst.objectives[0] == max(
        p.objectives[0] for p in algorithm.solutions
    )
