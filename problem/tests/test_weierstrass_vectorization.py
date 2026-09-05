"""Certification of the vectorized ShiftedRotatedWeierstrass.

The reference below is the original pure-Python double loop, verbatim. The
vectorized version must agree to tight tolerance, the optimum must sit at
the shift vector, and the seeded instance identity must be untouched (the
precomputed constants consume no RNG).
"""

import math

import numpy as np
import pytest
from jmetal.core.solution import FloatSolution

from experiment.problem_identity import create_seeded_problem
from problem.n_variables.weierstrass import ShiftedRotatedWeierstrass


def _reference(problem, variables):
    """Original loop implementation, kept verbatim as the reference."""
    x = np.array(variables)
    rotated = np.dot(problem.rotation_matrix, x - problem.shift)

    k_max = 20
    a = 0.5
    b = 3
    sum1 = sum([a ** k * math.cos(2 * math.pi * b ** k * (xi + 0.5))
                for k in range(k_max + 1) for xi in rotated])
    sum2 = sum([a ** k * math.cos(2 * math.pi * b ** k * 0.5)
                for k in range(k_max + 1)])
    return sum1 - problem.number_of_variables() * sum2


def _evaluate(problem, variables):
    s = FloatSolution(problem.lower_bound, problem.upper_bound, 1, 0)
    s.objectives = [0.0]
    s.variables = list(variables)
    return problem.evaluate(s).objectives[0]


@pytest.mark.parametrize("dim", [2, 10, 100, 1000])
def test_vectorized_matches_reference(dim):
    problem = create_seeded_problem(ShiftedRotatedWeierstrass, dim, 42)
    rng = np.random.default_rng(dim)
    for _ in range(3):
        x = rng.uniform(-0.5, 0.5, size=dim)
        expected = _reference(problem, x)
        # abs floor matters near the optimum where the value approaches 0.
        assert _evaluate(problem, x) == pytest.approx(expected, rel=1e-12, abs=1e-9)


def test_optimum_is_at_the_shift_vector():
    problem = create_seeded_problem(ShiftedRotatedWeierstrass, 50, 42)
    value_at_shift = _evaluate(problem, problem.shift)
    assert value_at_shift == pytest.approx(0.0, abs=1e-8)

    # Neighborhood check: moving off the shift worsens the objective.
    rng = np.random.default_rng(5)
    for _ in range(3):
        perturbed = problem.shift + rng.uniform(-0.05, 0.05, size=50)
        assert _evaluate(problem, perturbed) > value_at_shift


def test_seeded_instance_identity_is_preserved():
    # Fingerprints captured on the pre-vectorization implementation: the
    # precomputed constants must not consume RNG or alter shift/rotation.
    expected = {
        10: "e3e97799b50854408b9dff02666bc3c6ab95a605a37dc4d5c9dc85102218e6a7",
        100: "02c0da467ed0329076b158e99dec22ebcef6f5c6314a1ea07d7532561bdc9e21",
    }
    for dim, fingerprint in expected.items():
        problem = create_seeded_problem(ShiftedRotatedWeierstrass, dim, 42)
        assert problem.instance_id == fingerprint
