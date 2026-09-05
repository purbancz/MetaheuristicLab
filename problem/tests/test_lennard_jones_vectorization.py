"""Certification of the vectorized Lennard-Jones evaluate().

The reference implementation below is the original pure-Python pair loop,
verbatim. The vectorized version must agree to tight relative tolerance on
random configurations, edge cases (clamped near-contact pairs, leftover
variables when n % 3 != 0), and the known certification point.
"""

import numpy as np
import pytest
from jmetal.core.solution import FloatSolution

from problem.n_variables.lenard_johnes_minimum_energy_cluster import (
    LennardJonesMinimumEnergyCluster,
)


def _reference_energy(variables, number_of_variables):
    """Original loop implementation kept as the reference (minus the removed
    +12.712062 offset, so values are CCD-comparable cluster energies)."""
    def d(i, j, variables):
        sum_k = 0
        for k in range(3):
            sum_k += (variables[3 * i + k] - variables[3 * j + k]) ** 2
        return sum_k ** 3

    s = 0.0
    num_points = number_of_variables // 3
    epsilon = 1e-6
    for i in range(num_points - 1):
        sum_j = 0
        for j in range(i + 1, num_points):
            d_tmp = d(i, j, variables)
            if d_tmp < epsilon:
                d_tmp = epsilon
            sum_j += (1 / (d_tmp ** 2)) - (2 / d_tmp)
        s += sum_j
    return s


def _evaluate(problem, variables):
    s = FloatSolution(problem.lower_bound, problem.upper_bound, 1, 0)
    s.objectives = [0.0]
    s.variables = list(variables)
    return problem.evaluate(s).objectives[0]


@pytest.mark.parametrize("n_vars", [6, 9, 10, 30, 100])
def test_vectorized_matches_reference_on_random_configurations(n_vars):
    rng = np.random.default_rng(42)
    problem = LennardJonesMinimumEnergyCluster(number_of_variables=n_vars)
    for _ in range(5):
        x = rng.uniform(-3.0, 3.0, size=n_vars)
        expected = _reference_energy(list(x), n_vars)
        actual = _evaluate(problem, x)
        assert actual == pytest.approx(expected, rel=1e-12)


def test_vectorized_matches_reference_at_campaign_dimension():
    n_vars = 1000  # 333 atoms + 1 ignored leftover variable
    rng = np.random.default_rng(7)
    problem = LennardJonesMinimumEnergyCluster(number_of_variables=n_vars)
    x = rng.uniform(-5.0, 5.0, size=n_vars)
    expected = _reference_energy(list(x), n_vars)
    actual = _evaluate(problem, x)
    assert actual == pytest.approx(expected, rel=1e-12)


def test_clamped_near_contact_pair_matches_reference():
    # Two atoms 0.05 apart: r^6 = 1.56e-8 < 1e-6 -> the epsilon clamp fires.
    x = [0.0, 0.0, 0.0, 0.05, 0.0, 0.0]
    problem = LennardJonesMinimumEnergyCluster(number_of_variables=6)
    expected = _reference_energy(x, 6)
    actual = _evaluate(problem, x)
    assert actual == pytest.approx(expected, rel=1e-12)
    # Clamp value: 1/eps^2 - 2/eps.
    assert actual == pytest.approx(1e12 - 2e6, rel=1e-12)


def test_leftover_variables_are_ignored():
    # n = 10 -> 3 atoms; the 10th variable must not influence the energy.
    problem = LennardJonesMinimumEnergyCluster(number_of_variables=10)
    base = [1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 1.5, 0.0, 0.0]
    tweaked = base[:9] + [99.0]
    assert _evaluate(problem, base) == pytest.approx(
        _evaluate(problem, tweaked), rel=0, abs=0
    )
