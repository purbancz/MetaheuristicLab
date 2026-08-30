import math

import pytest
from jmetal.core.solution import FloatSolution


def _sol(variables, lower, upper):
    s = FloatSolution(lower, upper, 1, 0)
    s.objectives = [0.0]
    s.variables = list(variables)
    return s


# --- CrownedCross (PASS: verifies the 2D fix) ---

def test_crowned_cross_both_coordinates_influence_result():
    from problem.fixed_varaibles.cross import CrownedCross
    prob = CrownedCross()
    bounds = (prob.lower_bound, prob.upper_bound)

    base = prob.evaluate(_sol([1.0, 1.0], *bounds)).objectives[0]
    shifted_x0 = prob.evaluate(_sol([3.0, 1.0], *bounds)).objectives[0]
    shifted_x1 = prob.evaluate(_sol([1.0, 3.0], *bounds)).objectives[0]

    assert shifted_x0 != base
    assert shifted_x1 != base


# --- LennardJones defects (XFAIL) ---

def test_lennard_jones_two_atom_energy_varies_with_separation():
    from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster
    prob = LennardJonesMinimumEnergyCluster(number_of_variables=6)
    lb, ub = prob.lower_bound, prob.upper_bound

    close = prob.evaluate(_sol([1.0, 0.0, 0.0, 1.5, 0.0, 0.0], lb, ub)).objectives[0]
    far   = prob.evaluate(_sol([1.0, 0.0, 0.0, 5.0, 0.0, 0.0], lb, ub)).objectives[0]

    assert close != far


def test_lennard_jones_pairwise_distance_uses_correct_coordinates():
    from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster
    prob = LennardJonesMinimumEnergyCluster(number_of_variables=9)
    lb, ub = prob.lower_bound, prob.upper_bound

    # atom0=(1,0,0), atom1=(2,0,0), atom2=(100,5,3)
    # Correct pair(0,1): squared_dist=1, d=1, pairwise=1/1-2/1=-1 → total≈11.71
    # Buggy pair(0,1): reads atom2.y and atom2.z → squared_dist=35, d=42875 → pairwise≈-5e-5 → total≈12.71
    s = prob.evaluate(_sol([1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 100.0, 5.0, 3.0], lb, ub))
    assert s.objectives[0] == pytest.approx(11.71, abs=0.1)


# --- GeneralizedSchafferN3 (XFAIL) ---

def test_generalized_schaffer_n3_optimum_value_is_zero():
    from problem.n_variables.schaffer import GeneralizedSchafferN3
    prob = GeneralizedSchafferN3(number_of_variables=4)
    lb, ub = prob.lower_bound, prob.upper_bound

    s = prob.evaluate(_sol([0.0, 0.0, 0.0, 0.0], lb, ub))
    assert s.objectives[0] == pytest.approx(0.0)


# --- SineEnvelope (XFAIL) ---

def test_sine_envelope_optimum_value_is_zero():
    from problem.n_variables.sine_envelope import SineEnvelope
    prob = SineEnvelope(number_of_variables=3)
    lb, ub = prob.lower_bound, prob.upper_bound

    s = prob.evaluate(_sol([0.0, 0.0, 0.0], lb, ub))
    assert s.objectives[0] == pytest.approx(0.0)


# --- SchwefelN36 (XFAIL) ---

def test_schwefel_n36_matches_docstring_formula():
    from problem.n_variables.schwefel import SchwefelN36
    prob = SchwefelN36(number_of_variables=2)
    lb, ub = prob.lower_bound, prob.upper_bound

    test_point = [1.0, 1.0]
    s = prob.evaluate(_sol(test_point, lb, ub))

    # Docstring formula: sum((418.9829 - xi * sin(sqrt(|xi|)))^2)
    expected = sum((418.9829 - xi * math.sin(math.sqrt(abs(xi)))) ** 2 for xi in test_point)
    assert s.objectives[0] == pytest.approx(expected)
