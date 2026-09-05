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


# --- LennardJones ---

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

    # atom0 and atom1 at distance 1 (pair energy -1); atom2 far away in a
    # corner of the box contributes only a tiny attraction.
    far = lb[0] + 0.05  # inside the N-scaled bounds
    s = prob.evaluate(_sol([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, far, far, far], lb, ub))
    assert s.objectives[0] == pytest.approx(-1.0, abs=0.02)


def test_lennard_jones_known_optima_match_cambridge_cluster_database():
    from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster

    # N=2: optimal pair at distance 1, CCD energy -1.
    prob = LennardJonesMinimumEnergyCluster(number_of_variables=6)
    s = prob.evaluate(_sol([-0.5, 0.0, 0.0, 0.5, 0.0, 0.0], prob.lower_bound, prob.upper_bound))
    assert s.objectives[0] == pytest.approx(-1.0, rel=1e-12)

    # N=3: unit equilateral triangle, CCD energy -3.
    prob = LennardJonesMinimumEnergyCluster(number_of_variables=9)
    triangle = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.5, math.sqrt(3) / 2, 0.0]
    s = prob.evaluate(_sol(triangle, prob.lower_bound, prob.upper_bound))
    assert s.objectives[0] == pytest.approx(-3.0, rel=1e-12)

    # N=4: unit regular tetrahedron, CCD energy -6.
    prob = LennardJonesMinimumEnergyCluster(number_of_variables=12)
    tetrahedron = [
        0.0, 0.0, 0.0,
        1.0, 0.0, 0.0,
        0.5, math.sqrt(3) / 2, 0.0,
        0.5, math.sqrt(3) / 6, math.sqrt(2.0 / 3.0),
    ]
    s = prob.evaluate(_sol(tetrahedron, prob.lower_bound, prob.upper_bound))
    assert s.objectives[0] == pytest.approx(-6.0, rel=1e-12)

    # Argmin-neighborhood check: perturbing the tetrahedron must worsen it.
    perturbed = list(tetrahedron)
    perturbed[0] += 0.1
    s_perturbed = prob.evaluate(_sol(perturbed, prob.lower_bound, prob.upper_bound))
    assert s_perturbed.objectives[0] > -6.0


def test_lennard_jones_bounds_scale_with_atom_count_and_contain_optima():
    from problem.n_variables.lenard_johnes_minimum_energy_cluster import LennardJonesMinimumEnergyCluster

    for n_vars in (6, 12, 300):
        prob = LennardJonesMinimumEnergyCluster(number_of_variables=n_vars)
        n_atoms = n_vars // 3
        expected_half_width = 2.0 * n_atoms ** (1.0 / 3.0)
        assert prob.upper_bound[0] == pytest.approx(expected_half_width)
        assert prob.lower_bound[0] == pytest.approx(-expected_half_width)
        # The box comfortably contains a unit-density cluster of N atoms
        # (radius ~0.62 * N^(1/3)).
        assert expected_half_width > 0.62 * n_atoms ** (1.0 / 3.0)


# --- GeneralizedSchafferN3 (classical Schaffer F7 pair form) ---

def test_generalized_schaffer_n3_optimum_value_is_zero():
    from problem.n_variables.schaffer import GeneralizedSchafferN3
    prob = GeneralizedSchafferN3(number_of_variables=4)
    lb, ub = prob.lower_bound, prob.upper_bound

    s = prob.evaluate(_sol([0.0, 0.0, 0.0, 0.0], lb, ub))
    assert s.objectives[0] == pytest.approx(0.0)


# --- SineEnvelope ---

def test_sine_envelope_matches_classical_formula():
    from problem.n_variables.sine_envelope import SineEnvelope
    prob = SineEnvelope(number_of_variables=3)
    lb, ub = prob.lower_bound, prob.upper_bound

    s = prob.evaluate(_sol([0.0, 0.0, 0.0], lb, ub))
    assert s.objectives[0] == pytest.approx(-2.0 * (math.sin(0.5) ** 2 + 0.5))

    x = [1.0, -2.0, 3.0]
    expected = 0.0
    for a, b in zip(x, x[1:]):
        r2 = a * a + b * b
        expected -= (math.sin(math.sqrt(r2) - 0.5) ** 2) / ((0.001 * r2 + 1) ** 2) + 0.5
    s = prob.evaluate(_sol(x, lb, ub))
    assert s.objectives[0] == pytest.approx(expected)


def test_sine_envelope_origin_is_not_the_minimum():
    from problem.n_variables.sine_envelope import SineEnvelope
    prob = SineEnvelope(number_of_variables=2)
    lb, ub = prob.lower_bound, prob.upper_bound

    origin = prob.evaluate(_sol([0.0, 0.0], lb, ub)).objectives[0]

    ring_radius = 0.5 + math.pi / 2
    ring = prob.evaluate(_sol([ring_radius, 0.0], lb, ub)).objectives[0]

    assert ring < origin
    assert ring == pytest.approx(-1.4915, abs=1e-3)


# --- SchwefelN36 (squared Schwefel 2.26 variant) ---

def test_schwefel_n36_matches_docstring_formula():
    from problem.n_variables.schwefel import SchwefelN36
    prob = SchwefelN36(number_of_variables=2)
    lb, ub = prob.lower_bound, prob.upper_bound

    test_point = [1.0, 1.0]
    s = prob.evaluate(_sol(test_point, lb, ub))

    # Docstring formula: sum((418.9829 - xi * sin(sqrt(|xi|)))^2)
    expected = sum((418.9829 - xi * math.sin(math.sqrt(abs(xi)))) ** 2 for xi in test_point)
    assert s.objectives[0] == pytest.approx(expected)


def test_schwefel_n36_optimum_is_near_zero_at_420_9687():
    from problem.n_variables.schwefel import SchwefelN36
    prob = SchwefelN36(number_of_variables=3)
    lb, ub = prob.lower_bound, prob.upper_bound

    x_star = 420.9687
    s = prob.evaluate(_sol([x_star] * 3, lb, ub))
    assert s.objectives[0] == pytest.approx(0.0, abs=1e-6)

    # And nearby points are worse: the optimum is a genuine minimum.
    s_off = prob.evaluate(_sol([x_star + 5.0] * 3, lb, ub))
    assert s_off.objectives[0] > s.objectives[0]
