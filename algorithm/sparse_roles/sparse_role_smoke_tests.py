"""Smoke tests for sparse role PSO variants."""

from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.sparse_roles.coordinate_mask_utilities import CoordinateMaskMixin
from algorithm.sparse_roles.sparse_hybrid import SparseHybridPartialDisjointPSO, SparseHybridFullDisjointPSO, \
    SparseHybridAdditivePSO
from algorithm.sparse_roles.sparse_role_based import SparseWandererPSO, SparseDefeatistPSO, \
    SparseContrarianDefeatistPSO, SparseRebelPSO, SparseRejectorPSO, SparseRebelRejectorPSO, \
    SparseContrarianPSO, SparseEschewerPSO, SparseEscapistPSO, SparseEschewerEscapistPSO, \
    SparseAnarchicPSO, SparseAmnesiacPSO, SparseAnarchicAmnesiacPSO, SparseErraticPSO, SparseDrifterPSO


class DummyMaskUser(CoordinateMaskMixin):
    pass


def test_masks() -> None:
    helper = DummyMaskUser()

    mask = helper.coordinate_mask(dim=100)
    assert mask.dtype == bool
    assert len(mask) == 100

    sqrt_mask = helper.coordinate_mask(dim=100, mode="sqrt", scale=1.0)
    assert int(np.sum(sqrt_mask)) == 10

    constant_mask = helper.coordinate_mask(dim=100, mode="constant", count=7)
    assert int(np.sum(constant_mask)) == 7

    one_dim_mask = helper.coordinate_mask(dim=1, mode="sqrt")
    assert int(np.sum(one_dim_mask)) == 1

    full_mask = helper.coordinate_mask(dim=5, mode="constant", count=999)
    assert int(np.sum(full_mask)) == 5

    try:
        helper.coordinate_mask(dim=100, mode="bad-mode")
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid coordinate mode should raise ValueError")


def instantiate_algorithms(problem) -> None:
    termination = StoppingByEvaluations(max_evaluations=20)

    algorithms = [
        SparseWandererPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            noise_strength=0.5,
            wanderer_fraction=0.2,
        ),
        SparseDefeatistPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            defeatist_c=1.0,
            w=0.1,
            defeatist_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseContrarianDefeatistPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            defeatist_c=1.0,
            contrarian_c=1.0,
            w=0.1,
            termination_criterion=termination,
            contrarian_fraction=0.2,
            defeatist_fraction=0.2,
        ),
        SparseRebelPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            rebel_c=1.0,
            w=0.1,
            rebel_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseRejectorPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            rejector_c=1.0,
            w=0.1,
            rejector_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseRebelRejectorPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            rejector_c=1.0,
            rebel_c=1.0,
            w=0.1,
            rebel_fraction=0.2,
            rejector_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseContrarianPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            contrarian_c=1.0,
            w=0.1,
            contrarian_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseEschewerPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            eschewer_c=1.0,
            w=0.1,
            eschewer_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseEscapistPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            escapist_c=1.0,
            w=0.1,
            escapist_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseEschewerEscapistPSO(
            problem=problem,
            swarm_size=10,
            c1=1.5,
            c2=1.5,
            escapist_c=1.0,
            eschewer_c=1.0,
            w=0.1,
            eschewer_fraction=0.2,
            escapist_fraction=0.2,
            termination_criterion=termination,
        ),
        SparseAnarchicPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            random_strength=0.5,
            anarchic_fraction=0.2,
        ),
        SparseAmnesiacPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            random_strength=0.5,
            amnesiac_fraction=0.2,
        ),
        SparseAnarchicAmnesiacPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            random_strength_social=0.5,
            random_strength_cognitive=0.5,
            anarchic_fraction=0.2,
            amnesiac_fraction=0.2,
        ),
        SparseErraticPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            random_strength=0.5,
            erratic_fraction=0.2,
        ),
        SparseDrifterPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            drifter_fraction=0.2,
            perturbation_scale=0.5,
        ),
        SparseHybridPartialDisjointPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            rejector_fraction=0.1,
            defeatist_fraction=0.1,
            escapist_fraction=0.1,
            rebel_fraction=0.1,
            contrarian_fraction=0.1,
            eschewer_fraction=0.1,
        ),
        SparseHybridFullDisjointPSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            rejector_fraction=0.1,
            defeatist_fraction=0.1,
            rebel_fraction=0.1,
        ),
        SparseHybridAdditivePSO(
            problem=problem,
            swarm_size=10,
            termination_criterion=termination,
            w=0.1,
            c1=1.5,
            c2=1.5,
            rejector_prob=0.1,
            defeatist_prob=0.1,
            rebel_prob=0.1,
            contrarian_prob=0.1,
        ),
    ]

    for algorithm in algorithms:
        swarm = algorithm.create_initial_solutions()
        assert len(swarm) == 10

        algorithm.update_velocity(swarm)

        for particle in swarm:
            assert "velocity" in particle.attributes
            assert len(particle.attributes["velocity"]) == problem.number_of_variables()

    print("Sparse role PSO smoke tests passed.")

def test_sparse_algorithms_smoke() -> None:
    problem = Sphere(10)
    instantiate_algorithms(problem)

def test_coordinate_count_modes() -> None:
    helper = DummyMaskUser()

    assert helper.coordinate_count(dim=100, mode="sqrt", scale=1.0) == 10
    assert helper.coordinate_count(dim=500, mode="sqrt", scale=1.0) == 22
    assert helper.coordinate_count(dim=1000, mode="sqrt", scale=1.0) == 31

    assert helper.coordinate_count(dim=100, mode="fraction", fraction=0.1) == 10
    assert helper.coordinate_count(dim=100, mode="constant", count=7) == 7
    assert helper.coordinate_count(dim=5, mode="constant", count=999) == 5
    assert helper.coordinate_count(dim=1, mode="sqrt", scale=1.0) == 1


def make_particle(**roles):
    return SimpleNamespace(
        variables=[1.0, 2.0, 3.0, 4.0],
        objectives=[0.0],
        attributes={
            "velocity": [0.5, 0.5, 0.5, 0.5],
            "best_position": [2.0, 4.0, 6.0, 8.0],
            "worst_position": [-1.0, 0.0, 1.0, 2.0],
            "best_objective": 0.0,
            "worst_objective": 10.0,
            **roles,
        },
    )


def configure_algorithm(algorithm):
    algorithm.best_global = SimpleNamespace(variables=[0.0, 0.0, 0.0, 0.0])
    algorithm.global_worst = SimpleNamespace(variables=[5.0, 5.0, 5.0, 5.0])
    return algorithm


def standard_velocity():
    current = np.array([1.0, 2.0, 3.0, 4.0])
    velocity = np.array([0.5, 0.5, 0.5, 0.5])
    p_best = np.array([2.0, 4.0, 6.0, 8.0])
    g_best = np.array([0.0, 0.0, 0.0, 0.0])
    return 0.1 * velocity + 2.0 * (p_best - current) + 3.0 * (g_best - current)


def assert_sparse_velocity(algorithm, particle, expected):
    with patch("algorithm.sparse_roles.sparse_role_based.random.random", return_value=1.0), \
            patch("algorithm.sparse_roles.sparse_role_based.np.random.uniform",
                  return_value=np.array([10.0, 20.0, 30.0, 40.0])):
        algorithm.update_velocity([particle])

    np.testing.assert_allclose(particle.attributes["velocity"], expected)


def test_sparse_single_role_variants_apply_role_only_on_masked_coordinates() -> None:
    problem = Sphere(4)
    termination = StoppingByEvaluations(max_evaluations=20)
    mask = np.array([True, False, True, False])
    current = np.array([1.0, 2.0, 3.0, 4.0])
    velocity = np.array([0.5, 0.5, 0.5, 0.5])
    p_best = np.array([2.0, 4.0, 6.0, 8.0])
    p_worst = np.array([-1.0, 0.0, 1.0, 2.0])
    g_best = np.array([0.0, 0.0, 0.0, 0.0])
    g_worst = np.array([5.0, 5.0, 5.0, 5.0])
    base = standard_velocity()

    cases = [
        (
            configure_algorithm(SparseWandererPSO(problem, 1, termination, 0.1, 2.0, 3.0, 0.5, 1.0)),
            make_particle(is_wanderer=True),
            base + np.where(mask, 0.5 * np.array([10.0, 20.0, 30.0, 40.0]), 0.0),
        ),
        (
            configure_algorithm(SparseDefeatistPSO(problem, 1, 2.0, 3.0, 7.0, 0.1, 1.0, termination)),
            make_particle(is_defeatist=True),
            0.1 * velocity + np.where(mask, 7.0 * (p_worst - current), 2.0 * (p_best - current))
            + 3.0 * (g_best - current),
        ),
        (
            configure_algorithm(SparseRebelPSO(problem, 1, 2.0, 3.0, 7.0, 0.1, 1.0, termination)),
            make_particle(is_rebel=True),
            0.1 * velocity + 2.0 * (p_best - current)
            + np.where(mask, 7.0 * (current - g_best), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseRejectorPSO(problem, 1, 2.0, 3.0, 7.0, 0.1, 1.0, termination)),
            make_particle(is_rejector=True),
            0.1 * velocity + np.where(mask, 7.0 * (current - p_best), 2.0 * (p_best - current))
            + 3.0 * (g_best - current),
        ),
        (
            configure_algorithm(SparseContrarianPSO(problem, 1, 2.0, 3.0, 7.0, 0.1, 1.0, termination)),
            make_particle(is_contrarian=True),
            0.1 * velocity + 2.0 * (p_best - current)
            + np.where(mask, 7.0 * (g_worst - current), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseEschewerPSO(problem, 1, 2.0, 3.0, 7.0, 0.1, 1.0, termination)),
            make_particle(is_eschewer=True),
            0.1 * velocity + 2.0 * (p_best - current)
            + np.where(mask, 7.0 * (current - g_worst), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseEscapistPSO(problem, 1, 2.0, 3.0, 7.0, 0.1, 1.0, termination)),
            make_particle(is_escapist=True),
            0.1 * velocity + np.where(mask, 7.0 * (current - p_worst), 2.0 * (p_best - current))
            + 3.0 * (g_best - current),
        ),
        (
            configure_algorithm(SparseAnarchicPSO(problem, 1, termination, 0.1, 2.0, 3.0, 0.5, 1.0)),
            make_particle(is_anarchic=True),
            0.1 * velocity + 2.0 * (p_best - current)
            + np.where(mask, 0.5 * np.array([10.0, 20.0, 30.0, 40.0]), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseAmnesiacPSO(problem, 1, termination, 0.1, 2.0, 3.0, 0.5, 1.0)),
            make_particle(is_amnesiac=True),
            0.1 * velocity + np.where(mask, 0.5 * np.array([10.0, 20.0, 30.0, 40.0]), 2.0 * (p_best - current))
            + 3.0 * (g_best - current),
        ),
        (
            configure_algorithm(SparseErraticPSO(problem, 1, termination, 0.1, 2.0, 3.0, 0.5, 1.0)),
            make_particle(is_erratic=True),
            np.where(mask, 0.1 * velocity + 0.5 * np.array([10.0, 20.0, 30.0, 40.0]), base),
        ),
    ]

    for algorithm, particle, expected in cases:
        algorithm.coordinate_mask = lambda dim, **kwargs: mask
        algorithm._single_mask = lambda dim: mask
        assert_sparse_velocity(algorithm, particle, expected)
        np.testing.assert_allclose(np.array(particle.attributes["velocity"])[~mask], expected[~mask])
        np.testing.assert_allclose(np.array(particle.attributes["velocity"])[~mask], base[~mask])


def test_sparse_double_role_variants_apply_each_role_only_on_its_masked_coordinates() -> None:
    problem = Sphere(4)
    termination = StoppingByEvaluations(max_evaluations=20)
    social_mask = np.array([True, False, False, False])
    cognitive_mask = np.array([False, True, False, False])
    current = np.array([1.0, 2.0, 3.0, 4.0])
    velocity = np.array([0.5, 0.5, 0.5, 0.5])
    p_best = np.array([2.0, 4.0, 6.0, 8.0])
    p_worst = np.array([-1.0, 0.0, 1.0, 2.0])
    g_best = np.array([0.0, 0.0, 0.0, 0.0])
    g_worst = np.array([5.0, 5.0, 5.0, 5.0])
    random_vec = np.array([10.0, 20.0, 30.0, 40.0])
    base = standard_velocity()

    cases = [
        (
            configure_algorithm(
                SparseContrarianDefeatistPSO(
                    problem, 1, 2.0, 3.0, 7.0, 8.0, 0.1, termination, 1.0, 1.0
                )
            ),
            make_particle(is_contrarian=True, is_defeatist=True),
            0.1 * velocity
            + np.where(cognitive_mask, 7.0 * (p_worst - current), 2.0 * (p_best - current))
            + np.where(social_mask, 8.0 * (g_worst - current), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseRebelRejectorPSO(problem, 1, 2.0, 3.0, 7.0, 8.0, 0.1, 1.0, 1.0, termination)),
            make_particle(is_rebel=True, is_rejector=True),
            0.1 * velocity
            + np.where(cognitive_mask, 7.0 * (current - p_best), 2.0 * (p_best - current))
            + np.where(social_mask, 8.0 * (current - g_best), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseEschewerEscapistPSO(problem, 1, 2.0, 3.0, 7.0, 8.0, 0.1, 1.0, 1.0, termination)),
            make_particle(is_eschewer=True, is_escapist=True),
            0.1 * velocity
            + np.where(cognitive_mask, 7.0 * (current - p_worst), 2.0 * (p_best - current))
            + np.where(social_mask, 8.0 * (current - g_worst), 3.0 * (g_best - current)),
        ),
        (
            configure_algorithm(SparseAnarchicAmnesiacPSO(problem, 1, termination, 0.1, 2.0, 3.0, 0.5, 0.25, 1.0, 1.0)),
            make_particle(is_anarchic=True, is_amnesiac=True),
            0.1 * velocity
            + np.where(cognitive_mask, 0.25 * random_vec, 2.0 * (p_best - current))
            + np.where(social_mask, 0.5 * random_vec, 3.0 * (g_best - current)),
        ),
    ]

    for algorithm, particle, expected in cases:
        if isinstance(algorithm, SparseContrarianDefeatistPSO):
            algorithm.social_coordinate_mode = "social"
            algorithm.cognitive_coordinate_mode = "cognitive"
            algorithm.coordinate_mask = (
                lambda dim, mode="sqrt", **kwargs: social_mask if mode == "social" else cognitive_mask
            )
        else:
            algorithm._social_mask = lambda dim: social_mask
            algorithm._cognitive_mask = lambda dim: cognitive_mask
        assert_sparse_velocity(algorithm, particle, expected)
        untouched_mask = ~(social_mask | cognitive_mask)
        np.testing.assert_allclose(np.array(particle.attributes["velocity"])[untouched_mask], base[untouched_mask])


def test_sparse_drifter_perturbs_only_masked_coordinates() -> None:
    problem = Sphere(4)
    termination = StoppingByEvaluations(max_evaluations=20)
    mask = np.array([True, False, True, False])
    algorithm = configure_algorithm(
        SparseDrifterPSO(problem, 2, termination, 0.1, 2.0, 3.0, 1.0, 0.5)
    )
    algorithm._single_mask = lambda dim: mask
    drifter = make_particle(is_drifter=True)
    standard = make_particle(is_drifter=False)

    with patch("algorithm.sparse_roles.sparse_role_based.np.random.normal",
               return_value=np.array([0.25, 0.5, 0.75, 1.0])):
        algorithm.perturbation([drifter, standard])

    np.testing.assert_allclose(drifter.variables, [1.25, 2.0, 3.75, 4.0])
    np.testing.assert_allclose(standard.variables, [1.0, 2.0, 3.0, 4.0])


if __name__ == "__main__":
    test_masks()
    test_sparse_algorithms_smoke()
    test_coordinate_count_modes()
    test_sparse_single_role_variants_apply_role_only_on_masked_coordinates()
    test_sparse_double_role_variants_apply_each_role_only_on_its_masked_coordinates()
    test_sparse_drifter_perturbs_only_masked_coordinates()
    print("Sparse role PSO tests passed.")
