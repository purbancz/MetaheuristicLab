from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.AdaptivePSO import CoAdaptativePSO
from algorithm.WAPSO import WorstAwarePSO
from algorithm.particles_with_roles import AdaptiveRolePSO, RoleMixin
from algorithm.reinitialized_PSO import CollectiveResetPSO, PartialResetPSO
from algorithm.single_objective_PSO import SingleObjectivePSO


def termination(max_evaluations: int = 20) -> StoppingByEvaluations:
    return StoppingByEvaluations(max_evaluations=max_evaluations)


def test_single_objective_pso_initializes_particle_state() -> None:
    problem = Sphere(3)
    algorithm = SingleObjectivePSO(problem, swarm_size=4, c1=1.0, c2=1.0, w=0.5, termination_criterion=termination())

    swarm = algorithm.create_initial_solutions()

    assert len(swarm) == 4
    assert algorithm.best_global is not None
    for particle in swarm:
        assert len(particle.attributes["velocity"]) == problem.number_of_variables()
        assert particle.attributes["best_position"] == particle.variables
        assert particle.attributes["best_objective"] == particle.objectives[0]


def test_single_objective_pso_clip_constraint_keeps_position_in_bounds() -> None:
    problem = Sphere(2)
    algorithm = SingleObjectivePSO(problem, swarm_size=1, c1=1.0, c2=1.0, w=0.5, termination_criterion=termination())
    position, velocity = algorithm.handle_constraints(
        np.array([-10.0, 10.0]),
        np.array([1.0, -1.0]),
        np.array(problem.lower_bound),
        np.array(problem.upper_bound),
    )

    np.testing.assert_allclose(position, [problem.lower_bound[0], problem.upper_bound[1]])
    np.testing.assert_allclose(velocity, [1.0, -1.0])


def test_worst_aware_pso_initializes_and_updates_worst_state() -> None:
    problem = Sphere(2)
    algorithm = WorstAwarePSO(problem, swarm_size=3, c1=1.0, c2=1.0, w=0.5, termination_criterion=termination())
    swarm = algorithm.create_initial_solutions()

    assert algorithm.global_worst is not None
    for particle in swarm:
        assert particle.attributes["worst_position"] == particle.variables
        assert particle.attributes["worst_objective"] == particle.objectives[0]

    particle = swarm[0]
    particle.variables = [problem.upper_bound[0], problem.upper_bound[1]]
    particle.objectives[0] = particle.attributes["worst_objective"] + 1.0
    algorithm.update_particle_worst(swarm)

    assert particle.attributes["worst_position"] == particle.variables
    assert particle.attributes["worst_objective"] == particle.objectives[0]


def test_role_mixin_marks_expected_number_of_particles() -> None:
    swarm = [SimpleNamespace(attributes={}) for _ in range(10)]

    with patch("algorithm.particles_with_roles.random.sample", return_value=[1, 3, 5]):
        RoleMixin.mark_particles(swarm, 0.3, "is_rebel")

    assert sum(p.attributes["is_rebel"] for p in swarm) == 3
    assert [p.attributes["is_rebel"] for p in swarm][1] is True


def test_adaptive_role_pso_uses_role_mixin_marking() -> None:
    problem = Sphere(2)
    algorithm = AdaptiveRolePSO(
        problem=problem,
        swarm_size=4,
        c1=1.0,
        c2=1.0,
        w=0.5,
        termination_criterion=termination(),
        base_inertia=0.5,
        min_inertia=0.1,
        max_inertia=0.9,
        role_fractions={"is_rebel": 0.5},
        max_role_fractions={"is_rebel": 0.75},
        diversity_threshold=0.1,
        improvement_threshold=0.01,
        window_size=3,
    )

    with patch("algorithm.particles_with_roles.random.sample", return_value=[0, 2]):
        swarm = algorithm.create_initial_solutions()

    assert sum(p.attributes["is_rebel"] for p in swarm) == 2


def test_adaptive_and_reset_variants_create_expected_state() -> None:
    problem = Sphere(2)
    capso = CoAdaptativePSO(problem, 3, 1.0, 1.0, 2.0, 2.0, 0.5, termination())
    capso.create_initial_solutions()
    capso.update_coefficient()
    assert capso.min_c1 <= capso.c1 <= capso.max_c1
    assert capso.min_c2 <= capso.c2 <= capso.max_c2

    partial_reset = PartialResetPSO(problem, 4, termination(), 0.5, 1.0, 1.0, restarter_fraction=0.5)
    with patch("algorithm.particles_with_roles.random.sample", return_value=[0, 1]):
        swarm = partial_reset.create_initial_solutions()
    assert sum(p.attributes["is_restarter"] for p in swarm) == 2

    collective_reset = CollectiveResetPSO(problem, 4, termination(), 0.5, 1.0, 1.0)
    collective_reset.create_initial_solutions()
    assert isinstance(collective_reset.converged(), bool)
