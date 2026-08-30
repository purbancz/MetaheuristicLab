from copy import deepcopy
import math
import random

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.reinitialization import BoundaryReinitializedPSO


class BoxProblem(FloatProblem):
    def __init__(self, lower_bound, upper_bound):
        super().__init__()
        self.lower_bound = list(lower_bound)
        self.upper_bound = list(upper_bound)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = sum(value * value for value in solution.variables)
        return solution

    def name(self) -> str:
        return "BoxProblem"


def termination(max_evaluations: int = 30) -> StoppingByEvaluations:
    return StoppingByEvaluations(max_evaluations=max_evaluations)


def make_algorithm(problem, **kwargs) -> BoundaryReinitializedPSO:
    defaults = {
        "swarm_size": 5,
        "termination_criterion": termination(),
        "w": 0.5,
        "c1": 1.0,
        "c2": 1.0,
    }
    defaults.update(kwargs)
    return BoundaryReinitializedPSO(problem=problem, **defaults)


def test_boundary_reinitialized_pso_imports_and_runs() -> None:
    algorithm = make_algorithm(Sphere(3), min_iterations_before_reinit=100)

    algorithm.run()

    assert algorithm.result() is not None
    assert len(algorithm.result().variables) == 3


def test_normalized_rms_distance_is_dimension_stable() -> None:
    problem = BoxProblem([0.0] * 100, [1.0] * 100)
    algorithm = make_algorithm(problem)

    distance = algorithm._normalized_distance(
        np.array([0.001] * 100),
        np.array([0.0] * 100),
        "normalized_rms",
    )

    assert math.isclose(distance, 0.001)


def test_zero_width_dimension_does_not_divide_by_zero() -> None:
    problem = BoxProblem([2.0, -1.0], [2.0, 1.0])
    algorithm = make_algorithm(problem)

    distance = algorithm._normalized_distance(
        np.array([2.0, 0.001]),
        np.array([2.0, 0.0]),
        "normalized_rms",
    )

    assert math.isfinite(distance)


def test_redundant_detection_distinguishes_near_and_far_personal_bests() -> None:
    problem = BoxProblem([0.0, 0.0], [1.0, 1.0])
    algorithm = make_algorithm(problem, pbest_gbest_epsilon=1e-2)
    swarm = algorithm.create_initial_solutions()
    algorithm.best_global = deepcopy(swarm[0])
    algorithm.best_global.variables = [0.5, 0.5]

    near_particle = swarm[1]
    near_particle.attributes["best_position"] = [0.5001, 0.4999]

    far_particle = swarm[2]
    far_particle.attributes["best_position"] = [0.8, 0.2]

    assert algorithm.is_diversity_redundant(near_particle) is True
    assert algorithm.is_diversity_redundant(far_particle) is False


def test_random_face_places_one_coordinate_exactly_on_boundary() -> None:
    random.seed(1)
    np.random.seed(1)
    problem = BoxProblem([-1.0, 2.0, 10.0], [1.0, 4.0, 20.0])
    algorithm = make_algorithm(problem, boundary_strategy="random_face")

    position = algorithm._sample_boundary_position()

    assert np.all(position >= np.array(problem.lower_bound))
    assert np.all(position <= np.array(problem.upper_bound))
    assert any(
        value == lower or value == upper
        for value, lower, upper in zip(position, problem.lower_bound, problem.upper_bound)
    )


def test_near_boundary_samples_boundary_layer_without_forcing_all_dimensions_to_edges() -> None:
    random.seed(2)
    np.random.seed(2)
    problem = BoxProblem([-10.0, 0.0, 5.0], [10.0, 100.0, 9.0])
    algorithm = make_algorithm(problem, boundary_strategy="near_boundary", boundary_margin=0.1)

    position = algorithm._sample_boundary_position()

    assert np.all(position >= np.array(problem.lower_bound))
    assert np.all(position <= np.array(problem.upper_bound))
    boundary_layer_coordinates = 0
    for value, lower, upper in zip(position, problem.lower_bound, problem.upper_bound):
        margin = 0.1 * (upper - lower)
        if lower <= value <= lower + margin or upper - margin <= value <= upper:
            boundary_layer_coordinates += 1

    assert boundary_layer_coordinates >= 1


def test_reinitialization_resets_personal_best_and_preserves_saved_global_best() -> None:
    random.seed(3)
    np.random.seed(3)
    problem = BoxProblem([-1.0, -1.0], [1.0, 1.0])
    algorithm = make_algorithm(
        problem,
        swarm_size=6,
        min_iterations_before_reinit=0,
        max_reinitialized_particles_per_iteration=6,
        boundary_strategy="random_face",
        reset_personal_best_on_reinit=True,
        protect_global_best_particle=True,
    )
    swarm = algorithm.create_initial_solutions()
    algorithm.best_global = deepcopy(swarm[0])
    algorithm.best_global.variables = [0.0, 0.0]
    algorithm.best_global.objectives[0] = 0.0

    for particle in swarm:
        particle.variables = [0.0, 0.0]
        particle.objectives[0] = 0.0
        particle.attributes["best_position"] = [0.0, 0.0]
        particle.attributes["best_objective"] = 0.0

    algorithm.reinitialize_redundant_particles(swarm)

    assert algorithm.reinitializations_this_iteration > 0
    assert algorithm.best_global.objectives[0] == 0.0
    for particle in swarm:
        assert np.all(np.array(particle.variables) >= np.array(problem.lower_bound))
        assert np.all(np.array(particle.variables) <= np.array(problem.upper_bound))

    reset_particles = [particle for particle in swarm if particle.variables != [0.0, 0.0]]
    assert reset_particles
    for particle in reset_particles:
        assert particle.attributes["best_position"] == particle.variables
        assert particle.attributes["best_objective"] == particle.objectives[0]
