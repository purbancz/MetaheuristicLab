from copy import deepcopy
import math
import random
from typing import List, Optional, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar("S", bound=FloatSolution)


class BoundaryReinitializedPSO(SingleObjectivePSO):
    """PSO with boundary reinitialization for particles whose pbest collapses to gbest.

    Particles whose personal best is effectively the same as the global best are
    treated as diversity-redundant and relocated to a boundary or peripheral
    region of the search space. The pbest-gbest similarity checks use normalized
    distances by default, avoiding raw Euclidean scaling issues in
    high-dimensional continuous domains.
    """

    _DISTANCE_METRICS = {"normalized_rms", "normalized_linf", "fraction_close"}
    _BOUNDARY_STRATEGIES = {"random_face", "near_boundary", "mixed_boundary"}
    _VELOCITY_RESET_STRATEGIES = {"zero", "random", "away_from_gbest"}

    def __init__(
        self,
        problem: FloatProblem,
        swarm_size: int,
        termination_criterion: TerminationCriterion,
        w: float,
        c1: float,
        c2: float,
        pbest_gbest_epsilon: float = 1e-3,
        distance_metric: str = "normalized_rms",
        fraction_close_threshold: float = 0.95,
        per_dimension_epsilon: float = 1e-4,
        boundary_strategy: str = "mixed_boundary",
        boundary_margin: float = 0.05,
        boundary_dimension_probability: Optional[float] = None,
        velocity_reset_strategy: str = "random",
        velocity_scale: float = 0.1,
        reset_personal_best_on_reinit: bool = True,
        protect_global_best_particle: bool = True,
        max_reinitialized_particles_per_iteration: Optional[int] = None,
        reinitialization_probability: float = 1.0,
        min_iterations_before_reinit: int = 1,
        require_no_improvement: bool = False,
        no_improvement_iterations: int = 10,
        require_low_velocity: bool = False,
        normalized_velocity_epsilon: float = 1e-4,
        constraint_handling_mode: str = "clip",
    ):
        super().__init__(
            problem=problem,
            swarm_size=swarm_size,
            c1=c1,
            c2=c2,
            w=w,
            termination_criterion=termination_criterion,
            constraint_handling_mode=constraint_handling_mode,
        )

        if distance_metric not in self._DISTANCE_METRICS:
            raise ValueError(f"Unknown distance_metric: {distance_metric}")
        if boundary_strategy not in self._BOUNDARY_STRATEGIES:
            raise ValueError(f"Unknown boundary_strategy: {boundary_strategy}")
        if velocity_reset_strategy not in self._VELOCITY_RESET_STRATEGIES:
            raise ValueError(f"Unknown velocity_reset_strategy: {velocity_reset_strategy}")

        self.pbest_gbest_epsilon = pbest_gbest_epsilon
        self.distance_metric = distance_metric
        self.fraction_close_threshold = fraction_close_threshold
        self.per_dimension_epsilon = per_dimension_epsilon
        self.boundary_strategy = boundary_strategy
        self.boundary_margin = max(0.0, min(1.0, boundary_margin))
        self.boundary_dimension_probability = boundary_dimension_probability
        self.velocity_reset_strategy = velocity_reset_strategy
        self.velocity_scale = max(0.0, velocity_scale)
        self.reset_personal_best_on_reinit = reset_personal_best_on_reinit
        self.protect_global_best_particle = protect_global_best_particle
        self.max_reinitialized_particles_per_iteration = max_reinitialized_particles_per_iteration
        self.reinitialization_probability = max(0.0, min(1.0, reinitialization_probability))
        self.min_iterations_before_reinit = max(0, min_iterations_before_reinit)
        self.require_no_improvement = require_no_improvement
        self.no_improvement_iterations = max(0, no_improvement_iterations)
        self.require_low_velocity = require_low_velocity
        self.normalized_velocity_epsilon = normalized_velocity_epsilon

        self.iterations = 0
        self.total_reinitializations = 0
        self.reinitializations_this_iteration = 0

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        for particle in solutions:
            particle.attributes["no_improvement_count"] = 0
        return solutions

    def set_solutions(self, solutions: List[S]):
        super().set_solutions(solutions)
        for particle in self.solutions:
            particle.attributes.setdefault("no_improvement_count", 0)

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if particle.objectives[0] < particle.attributes["best_objective"]:
                particle.attributes["best_position"] = particle.variables.copy()
                particle.attributes["best_objective"] = particle.objectives[0]
                particle.attributes["no_improvement_count"] = 0
            else:
                particle.attributes["no_improvement_count"] = (
                    particle.attributes.get("no_improvement_count", 0) + 1
                )

    def step(self):
        super().step()
        self.iterations += 1
        self.reinitialize_redundant_particles(self.solutions)
        if self.reinitializations_this_iteration:
            self.update_global_best(self.solutions)

    def reinitialize_redundant_particles(self, swarm: List[FloatSolution]) -> None:
        self.reinitializations_this_iteration = 0
        if self.iterations < self.min_iterations_before_reinit:
            return

        limit = self.max_reinitialized_particles_per_iteration
        if limit is None:
            limit = max(1, len(swarm) // 10)
        limit = max(0, limit)
        if limit == 0:
            return

        protected_ids = self._protected_particle_ids(swarm)

        for particle in swarm:
            if self.reinitializations_this_iteration >= limit:
                break
            if id(particle) in protected_ids:
                continue
            if random.random() >= self.reinitialization_probability:
                continue
            if not self.is_diversity_redundant(particle):
                continue

            self._reinitialize_particle(particle)
            self.reinitializations_this_iteration += 1
            self.total_reinitializations += 1

    def is_diversity_redundant(self, particle: FloatSolution) -> bool:
        if self.best_global is None or "best_position" not in particle.attributes:
            return False

        if self.require_no_improvement:
            if particle.attributes.get("no_improvement_count", 0) < self.no_improvement_iterations:
                return False

        if self.require_low_velocity:
            if self._normalized_velocity_rms(particle) > self.normalized_velocity_epsilon:
                return False

        pbest = np.array(particle.attributes["best_position"], dtype=float)
        gbest = np.array(self.best_global.variables, dtype=float)

        if self.distance_metric == "fraction_close":
            return self._fraction_close(pbest, gbest) >= self.fraction_close_threshold

        return self._normalized_distance(pbest, gbest, self.distance_metric) < self.pbest_gbest_epsilon

    def _normalized_difference(self, first: np.ndarray, second: np.ndarray) -> np.ndarray:
        lower_bound, upper_bound = self._bounds()
        ranges = upper_bound - lower_bound
        difference = first - second
        normalized = np.zeros_like(difference, dtype=float)

        non_zero_range = ranges != 0.0
        normalized[non_zero_range] = difference[non_zero_range] / ranges[non_zero_range]
        return normalized

    def _normalized_distance(self, first: np.ndarray, second: np.ndarray, metric: str) -> float:
        normalized = np.abs(self._normalized_difference(first, second))
        if normalized.size == 0:
            return 0.0
        if metric == "normalized_rms":
            # RMS on range-normalized coordinates is less dimension-biased than raw Euclidean distance.
            return float(math.sqrt(np.mean(normalized ** 2)))
        if metric == "normalized_linf":
            return float(np.max(normalized))
        raise ValueError(f"Unknown normalized distance metric: {metric}")

    def _fraction_close(self, first: np.ndarray, second: np.ndarray) -> float:
        normalized = np.abs(self._normalized_difference(first, second))
        if normalized.size == 0:
            return 1.0
        return float(np.mean(normalized < self.per_dimension_epsilon))

    def _normalized_velocity_rms(self, particle: FloatSolution) -> float:
        velocity = np.array(particle.attributes.get("velocity", []), dtype=float)
        if velocity.size == 0:
            return 0.0

        lower_bound, upper_bound = self._bounds()
        ranges = upper_bound - lower_bound
        normalized = np.zeros_like(velocity, dtype=float)
        non_zero_range = ranges != 0.0
        normalized[non_zero_range] = velocity[non_zero_range] / ranges[non_zero_range]
        return float(math.sqrt(np.mean(normalized ** 2)))

    def _protected_particle_ids(self, swarm: List[FloatSolution]) -> set[int]:
        if not self.protect_global_best_particle or self.best_global is None:
            return set()

        matches = [particle for particle in swarm if self._matches_global_best_particle(particle)]
        if not matches:
            return set()
        return {id(matches[0])}

    def _matches_global_best_particle(self, particle: FloatSolution) -> bool:
        if self.best_global is None:
            return False
        if not np.isclose(particle.objectives[0], self.best_global.objectives[0]):
            return False
        return bool(np.allclose(particle.variables, self.best_global.variables))

    def _reinitialize_particle(self, particle: FloatSolution) -> None:
        new_position = self._sample_boundary_position()
        particle.variables = new_position.tolist()
        particle.attributes["velocity"] = self._reset_velocity(new_position).tolist()

        self.problem.evaluate(particle)
        self.evaluations += 1
        particle.attributes["no_improvement_count"] = 0

        if self.reset_personal_best_on_reinit:
            # Resetting pbest makes the relocated particle a renewed explorer instead of
            # immediately pulling it back to the collapsed global-best region.
            particle.attributes["best_position"] = particle.variables.copy()
            particle.attributes["best_objective"] = particle.objectives[0]
            particle.attributes["no_improvement_count"] = 0

    def _sample_boundary_position(self) -> np.ndarray:
        if self.boundary_strategy == "random_face":
            return self._sample_random_face()
        if self.boundary_strategy == "near_boundary":
            return self._sample_near_boundary()
        if self.boundary_strategy == "mixed_boundary":
            return self._sample_mixed_boundary()
        raise ValueError(f"Unknown boundary_strategy: {self.boundary_strategy}")

    def _sample_random_face(self) -> np.ndarray:
        lower_bound, upper_bound = self._bounds()
        position = np.random.uniform(lower_bound, upper_bound)
        if position.size == 0:
            return position

        dimension = random.randrange(position.size)
        position[dimension] = lower_bound[dimension] if random.random() < 0.5 else upper_bound[dimension]
        return position

    def _sample_near_boundary(self) -> np.ndarray:
        lower_bound, upper_bound = self._bounds()
        ranges = upper_bound - lower_bound
        position = np.random.uniform(lower_bound, upper_bound)
        if position.size == 0:
            return position

        # A boundary layer sample should touch a peripheral slab, not force every
        # dimension toward a bound. In high dimensions that would bias strongly
        # toward corners.
        dimension = random.randrange(position.size)
        if ranges[dimension] == 0.0:
            position[dimension] = lower_bound[dimension]
        else:
            margin = self.boundary_margin * ranges[dimension]
            if random.random() < 0.5:
                position[dimension] = random.uniform(lower_bound[dimension], lower_bound[dimension] + margin)
            else:
                position[dimension] = random.uniform(upper_bound[dimension] - margin, upper_bound[dimension])

        return np.clip(position, lower_bound, upper_bound)

    def _sample_mixed_boundary(self) -> np.ndarray:
        lower_bound, upper_bound = self._bounds()
        ranges = upper_bound - lower_bound
        position = np.random.uniform(lower_bound, upper_bound)
        if position.size == 0:
            return position

        probability = self.boundary_dimension_probability
        if probability is None:
            probability = min(1.0, 1.0 / math.sqrt(position.size))
        probability = max(0.0, min(1.0, probability))

        selected_any = False
        for index, value_range in enumerate(ranges):
            if random.random() >= probability:
                continue
            selected_any = True
            if value_range == 0.0:
                position[index] = lower_bound[index]
                continue

            margin = self.boundary_margin * value_range
            if random.random() < 0.5:
                position[index] = random.uniform(lower_bound[index], lower_bound[index] + margin)
            else:
                position[index] = random.uniform(upper_bound[index] - margin, upper_bound[index])

        if not selected_any:
            dimension = random.randrange(position.size)
            if ranges[dimension] == 0.0:
                position[dimension] = lower_bound[dimension]
            else:
                margin = self.boundary_margin * ranges[dimension]
                if random.random() < 0.5:
                    position[dimension] = random.uniform(lower_bound[dimension], lower_bound[dimension] + margin)
                else:
                    position[dimension] = random.uniform(upper_bound[dimension] - margin, upper_bound[dimension])

        return np.clip(position, lower_bound, upper_bound)

    def _reset_velocity(self, new_position: np.ndarray) -> np.ndarray:
        lower_bound, upper_bound = self._bounds()
        ranges = upper_bound - lower_bound

        if self.velocity_reset_strategy == "zero":
            return np.zeros_like(new_position, dtype=float)
        if self.velocity_reset_strategy == "random":
            return np.random.uniform(-self.velocity_scale * ranges, self.velocity_scale * ranges)
        if self.velocity_reset_strategy == "away_from_gbest":
            if self.best_global is None:
                return np.random.uniform(-self.velocity_scale * ranges, self.velocity_scale * ranges)
            direction = new_position - np.array(self.best_global.variables, dtype=float)
            if np.allclose(direction, 0.0):
                return np.random.uniform(-self.velocity_scale * ranges, self.velocity_scale * ranges)
            return self.velocity_scale * direction

        raise ValueError(f"Unknown velocity_reset_strategy: {self.velocity_reset_strategy}")

    def _bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            np.array(self.problem.lower_bound, dtype=float),
            np.array(self.problem.upper_bound, dtype=float),
        )

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        super().initialize_global_best(swarm)
        self.best_global = deepcopy(self.best_global)

    def observable_data(self) -> dict:
        data = super().observable_data()
        data["TOTAL_REINITIALIZATIONS"] = self.total_reinitializations
        data["REINITIALIZATIONS_THIS_ITERATION"] = self.reinitializations_this_iteration
        return data

    def get_name(self) -> str:
        return "BoundaryReinitializedPSO"
