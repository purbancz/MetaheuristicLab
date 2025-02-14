import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.single_objective_PSO import SingleObjectivePSO


class FAPSO(SingleObjectivePSO):
    """
    Fractal Adaptive PSO (FAPSO)
    Concept: Uses fractal decomposition for hierarchical search
    Key Features:
      - Fractal Space Partitioning: Recursively divides search space
      - Multi-Resolution Search: Particles alternate between global and refined local search
      - Adaptive Focus: Automatically concentrates particles near promising regions
    """
    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion, fractal_depth=3):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.fractal_depth = fractal_depth
        self.current_depth = 0
        self.convergence_threshold = 1e-3

    def converged(self) -> bool:
        positions = np.array([p.variables for p in self.solutions])
        centroid = np.mean(positions, axis=0)
        diversity = np.mean(np.linalg.norm(positions - centroid, axis=1))
        return diversity < self.convergence_threshold

    def reinitialize_swarm(self):
        for particle in self.solutions:
            particle.variables = np.random.uniform(
                self.problem.lower_bound,
                self.problem.upper_bound
            ).tolist()
            particle.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables()).tolist()
            particle.attributes['best_position'] = particle.variables.copy()
            particle.attributes['best_objective'] = particle.objectives[0]

    def calculate_subregion(self, best_particle: FloatSolution):
        lower = np.array(self.problem.lower_bound, dtype=float)
        upper = np.array(self.problem.upper_bound, dtype=float)
        range_size = upper - lower
        new_range = range_size / (2 ** (self.current_depth + 1))
        best_vars = np.array(best_particle.variables, dtype=float)
        new_lower = best_vars - new_range / 2
        new_upper = best_vars + new_range / 2
        return new_lower.tolist(), new_upper.tolist()

    def fractal_decomposition(self):
        if self.converged() and self.current_depth < self.fractal_depth:
            self.current_depth += 1
            new_bounds = self.calculate_subregion(self.best_global)
            self.problem.lower_bound, self.problem.upper_bound = new_bounds
            self.reinitialize_swarm()

    def step(self):
        self.fractal_decomposition()
        super().step()