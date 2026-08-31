from typing import List, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.diversity import normalized_swarm_diversity
from algorithm.role_based.roles import RoleMixin
from algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar('S')
R = TypeVar('R')

class FRAPSO(SingleObjectivePSO):
    """
    Fractal Restart Adaptive PSO (FRAPSO)
    Fractal, Focus
    Concept: Uses fractal decomposition for hierarchical search
    Key Features:
      - Fractal Space Partitioning: Recursively divides search space
      - Multi-Resolution Search: Particles alternate between global and refined local search
      - Adaptive Focus: Automatically concentrates particles near promising regions
    """

    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion, fractal_depth=3, convergence_threshold=1e-3):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.fractal_depth = fractal_depth
        self.convergence_threshold = convergence_threshold
        self.current_depth = 0
        self.total_restarts = 0
        # Instead of altering the problem bounds, store a copy that defines the current search region.
        self.current_lower_bound = np.array(self.problem.lower_bound, dtype=float)
        self.current_upper_bound = np.array(self.problem.upper_bound, dtype=float)

    def converged(self) -> bool:
        # Normalized by the CURRENT zoom region, so the trigger keeps the same
        # "collapsed relative to its search box" meaning at every depth.
        diversity = normalized_swarm_diversity(
            [p.variables for p in self.solutions],
            self.current_lower_bound,
            self.current_upper_bound,
        )
        return bool(diversity < self.convergence_threshold)

    def reinitialize_swarm(self):
        for particle in self.solutions:
            particle.variables = np.random.uniform(
                self.current_lower_bound,
                self.current_upper_bound
            ).tolist()
            particle.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables()).tolist()
            particle.attributes['best_position'] = particle.variables.copy()
            particle.attributes['best_objective'] = float('inf')

    def calculate_subregion(self, best_particle: FloatSolution):
        # Use the current search region as the starting point.
        lower = self.current_lower_bound
        upper = self.current_upper_bound
        range_size = upper - lower
        new_range = range_size / (2 ** (self.current_depth + 1))
        best_vars = np.array(best_particle.variables, dtype=float)
        new_lower = best_vars - new_range / 2
        new_upper = best_vars + new_range / 2
        # Ensure the new bounds do not exceed the original search space (optional safeguard).
        new_lower = np.maximum(new_lower, np.array(self.problem.lower_bound, dtype=float))
        new_upper = np.minimum(new_upper, np.array(self.problem.upper_bound, dtype=float))
        return new_lower, new_upper

    def fractal_decomposition(self):
        if self.converged() and self.current_depth < self.fractal_depth:
            self.current_depth += 1
            self.total_restarts += 1
            new_lower, new_upper = self.calculate_subregion(self.best_global)
            # Update current search region; do not modify problem.lower_bound/upper_bound.
            self.current_lower_bound = new_lower
            self.current_upper_bound = new_upper
            self.reinitialize_swarm()

    def step(self):
        self.fractal_decomposition()
        super().step()

    def observable_data(self) -> dict:
        data = super().observable_data()
        data['TOTAL_RESTARTS'] = self.total_restarts
        data['CURRENT_DEPTH'] = self.current_depth
        return data

    def get_name(self) -> str:
        return "FRAPSO"

class PartialResetPSO(SingleObjectivePSO, RoleMixin):
    """
    Detects swarm convergence based on diversity. When converged,
    particles marked with the 'is_restarter' role are randomly
    reinitialized within the original problem bounds. The role assignment
    is fixed at the beginning.
    """
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 termination_criterion: TerminationCriterion,
                 w: float,
                 c1: float,
                 c2: float,
                 convergence_threshold: float = 1e-3,
                 restarter_fraction: float = 0.1,
                 constraint_handling_mode: str = "clip"
                 ):

        super().__init__(problem=problem, swarm_size=swarm_size, c1=c1, c2=c2, w=w,
                         termination_criterion=termination_criterion,
                         constraint_handling_mode=constraint_handling_mode)

        self.convergence_threshold = convergence_threshold
        self.restarter_fraction = max(0.0, min(1.0, restarter_fraction))
        self.total_restarts = 0
        self._num_vars = self.problem.number_of_variables
        self._lower_bound = np.array(self.problem.lower_bound)
        self._upper_bound = np.array(self.problem.upper_bound)

    def check_convergence(self) -> bool:
        diversity = normalized_swarm_diversity(
            [p.variables for p in self.solutions],
            self.problem.lower_bound,
            self.problem.upper_bound,
        )
        return bool(diversity < self.convergence_threshold)

    def selective_reinitialization(self):
        particles_to_reset = [p for p in self.solutions if p.attributes.get('is_restarter', False)]

        for particle in particles_to_reset:
            particle.variables = np.random.uniform(
                self.problem.lower_bound,
                self.problem.upper_bound
            ).tolist()
            particle.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables()).tolist()
            particle.attributes['best_position'] = particle.variables.copy()
            particle.attributes['best_objective'] = float('inf')

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        self.mark_particles(solutions, self.restarter_fraction, 'is_restarter')
        for p in solutions:
             if not hasattr(p, 'attributes'): p.attributes = {}
             if 'is_restarter' not in p.attributes: p.attributes['is_restarter'] = False
        return solutions

    def step(self):
        if self.check_convergence():
            self.total_restarts += 1
            self.selective_reinitialization()
        super().step()

    def observable_data(self) -> dict:
        data = super().observable_data()
        data['TOTAL_RESTARTS'] = self.total_restarts
        return data

    def get_name(self) -> str:
        return "PartialResetPSO"

class CollectiveResetPSO(SingleObjectivePSO):

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 termination_criterion: TerminationCriterion,
                 w: float,
                 c1: float,
                 c2: float,
                 convergence_threshold: float = 1e-3,
                 constraint_handling_mode: str = "clip"
                 ):

        super().__init__(problem=problem, swarm_size=swarm_size, c1=c1, c2=c2, w=w,
                         termination_criterion=termination_criterion,
                         constraint_handling_mode=constraint_handling_mode)

        self.convergence_threshold = convergence_threshold
        self.total_restarts = 0


    def converged(self) -> bool:
        diversity = normalized_swarm_diversity(
            [p.variables for p in self.solutions],
            self.problem.lower_bound,
            self.problem.upper_bound,
        )
        return bool(diversity < self.convergence_threshold)

    def reinitialize_swarm(self):
        for particle in self.solutions:
            particle.variables = np.random.uniform(
                self.problem.lower_bound,
                self.problem.upper_bound
            ).tolist()
            particle.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables()).tolist()
            particle.attributes['best_position'] = particle.variables.copy()
            particle.attributes['best_objective'] = float('inf')

    def step(self):
        if self.converged():
            self.total_restarts += 1
            self.reinitialize_swarm()
        super().step()

    def observable_data(self) -> dict:
        data = super().observable_data()
        data['TOTAL_RESTARTS'] = self.total_restarts
        return data

    def get_name(self) -> str:
        return "CollectiveResetPSO"

