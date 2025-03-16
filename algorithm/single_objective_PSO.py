from copy import deepcopy
from typing import List, TypeVar
import random
import numpy as np
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = TypeVar('R')


class SingleObjectivePSO(ParticleSwarmOptimization):
    """Original base class for single-objective PSO variants with common functionality"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size)
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.best_global = None

    def create_initial_solutions(self) -> List[S]:
        self.solutions = [self.problem.create_solution() for _ in range(self.swarm_size)]

        for solution in self.solutions:
            self.problem.evaluate(solution)
            solution.attributes['best_position'] = solution.variables.copy()
            solution.attributes['best_objective'] = solution.objectives[0]
            solution.attributes['velocity'] = np.random.uniform(
                -1, 1, self.problem.number_of_variables()
            ).tolist()

        self.best_global = deepcopy(min(self.solutions, key=lambda sol: sol.objectives[0]))
        return self.solutions

    def run(self):
        super().run()
        return self

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return [self.problem.evaluate(sol) for sol in solution_list]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_velocity(self, swarm: List[S]) -> None:
        for particle in swarm:
            particle.attributes['velocity'] = np.random.uniform(
                -1, 1, self.problem.number_of_variables()
            ).tolist()

    def initialize_particle_best(self, swarm: List[S]) -> None:
        for particle in swarm:
            if 'best_position' not in particle.attributes:
                particle.attributes['best_position'] = particle.variables.copy()
            if 'best_objective' not in particle.attributes:
                particle.attributes['best_objective'] = particle.objectives[0]

    def initialize_global_best(self, swarm: List[S]) -> None:
        if not swarm:
            raise RuntimeError("Swarm is empty during global best initialization!")
        self.best_global = min(swarm, key=lambda x: x.objectives[0])

    def update_velocity(self, swarm: List[S]) -> None:
        gbest = np.array(self.best_global.variables)
        for particle in swarm:
            r1 = random.random()
            r2 = random.random()
            velocity = np.array(particle.attributes['velocity'])
            pbest = np.array(particle.attributes['best_position'])

            new_velocity = (self.w * velocity +
                            self.c1 * r1 * (pbest - np.array(particle.variables)) +
                            self.c2 * r2 * (gbest - np.array(particle.variables)))

            particle.attributes['velocity'] = new_velocity.tolist()

    def update_position(self, swarm: List[S]) -> None:
        for particle in swarm:
            current_position = np.array(particle.variables)
            new_position = current_position + np.array(particle.attributes['velocity'])
            clipped_position = np.clip(new_position,
                                       self.problem.lower_bound,
                                       self.problem.upper_bound)
            particle.variables = clipped_position.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        for particle in swarm:
            if particle.objectives[0] < particle.attributes['best_objective']:
                particle.attributes['best_position'] = particle.variables.copy()
                particle.attributes['best_objective'] = particle.objectives[0]

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        current_best = min(swarm, key=lambda x: x.objectives[0])
        if current_best.objectives[0] < self.best_global.objectives[0]:
            self.best_global = deepcopy(current_best)

    def set_solutions(self, solutions: List[S]):
        self.solutions = deepcopy(solutions)
        for solution in self.solutions:
            if 'velocity' not in solution.attributes:
                solution.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables())
            if ('best_position' not in solution.attributes or solution.objectives[0] <
                    solution.attributes['best_objective']):
                solution.attributes['best_position'] = deepcopy(solution.variables)
                solution.attributes['best_objective'] = solution.objectives[0]

        self.best_global = deepcopy(min(self.solutions, key=lambda sol: sol.objectives[0]))

    def perturbation(self, swarm: List[S]) -> None:
        pass

    def result(self) -> R:
        return self.best_global

    def get_name(self) -> str:
        return "SingleObjectivePSO"


# class RebelPSO(SingleObjectivePSO):
#     """PSO with rebel particles opposing global best"""
#
#     def __init__(self,
#                  problem: FloatProblem,
#                  swarm_size: int,
#                  b1: float,
#                  b2: float,
#                  ac2: float,
#                  w: float,
#                  rebel_fraction: float,
#                  termination_criterion: TerminationCriterion):
#         super().__init__(problem, swarm_size, b1, b2, w, termination_criterion)
#         self.ac2 = ac2
#         self.rebel_fraction = rebel_fraction
#
#     def create_initial_solutions(self) -> List[FloatSolution]:
#         solutions = super().create_initial_solutions()
#         self._mark_rebels(solutions)
#         return solutions
#
#     def _mark_rebels(self, swarm: List[FloatSolution]):
#         num_rebels = max(1, int(len(swarm) * self.rebel_fraction))
#         rebels = random.sample(swarm, num_rebels)
#         for particle in rebels:
#             particle.attributes['is_rebel'] = True
#
#     def update_velocity(self, swarm: List[S]) -> None:
#         gbest = np.array(self.best_global.variables)
#         for particle in swarm:
#             # Base components
#             cognitive = self.b1 * random.random()
#             social = self.b2 * random.random()
#             rebel = self.ac2 * random.random()
#
#             pbest = np.array(particle.attributes['best_position'])
#             current = np.array(particle.variables)
#
#             cognitive_dir = pbest - current
#             cognitive_vec = cognitive * cognitive_dir
#
#             # Rebel logic: Inverse social component
#             if particle.attributes['is_rebel']:
#                 social_dir = current - gbest
#                 social_vec = rebel * social_dir
#             else:
#                 social_dir = gbest - current
#                 social_vec = social * social_dir
#
#             # Hybrid velocity update
#             velocity = (self.w * np.array(particle.attributes['velocity'])
#                         + social_vec
#                         + cognitive_vec)
#
#             particle.attributes['velocity'] = velocity.tolist()


# class EscapistPSO(SingleObjectivePSO):
#     """PSO with escapist particles opposing personal best"""
#
#     def __init__(self,
#                  problem: FloatProblem,
#                  swarm_size: int,
#                  b1: float,
#                  b2: float,
#                  ac1: float,
#                  w: float,
#                  escapist_fraction: float,
#                  termination_criterion: TerminationCriterion):
#         super().__init__(problem, swarm_size, b1, b2, w, termination_criterion)
#         self.ac1 = ac1
#         self.escapist_fraction = escapist_fraction
#
#     def create_initial_solutions(self) -> List[S]:
#         solutions = super().create_initial_solutions()
#         self._mark_escapists(solutions)
#         return solutions
#
#     def _mark_escapists(self, swarm: List[S]):
#         num_escapists = max(1, int(len(swarm) * self.escapist_fraction))
#         escapists = random.sample(swarm, num_escapists)
#         for particle in escapists:
#             particle.attributes['is_escapist'] = True
#
#     def update_velocity(self, swarm: List[S]) -> None:
#         gbest = np.array(self.best_global.variables)
#         for particle in swarm:
#             # Base components
#             cognitive = self.b1 * random.random()
#             social = self.b2 * random.random()
#             escapist = self.ac1 * random.random()
#
#             pbest = np.array(particle.attributes['best_position'])
#             current = np.array(particle.variables)
#
#             social_dir = gbest - current
#             social_vec = social * social_dir
#
#             # Escapist logic: Inverse cognitive component
#             if particle.attributes['is_escapist']:
#                 cognitive_dir = current - pbest
#                 cognitive_vec = escapist * cognitive_dir
#             else:
#                 cognitive_dir = pbest - current
#                 cognitive_vec = cognitive * cognitive_dir
#
#             # Hybrid velocity update
#             velocity = (self.w * np.array(particle.attributes['velocity'])
#                         + social_vec
#                         + cognitive_vec)
#
#             particle.attributes['velocity'] = velocity.tolist()


# class RebelEscapistPSO(SingleObjectivePSO):
#     """PSO with both rebel and escapist particles"""
#
#     def __init__(self,
#                  problem: FloatProblem,
#                  swarm_size: int,
#                  b1: float,
#                  b2: float,
#                  ac1: float,
#                  ac2: float,
#                  w: float,
#                  rebel_fraction: float,
#                  escapist_fraction: float,
#                  termination_criterion: TerminationCriterion):
#         super().__init__(problem, swarm_size, b1, b2, w, termination_criterion)
#         self.ac1 = ac1
#         self.ac2 = ac2
#         self.rebel_fraction = rebel_fraction
#         self.escapist_fraction = escapist_fraction
#
#     def create_initial_solutions(self) -> List[S]:
#         solutions = super().create_initial_solutions()
#         self._mark_special_particles(solutions)
#         return solutions
#
#     def _mark_special_particles(self, swarm: List[S]):
#         # Mark rebels
#         num_rebels = max(1, int(len(swarm) * self.rebel_fraction))
#         rebels = random.sample(swarm, num_rebels)
#         for particle in rebels:
#             particle.attributes['is_rebel'] = True
#
#         # Mark escapists from remaining particles
#         remaining = [p for p in swarm if 'is_rebel' not in p.attributes]
#         num_escapists = max(1, int(len(remaining) * self.escapist_fraction))
#         escapists = random.sample(remaining, num_escapists)
#         for particle in escapists:
#             particle.attributes['is_escapist'] = True
#
#     def update_velocity(self, swarm: List[S]) -> None:
#         gbest = np.array(self.best_global.variables)
#         for particle in swarm:
#             # Base components
#             cognitive = self.b1 * random.random()
#             social = self.b2 * random.random()
#             escapist = self.ac1 * random.random()
#             rebel = self.ac2 * random.random()
#
#             pbest = np.array(particle.attributes['best_position'])
#             current = np.array(particle.variables)
#
#             # Rebel logic: Inverse social component
#             if particle.attributes['is_rebel']:
#                 social_dir = current - gbest
#                 social_vec = rebel * social_dir
#             else:
#                 social_dir = gbest - current
#                 social_vec = social * social_dir
#
#             # Escapist logic: Inverse cognitive component
#             if particle.attributes['is_escapist']:
#                 cognitive_dir = current - pbest
#                 cognitive_vec = escapist * cognitive_dir
#             else:
#                 cognitive_dir = pbest - current
#                 cognitive_vec = cognitive * cognitive_dir
#
#             # Hybrid velocity update
#             velocity = (self.w * np.array(particle.attributes['velocity'])
#                         + social_vec
#                         + cognitive_vec)
#
#             particle.attributes['velocity'] = velocity.tolist()


# class REAPSO(SingleObjectivePSO):
#     """PSO with rebel and escapist particles and adaptive parameters"""
#
#     def create_initial_solutions(self) -> List[S]:
#         solutions = super().create_initial_solutions()
#         self._mark_special_particles(solutions)
#         return solutions
#
#     def __init__(self,
#                  problem: FloatProblem,
#                  termination_criterion: TerminationCriterion,
#                  swarm_size: int,
#                  b1: float,
#                  b2: float,
#                  ac1: float,
#                  ac2: float,
#                  base_inertia: float,
#                  min_inertia: float,
#                  max_inertia: float,
#                  rebel_fraction: float,
#                  escapist_fraction: float,
#                  window_size: int = 10,
#                  perturbation_probability: float = 0.1,
#                  perturbation_scale: float = 0.1,
#                  max_rebel_fraction: float = 0.8,
#                  max_escapist_fraction: float = 0.8,
#                  diversity_threshold: float = 0.1,
#                  improvement_threshold: float = 0.01):
#         super().__init__(
#             problem=problem,
#             swarm_size=swarm_size,
#             b1=b1,
#             b2=b2,
#             w=base_inertia,
#             termination_criterion=termination_criterion
#         )
#
#         # Dynamic parameters
#         self.ac1 = ac1
#         self.ac2 = ac2
#         self.base_inertia = base_inertia
#         self.min_inertia = min_inertia
#         self.max_inertia = max_inertia
#         self.rebel_fraction = rebel_fraction
#         self.escapist_fraction = escapist_fraction
#         self.max_rebel_fraction = max_rebel_fraction
#         self.max_escapist_fraction = max_escapist_fraction
#         self.original_rebel_fraction = rebel_fraction
#         self.original_escapist_fraction = escapist_fraction
#         self.diversity_threshold = diversity_threshold
#         self.improvement_threshold = improvement_threshold
#
#         # Perturbation parameters
#         self.perturbation_probability = perturbation_probability
#         self.perturbation_scale = perturbation_scale
#
#         # Adaptive state tracking
#         self.window_size = window_size
#         self.convergence_window = deque(maxlen=self.window_size)
#
#     def _mark_special_particles(self, swarm: List[S]):
#         # Ensure minimum 1 particle per type
#         num_rebels = max(1, int(len(swarm) * self.rebel_fraction))
#         num_escapists = max(1, int(len(swarm) * self.escapist_fraction))
#
#         # Select distinct particles for each role
#         all_indices = np.random.permutation(len(swarm))
#         rebels = all_indices[:num_rebels]
#         escapists = all_indices[num_rebels:num_rebels + num_escapists]
#
#         # Assign roles with potential overlap
#         for i, particle in enumerate(swarm):
#             particle.attributes['is_rebel'] = (i in rebels)
#             particle.attributes['is_escapist'] = (i in escapists)
#
#     def update_special_particles(self, swarm: List[S]) -> None:
#         """
#         Adjust the rebel and escapist properties for the swarm based on
#         self.rebel_fraction and self.escapist_fraction.
#         """
#         total_particles = len(swarm)
#
#         # Determine desired counts (ensuring at least one particle per type)
#         desired_num_rebels = max(1, int(total_particles * self.rebel_fraction))
#         desired_num_escapists = max(1, int(total_particles * self.escapist_fraction))
#
#         # Get current particles with these properties
#         current_rebels = [p for p in swarm if p.attributes.get('is_rebel', False)]
#         current_escapists = [p for p in swarm if p.attributes.get('is_escapist', False)]
#
#         # --- Adjust Rebel Particles ---
#         if len(current_rebels) < desired_num_rebels:
#             # Increase: Only assign to those that are not yet rebels.
#             non_rebels = [p for p in swarm if not p.attributes.get('is_rebel', False)]
#             num_to_assign = desired_num_rebels - len(current_rebels)
#             if non_rebels and num_to_assign > 0:
#                 selected = random.sample(non_rebels, min(num_to_assign, len(non_rebels)))
#                 for particle in selected:
#                     particle.attributes['is_rebel'] = True
#         elif len(current_rebels) > desired_num_rebels:
#             # Decrease: Remove rebel property randomly from those that currently are rebels.
#             num_to_remove = len(current_rebels) - desired_num_rebels
#             if current_rebels and num_to_remove > 0:
#                 selected = random.sample(current_rebels, num_to_remove)
#                 for particle in selected:
#                     particle.attributes['is_rebel'] = False
#
#         # --- Adjust Escapist Particles ---
#         if len(current_escapists) < desired_num_escapists:
#             # Increase: Only assign to those that are not yet escapists.
#             non_escapists = [p for p in swarm if not p.attributes.get('is_escapist', False)]
#             num_to_assign = desired_num_escapists - len(current_escapists)
#             if non_escapists and num_to_assign > 0:
#                 selected = random.sample(non_escapists, min(num_to_assign, len(non_escapists)))
#                 for particle in selected:
#                     particle.attributes['is_escapist'] = True
#         elif len(current_escapists) > desired_num_escapists:
#             # Decrease: Remove escapist property randomly.
#             num_to_remove = len(current_escapists) - desired_num_escapists
#             if current_escapists and num_to_remove > 0:
#                 selected = random.sample(current_escapists, num_to_remove)
#                 for particle in selected:
#                     particle.attributes['is_escapist'] = False
#
#     def update_velocity(self, swarm: List[FloatSolution]) -> None:
#         diversity = self.calculate_swarm_diversity(swarm)
#         self.adapt_parameters(diversity, swarm)
#         gbest = np.array(self.best_global.variables)
#         for particle in swarm:
#             # Base components
#             cognitive = self.b1 * random.random()
#             social = self.b2 * random.random()
#             escapist = self.ac1 * random.random()
#             rebel = self.ac2 * random.random()
#
#             pbest = np.array(particle.attributes['best_position'])
#             current = np.array(particle.variables)
#
#             # Rebel logic: Inverse social component
#             if particle.attributes['is_rebel']:
#                 social_dir = current - gbest
#                 social_vec = rebel * social_dir
#             else:
#                 social_dir = gbest - current
#                 social_vec = social * social_dir
#
#             # Escapist logic: Inverse cognitive component
#             if particle.attributes['is_escapist']:
#                 cognitive_dir = current - pbest
#                 cognitive_vec = escapist * cognitive_dir
#             else:
#                 cognitive_dir = pbest - current
#                 cognitive_vec = cognitive * cognitive_dir
#
#             # Hybrid velocity update
#             velocity = (self.w * np.array(particle.attributes['velocity'])
#                         + social_vec
#                         + cognitive_vec)
#
#             particle.attributes['velocity'] = velocity.tolist()
#
#
#     def adapt_parameters(self, diversity: float, swarm: List[FloatSolution]) -> None:
#         # Inertia adaptation remains the same
#         if diversity < self.diversity_threshold:
#             self.w = min(self.max_inertia, self.w * 1.05)
#         else:
#             self.w = max(self.min_inertia, self.w * 0.95)
#
#         # Role adaptation: update ratios based on improvement rate
#         improvement_rate = self.calculate_improvement_rate()
#         if improvement_rate < self.improvement_threshold:
#             self.rebel_fraction = min(self.max_rebel_fraction, self.rebel_fraction * 1.1)
#             self.escapist_fraction = min(self.max_escapist_fraction, self.escapist_fraction * 1.1)
#         else:
#             self.rebel_fraction = max(self.original_rebel_fraction, self.rebel_fraction * 0.95)
#             self.escapist_fraction = max(self.original_escapist_fraction, self.escapist_fraction * 0.95)
#
#         self.update_special_particles(swarm)
#
#     def calculate_swarm_diversity(self, swarm) -> float:
#         """Measure population spread using mean pairwise distance"""
#         positions = np.array([p.variables for p in swarm])
#         centroid = np.mean(positions, axis=0)
#         return np.mean(np.linalg.norm(positions - centroid, axis=1))
#
#     def calculate_improvement_rate(self) -> float:
#         """Calculate relative fitness improvement over the last window_size iterations."""
#         self.convergence_window.append(self.best_global.objectives[0])
#
#         if len(self.convergence_window) < 2:
#             return 0.0
#
#         initial = self.convergence_window[0]
#         latest = self.convergence_window[-1]
#
#         epsilon = 1e-8
#         if abs(initial) < epsilon:
#             return 0.0
#
#         improvement_rate = (initial - latest) / abs(initial)
#         return improvement_rate
#
#     def perturbation(self, swarm: List[S]) -> None:
#         """Chaotic perturbation for diversity maintenance with parameterized probability and scale."""
#         best = self.best_global.variables
#         for particle in swarm:
#             if random.random() < self.perturbation_probability * (1 - self.w):
#                 noise = self.perturbation_scale * (self.max_inertia - self.w) * (np.random.rand() - 0.5)
#                 particle.variables = [
#                     np.clip(x + noise * (x - best[i]),
#                             self.problem.lower_bound[i],
#                             self.problem.upper_bound[i])
#                     for i, x in enumerate(particle.variables)
#                 ]
#
#     def get_name(self) -> str:
#         return "REAPSO"
