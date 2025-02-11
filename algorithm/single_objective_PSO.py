from copy import deepcopy
from typing import List, TypeVar
import random
import numpy as np
from jmetal.core.algorithm import ParticleSwarmOptimization
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar('S')
R = List[FloatSolution]


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

    def create_initial_solutions(self) -> List[FloatSolution]:
        self.solutions = [self.problem.create_solution() for _ in range(self.swarm_size)]
        return self.solutions

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return [self.problem.evaluate(sol) for sol in solution_list]

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def initialize_velocity(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables())

    def initialize_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            particle.attributes['best_position'] = np.array(particle.variables.copy())
            particle.attributes['best_objective'] = particle.objectives[0]

    def initialize_global_best(self, swarm: List[FloatSolution]) -> None:
        self.best_global = min(swarm, key=lambda x: x.objectives[0])

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            r1 = random.random()
            r2 = random.random()
            velocity = np.array(particle.attributes['velocity'])
            pbest = np.array(particle.attributes['best_position'])
            gbest = np.array(self.best_global.variables)

            new_velocity = (self.w * velocity +
                           self.c1 * r1 * (pbest - np.array(particle.variables)) +
                           self.c2 * r2 * (gbest - np.array(particle.variables)))

            particle.attributes['velocity'] = new_velocity.tolist()

    def update_position(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            # Convert to numpy array for vector operations
            current_position = np.array(particle.variables)
            new_position = current_position + np.array(particle.attributes['velocity'])

            # Apply bounds and convert back to list
            clipped_position = np.clip(new_position,
                                       self.problem.lower_bound,
                                       self.problem.upper_bound)
            particle.variables = clipped_position.tolist()

    def update_particle_best(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            if particle.objectives[0] < particle.attributes['best_objective']:
                particle.attributes['best_position'] = particle.variables.copy()
                particle.attributes['best_objective'] = particle.objectives[0]

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        current_best = min(swarm, key=lambda x: x.objectives[0])
        if current_best.objectives[0] < self.best_global.objectives[0]:
            self.best_global = current_best

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


    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass  # Optional implementation

    def result(self) -> FloatSolution:
        return self.best_global

    def get_name(self) -> str:
        return "SingleObjectivePSO"


class RebelPSO(SingleObjectivePSO):
    """PSO with rebel particles opposing global best"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 rebel_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.rebel_fraction = rebel_fraction

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self._mark_rebels(solutions)
        return solutions

    def _mark_rebels(self, swarm: List[FloatSolution]):
        num_rebels = max(1, int(len(swarm) * self.rebel_fraction))
        rebels = random.sample(swarm, num_rebels)
        for particle in rebels:
            particle.attributes['is_rebel'] = True


class EscapistPSO(SingleObjectivePSO):
    """PSO with escapist particles opposing personal best"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.escapist_fraction = escapist_fraction

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self._mark_escapists(solutions)
        return solutions

    def _mark_escapists(self, swarm: List[FloatSolution]):
        num_escapists = max(1, int(len(swarm) * self.escapist_fraction))
        escapists = random.sample(swarm, num_escapists)
        for particle in escapists:
            particle.attributes['is_escapist'] = True


class RebelEscapistPSO(SingleObjectivePSO):
    """PSO with both rebel and escapist particles"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 rebel_fraction: float,
                 escapist_fraction: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.rebel_fraction = rebel_fraction
        self.escapist_fraction = escapist_fraction

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self._mark_special_particles(solutions)
        return solutions

    def _mark_special_particles(self, swarm: List[FloatSolution]):
        # Mark rebels
        num_rebels = max(1, int(len(swarm) * self.rebel_fraction))
        rebels = random.sample(swarm, num_rebels)
        for particle in rebels:
            particle.attributes['is_rebel'] = True

        # Mark escapists from remaining particles
        remaining = [p for p in swarm if 'is_rebel' not in p.attributes]
        num_escapists = max(1, int(len(remaining) * self.escapist_fraction))
        escapists = random.sample(remaining, num_escapists)
        for particle in escapists:
            particle.attributes['is_escapist'] = True


class REAPSO(SingleObjectivePSO):
    """PSO with rebel and escapist particles and adaptive parameters"""

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 base_inertia: float,
                 min_inertia: float,
                 max_inertia: float,
                 rebel_ratio: float,
                 escapist_ratio: float,
                 termination_criterion: TerminationCriterion):

        super().__init__(
            problem=problem,
            swarm_size=swarm_size,
            c1=1.496,  # Cognitive coefficient
            c2=1.496,  # Social coefficient
            w=base_inertia,
            termination_criterion=termination_criterion
        )

        # Dynamic parameters
        self.base_inertia = base_inertia
        self.min_inertia = min_inertia
        self.max_inertia = max_inertia
        self.rebel_ratio = rebel_ratio
        self.escapist_ratio = escapist_ratio

        # Adaptive state tracking
        self.convergence_window = []
        self.diversity_threshold = 0.1

    def create_initial_solutions(self) -> List[FloatSolution]:
        solutions = super().create_initial_solutions()
        self._mark_special_particles(solutions)
        return solutions

    def _mark_special_particles(self, swarm: List[FloatSolution]):
        # Ensure minimum 1 particle per type
        num_rebels = max(1, int(len(swarm) * self.rebel_ratio))
        num_escapists = max(1, int(len(swarm) * self.escapist_ratio))

        # Select distinct particles for each role
        all_indices = np.random.permutation(len(swarm))
        rebels = all_indices[:num_rebels]
        escapists = all_indices[num_rebels:num_rebels + num_escapists]

        # Assign roles with potential overlap
        for i, particle in enumerate(swarm):
            particle.attributes['is_rebel'] = (i in rebels)
            particle.attributes['is_escapist'] = (i in escapists)

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        diversity = self.calculate_swarm_diversity(swarm)
        self.adapt_parameters(diversity)

        for particle in swarm:
            # Base components
            cognitive = self.c1 * random.random()
            social = self.c2 * random.random()

            # Get reference points
            pbest = np.array(particle.attributes['best_position'])
            gbest = np.array(self.best_global.variables)
            current = np.array(particle.variables)

            # Rebel logic: Inverse social component
            if particle.attributes['is_rebel']:
                social_dir = current - gbest
            else:
                social_dir = gbest - current

            # Escapist logic: Inverse cognitive component
            if particle.attributes['is_escapist']:
                cognitive_dir = current - pbest
            else:
                cognitive_dir = pbest - current

            # Hybrid velocity update
            velocity = (self.w * np.array(particle.attributes['velocity']) +
                        cognitive * cognitive_dir +
                        social * social_dir)

            particle.attributes['velocity'] = velocity.tolist()

    def adapt_parameters(self, diversity: float):
        """Dynamic parameter adaptation based on swarm state"""
        # Inertia adaptation
        if diversity < self.diversity_threshold:
            self.w = min(self.max_inertia, self.w * 1.05)  # Encourage exploration
        else:
            self.w = max(self.min_inertia, self.w * 0.95)  # Encourage exploitation

        # Role adaptation
        improvement_rate = self.calculate_improvement_rate()
        if improvement_rate < 0.01:
            self.rebel_ratio = min(0.3, self.rebel_ratio * 1.1)
            self.escapist_ratio = min(0.3, self.escapist_ratio * 1.1)

    def calculate_swarm_diversity(self, swarm) -> float:
        """Measure population spread using mean pairwise distance"""
        positions = np.array([p.variables for p in swarm])
        centroid = np.mean(positions, axis=0)
        return np.mean(np.linalg.norm(positions - centroid, axis=1))

    def calculate_improvement_rate(self) -> float:
        """Track fitness improvement over the last N iterations."""
        window_size = 10
        self.convergence_window.append(self.best_global.objectives[0])

        # if len(self.convergence_window) < window_size:
        #     return 1.0
        #
        # return (self.convergence_window[-window_size] -
        #         self.convergence_window[-1]) / self.convergence_window[-window_size]

        effective_window_size = min(len(self.convergence_window), window_size)
        return (self.convergence_window[-effective_window_size] -
                self.convergence_window[-1]) / self.convergence_window[-effective_window_size]

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        """Chaotic perturbation for diversity maintenance"""
        best = self.best_global.variables
        for particle in swarm:
            if random.random() < 0.1 * (1 - self.w):
                noise = 0.1 * (self.max_inertia - self.w) * (np.random.rand() - 0.5)
                particle.variables = [
                    np.clip(x + noise * (x - best[i]),
                            self.problem.lower_bound[i],
                            self.problem.upper_bound[i]
                            ) for i, x in enumerate(particle.variables)]