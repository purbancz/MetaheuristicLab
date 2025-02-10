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
        self.global_best = None

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
        self.global_best = min(swarm, key=lambda x: x.objectives[0])

    def update_velocity(self, swarm: List[FloatSolution]) -> None:
        for particle in swarm:
            r1 = random.random()
            r2 = random.random()
            velocity = np.array(particle.attributes['velocity'])
            pbest = np.array(particle.attributes['best_position'])
            gbest = np.array(self.global_best.variables)

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
        if current_best.objectives[0] < self.global_best.objectives[0]:
            self.global_best = current_best

    def perturbation(self, swarm: List[FloatSolution]) -> None:
        pass  # Optional implementation

    def result(self) -> FloatSolution:
        return self.global_best

    def get_name(self) -> str:
        return "SingleObjectivePSO"


# Updated RebelPSO class
class RebelPSO(SingleObjectivePSO):
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


# Similarly update EscapistPSO and EscapistRebelPSO
class EscapistPSO(SingleObjectivePSO):
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


class EscapistRebelPSO(SingleObjectivePSO):
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