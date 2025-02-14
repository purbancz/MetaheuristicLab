import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.single_objective_PSO import SingleObjectivePSO


class SPPPSO(SingleObjectivePSO):
    """
    Symbiotic Predator-Prey PSO (SPP-PSO)
    Concept: Mimics ecological interactions with three particle types:
      - Predators: Chase worst solutions to push swarm away from bad regions
      - Prey: Standard PSO particles
      - Scavengers: Exploit areas between predators and prey
    """
    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion, predator_ratio=0.05, scavenger_ratio=0.2):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.predator_ratio = predator_ratio
        self.scavenger_ratio = scavenger_ratio

    def create_initial_solutions(self) -> [FloatSolution]:
        solutions = super().create_initial_solutions()
        self.initialize_roles(solutions)
        return solutions

    def initialize_roles(self, swarm: [FloatSolution]):
        for particle in swarm:
            rand = np.random.rand()
            if rand < self.predator_ratio:
                particle.role = "predator"
            elif rand < self.scavenger_ratio:
                particle.role = "scavenger"
            else:
                particle.role = "prey"

    def update_velocity(self, swarm: [FloatSolution]) -> None:
        worst = max(swarm, key=lambda x: x.objectives[0])
        for particle in swarm:
            current = np.array(particle.variables)
            velocity = np.array(particle.attributes['velocity'])
            if particle.role == "predator":
                # Predator: aggressively chase the worst solution
                direction = np.array(worst.variables) - current
                new_velocity = self.w * velocity + 2.5 * np.random.rand() * direction
            elif particle.role == "scavenger":
                # Scavenger: head for the midpoint between best and worst
                midpoint = (np.array(self.best_global.variables) + np.array(worst.variables)) / 2
                direction = midpoint - current
                new_velocity = self.w * velocity + 1.0 * np.random.rand() * direction
            else:
                # Prey: standard PSO behavior
                cognitive = self.c1 * np.random.rand() * (np.array(particle.attributes['best_position']) - current)
                social = self.c2 * np.random.rand() * (np.array(self.best_global.variables) - current)
                new_velocity = self.w * velocity + cognitive + social
            particle.attributes['velocity'] = new_velocity.tolist()