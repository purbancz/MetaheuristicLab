import random
from typing import List, TypeVar

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar('S')


class CoAdaptativePSO(SingleObjectivePSO):
    """
    CoAdaptativePSO
    Common, Collective,
    """

    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, max_c1: float, max_c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.max_c1 = max_c1
        self.max_c2 = max_c2
        self.min_c1 = c1
        self.min_c2 = c2

    def update_coefficient(self):
        for particle in self.solutions:
            if particle.objectives[0] < particle.attributes.get('best_objective', float('inf')):
                self.c1 = min(self.max_c1, self.c1 * 1.1)
                self.c2 = max(self.min_c2, self.c2 * 0.9)
            else:
                self.c2 = min(self.max_c2, self.c2 * 1.1)
                self.c1 = max(self.min_c1, self.c1 * 0.9)

    def step(self):
        self.update_coefficient()
        super().step()

    def get_name(self) -> str:
        return "CAPSO"


class IndividualAdaptivePSO(SingleObjectivePSO):
    """
    IndividualAdaptivePSO
    Independently
    """

    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, max_c1: float, max_c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.max_c1 = max_c1
        self.max_c2 = max_c2
        self.min_c1 = c1
        self.min_c2 = c2

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        for sol in solutions:
            sol.attributes['c1'] = self.c1
            sol.attributes['c2'] = self.c2
        return solutions

    def update_coefficient(self):
        for particle in self.solutions:
            if particle.objectives[0] < particle.attributes.get('best_objective', float('inf')):
                particle.attributes['c1'] = min(self.max_c1, particle.attributes['c1'] * 1.1)
                particle.attributes['c2'] = max(self.min_c2, particle.attributes['c2'] * 0.9)
            else:
                particle.attributes['c2'] = min(self.max_c2, particle.attributes['c2'] * 1.1)
                particle.attributes['c1'] = max(self.min_c1, particle.attributes['c1'] * 0.9)

    def update_velocity(self, swarm: List[S]) -> None:
        global_best = np.array(self.best_global.variables)
        for particle in swarm:
            r1, r2 = random.random(), random.random()
            c1, c2 = particle.attributes['c1'], particle.attributes['c2']
            velocity = np.array(particle.attributes['velocity'])
            personal_best = np.array(particle.attributes['best_position'])
            current = np.array(particle.variables)

            new_velocity = (self.w * velocity +
                            c1 * r1 * (personal_best - current) +
                            c2 * r2 * (global_best - current))

            particle.attributes['velocity'] = new_velocity.tolist()

    def step(self):
        self.update_coefficient()
        super().step()

    def get_name(self) -> str:
        return "IAPSO"
