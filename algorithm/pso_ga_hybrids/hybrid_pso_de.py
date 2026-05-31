import random
import copy
from typing import List, TypeVar

import numpy as np
from jmetal.config import store
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import Solution
from jmetal.operator import DifferentialEvolutionCrossover
from jmetal.operator.selection import DifferentialEvolutionSelection
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.comparator import ObjectiveComparator
from algorithm.basic.single_objective_pso import SingleObjectivePSO

S = TypeVar("S")

class HybridPSODE(SingleObjectivePSO):
    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 de_probability: float,
                 crossover_operator: DifferentialEvolutionCrossover = DifferentialEvolutionCrossover(CR=0.9, F=0.5),
                 selection_operator: DifferentialEvolutionSelection = DifferentialEvolutionSelection(),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.de_probability = de_probability
        self.crossover_operator = crossover_operator
        self.selection_operator = selection_operator
        self.comparator = ObjectiveComparator(0)

    def perturbation(self, swarm: List[S]) -> None:
        # DE-based perturbation
        for i, particle in enumerate(swarm):
            if random.random() < self.de_probability:
                self.selection_operator.set_index_to_exclude(i)
                parents = self.selection_operator.execute(swarm)
                self.crossover_operator.current_individual = particle
                children = self.crossover_operator.execute(parents)
                trial = children[0]
                self.problem.evaluate(trial)
                if trial.objectives[0] < particle.objectives[0]:
                    swarm[i] = trial