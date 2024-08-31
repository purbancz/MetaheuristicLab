from typing import TypeVar, List
import numpy as np
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.config import store
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.evaluator import Evaluator
from jmetal.util.generator import Generator
from jmetal.util.comparator import ObjectiveComparator, Comparator
import random
import time
from copy import deepcopy

S = TypeVar("S")
R = TypeVar("R")


class SingleObjectivePSO(Algorithm[S, R]):
    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria,
                 particle_evaluator: Evaluator = store.default_evaluator,
                 swarm_generator: Generator = store.default_generator,
                 solution_comparator: Comparator = ObjectiveComparator(0)):
        super().__init__()
        self.problem = problem
        self.swarm_size = swarm_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.best_global = None
        self.termination_criterion = termination_criterion
        self.particle_evaluator = particle_evaluator
        self.swarm_generator = swarm_generator
        self.solution_comparator = solution_comparator
        self.observable.register(termination_criterion)

    def create_initial_solutions(self):
        self.solutions = [self.swarm_generator.new(self.problem) for _ in range(self.swarm_size)]
        for solution in self.solutions:
            self.problem.evaluate(solution)
            solution.attributes['best_position'] = solution.variables[:]
            solution.attributes['best_objective'] = solution.objectives[0]
            solution.attributes['velocity'] = np.random.uniform(-1, 1, self.problem.number_of_variables())
        self.best_global = deepcopy(min(self.solutions, key=lambda sol: sol.objectives[0]))
        return self.solutions

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

    def evaluate(self, solution_list):
        return self.particle_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self):
        self.evaluations = self.swarm_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        for i in range(self.swarm_size):
            particle = self.solutions[i]
            personal_best = particle.attributes['best_position']
            global_best = self.best_global.variables
            velocity = particle.attributes['velocity']

            for j in range(self.problem.number_of_variables()):
                r1 = random.random()
                r2 = random.random()
                velocity[j] = (self.w * velocity[j] +
                               self.c1 * r1 * (personal_best[j] - particle.variables[j]) +
                               self.c2 * r2 * (global_best[j] - particle.variables[j]))
                particle.variables[j] += velocity[j]

                # Boundary check
                if particle.variables[j] < self.problem.lower_bound[j]:
                    particle.variables[j] = self.problem.lower_bound[j]
                elif particle.variables[j] > self.problem.upper_bound[j]:
                    particle.variables[j] = self.problem.upper_bound[j]

            self.problem.evaluate(particle)
            self.update_best(particle)

    def update_best(self, particle):
        if particle.objectives[0] < particle.attributes['best_objective']:
            particle.attributes['best_position'] = particle.variables[:]
            particle.attributes['best_objective'] = particle.objectives[0]

        if particle.objectives[0] < self.best_global.attributes['best_objective']:
            self.best_global = deepcopy(particle)

    def update_progress(self):
        self.evaluations += self.swarm_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem.name(),
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> R:
        return self.best_global

    def get_name(self) -> str:
        return "SingleObjectivePSO"
