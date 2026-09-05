import math

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Michalewicz(FloatProblem):
    def __init__(self, number_of_variables: int = 2, m: float = 10):
        super(Michalewicz, self).__init__()
        self.lower_bound = [0.0] * number_of_variables
        self.upper_bound = [math.pi] * number_of_variables
        self.m = m

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.asarray(solution.variables, dtype=float)
        i = np.arange(1, x.size + 1)
        solution.objectives[0] = -float(np.sum(np.sin(x) * np.sin(i * x * x / math.pi) ** (2 * self.m)))
        return solution

    def name(self) -> str:
        return 'Michalewicz'
