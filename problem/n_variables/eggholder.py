import math

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class EggHolder(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(EggHolder, self).__init__()
        self.lower_bound = [-512.0] * number_of_variables
        self.upper_bound = [512.0] * number_of_variables

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
        a, b = x[:-1], x[1:]
        term1 = -a * np.sin(np.sqrt(np.abs(a - b - 47)))
        term2 = -(b + 47) * np.sin(np.sqrt(np.abs(0.5 * a + b + 47)))
        solution.objectives[0] = float(np.sum(term1 + term2))
        return solution

    def name(self) -> str:
        return "Egg-Holder"
