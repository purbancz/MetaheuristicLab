import math

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class StretchedV(FloatProblem):
    """
    StretchedV test objective function.

    This class defines the Stretched V global optimization problem.
    A multimodal minimization problem.
    """

    def __init__(self, number_of_variables: int = 2):
        super(StretchedV, self).__init__()
        self.lower_bound = [-100.0 for _ in range(number_of_variables)]
        self.upper_bound = [100.0 for _ in range(number_of_variables)]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]

        # Define the bounds for the solution space
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.asarray(solution.variables, dtype=float)
        t = x[:-1] ** 2 + x[1:] ** 2
        terms = (t ** 0.25) * (np.sin(50 * (t ** 0.1)) + 1) ** 2
        solution.objectives[0] = float(np.sum(terms))
        return solution

    def name(self) -> str:
        return "StretchedV"

