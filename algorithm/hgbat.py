import math
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class HGBat(FloatProblem):
    """
      f(x) = sqrt(|(∑_{i=1}^d x_i^2)^2 - (∑_{i=1}^d x_i)^2|)
             + (0.5*∑_{i=1}^d x_i^2 + ∑_{i=1}^d x_i) / d + 0.5
    """

    def __init__(self, number_of_variables: int = 30):
        super(HGBat, self).__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self.number_of_variables()
        sum_sq = np.sum(x ** 2)
        sum_lin = np.sum(x)
        # sqrt( | (∑ x_i^2)^2 - (∑ x_i)^2 | )
        term1 = math.sqrt(abs(sum_sq ** 2 - sum_lin ** 2))
        term2 = (0.5 * sum_sq + sum_lin) / d
        result = term1 + term2 + 0.5
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "HGBat"
