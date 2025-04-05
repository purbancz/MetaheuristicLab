import math
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class HappyCat(FloatProblem):
    def __init__(self, number_of_variables: int = 2, alpha: float = 0.25):
        super(HappyCat, self).__init__()
        self.alpha = alpha

        self.lower_bound = [-2.0] * number_of_variables
        self.upper_bound = [2.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound


    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0



    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = [ (||x||^2 - d)^2 ]^alpha + (1/d)*(0.5||x||^2 + sum(x)) + 0.5
        x = np.array(solution.variables)
        d = self.number_of_variables()
        norm = np.sum(x ** 2)
        result = ((norm - d) ** 2) ** self.alpha + (1.0 / d) * (0.5 * norm + np.sum(x)) + 0.5
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Happy Cat"
