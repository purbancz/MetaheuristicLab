from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import numpy as np


class Shekel(FloatProblem):
    def __init__(self, m: int = 10):
        super(Shekel, self).__init__()
        self.lower_bound = [0] * 4
        self.upper_bound = [10] * 4

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

        self.m = m  # Number of peaks

        # Predefined parameters for the Shekel function
        self.C = np.array([[4, 4, 4, 4],
                           [1, 1, 1, 1],
                           [8, 8, 8, 8],
                           [6, 6, 6, 6],
                           [3, 7, 3, 7],
                           [2, 9, 2, 9],
                           [5, 5, 3, 3],
                           [8, 1, 8, 1],
                           [6, 2, 6, 2],
                           [7, 3.6, 7, 3.6]])[:self.m]  # Up to m peaks
        self.beta = np.array([0.1, 0.2, 0.2, 0.4, 0.4, 0.6, 0.3, 0.7, 0.5, 0.5])[:self.m]

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        outer = 0.0
        for i in range(self.m):
            inner = np.sum((x - self.C[i])**2)
            outer += 1 / (inner + self.beta[i])
        result = -outer
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'Shekel'
