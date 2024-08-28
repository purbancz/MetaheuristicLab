from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
import numpy as np


class Hartmann(FloatProblem):
    def __init__(self):
        super(Hartmann, self).__init__()
        self.lower_bound = [0, 0, 0]
        self.upper_bound = [1, 1, 1]

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array([[3.0, 10, 30],
                      [0.1, 10, 35],
                      [3.0, 10, 30],
                      [0.1, 10, 35]])
        P = np.array([[0.3689, 0.1170, 0.2673],
                      [0.4699, 0.4387, 0.7470],
                      [0.1091, 0.8732, 0.5547],
                      [0.03815, 0.5743, 0.8828]])

        outer = 0.0
        for i in range(4):
            inner = 0.0
            for j in range(3):
                xj = solution.variables[j]
                Aij = A[i][j]
                Pij = P[i][j]
                inner += Aij * ((xj - Pij) ** 2)
            outer += alpha[i] * np.exp(-inner)
        result = -outer
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return 'Hartmann (3,4)'
