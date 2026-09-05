import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class StyblinskiTang(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(StyblinskiTang, self).__init__()
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables

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
        x2 = x * x
        solution.objectives[0] = 0.5 * float(np.sum(x2 * x2 - 16.0 * x2 + 5.0 * x))
        return solution

    def name(self) -> str:
        return "Styblinski–Tang"
