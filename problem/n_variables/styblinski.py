import math
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
        x = solution.variables
        result = 0.5 * sum(xi**4 - 16.0 * xi**2 + 5.0 * xi for xi in x)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Styblinski–Tang"
