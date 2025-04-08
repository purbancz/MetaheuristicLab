import math
import random

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class GeneralizedSchmidtVetters(FloatProblem):
    """
    Multi-dimensional generalization of Schmidt–Vetters:

      f(x) = sum_{i=1}^{n-1} [ ( sin^2(x_i^2-x_{i+1}^2) + cos^2(x_i^2+x_{i+1}^2) - 1 ) / (1+0.001*(x_i^2+x_{i+1}^2))^2 ]

    Domain: x_i ∈ [-100, 100] for all i.
    Global optimum: f(x)=0 when all x_i=0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(GeneralizedSchmidtVetters, self).__init__()
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    # def create_solution(self) -> FloatSolution:
    #     sol = FloatSolution(self.lower_bound,
    #                         self.upper_bound,
    #                         self.number_of_objectives(),
    #                         self.number_of_constraints())
    #     sol.variables = [random.uniform(lb, ub)
    #                      for lb, ub in zip(self.lower_bound, self.upper_bound)]
    #     return sol

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        d = self.number_of_variables()
        for i in range(d - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i+1]
            numerator = math.sin(x_i**2 - x_next**2)**2 + math.cos(x_i**2 + x_next**2)**2 - 1
            denominator = (1 + 0.001 * (x_i**2 + x_next**2))**2
            total += numerator / denominator
        solution.objectives[0] = total + (self.number_of_variables() - 1) * 0.75
        return solution

    def name(self) -> str:
        return "Generalized Schmidt–Vetters"
