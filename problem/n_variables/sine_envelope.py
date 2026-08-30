import math
import random
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class SineEnvelope(FloatProblem):
    """
    Generalized SineEnvelope test objective function.

    f(x) = -sum_{i=1}^{n-1} [ sin^2(sqrt(x_i^2 + x_{i+1}^2) - 0.5) / (0.001 * (x_i^2 + x_{i+1}^2) + 1)^2 + 0.5 ]

    Domain: x_i ∈ [-100, 100] for i = 1, …, n.

    Global optimum: f(x) = 0 for x_i = 0 for i = 1, …, n.
    """

    def __init__(self, number_of_variables: int = 2):
        super(SineEnvelope, self).__init__()
        self.lower_bound = [-15.0 for _ in range(number_of_variables)]
        self.upper_bound = [15.0 for _ in range(number_of_variables)]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["SineEnvelope"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for i in range(self.number_of_variables() - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i + 1]
            distance = math.sqrt(x_i**2 + x_next**2)
            component = (math.sin(distance) ** 2) / ((0.001 * (x_i**2 + x_next**2) + 1) ** 2)
            total += component
        solution.objectives[0] = -total
        return solution

    def name(self) -> str:
        return "SineEnvelope"
