import math
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
        total = 0.0
        d = self.number_of_variables()

        for i in range(d - 1):
            x_i = solution.variables[i]
            x_next = solution.variables[i + 1]
            t = x_i**2 + x_next**2
            term = (t**(1/4)) * (math.sin(50 * (t**0.1)) + 1)**2
            total += term

        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return "StretchedV"

