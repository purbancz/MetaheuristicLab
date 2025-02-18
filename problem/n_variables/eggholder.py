import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class EggHolder(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(EggHolder, self).__init__()
        self.lower_bound = [-512.0] * number_of_variables
        self.upper_bound = [512.0] * number_of_variables

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
        total = 0.0
        for i in range(len(x) - 1):
            term1 = - x[i] * math.sin(math.sqrt(abs(x[i] - x[i+1] - 47)))
            term2 = -(x[i+1] + 47) * math.sin(math.sqrt(abs(0.5 * x[i] + x[i+1] + 47)))
            total += term1 + term2
        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return "Egg-Holder"
