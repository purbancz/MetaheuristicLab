import math
from abc import ABC, abstractmethod
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class Base2DProblem(FloatProblem, ABC):
    def __init__(self, lower_bounds, upper_bounds):
        super(Base2DProblem, self).__init__()

        self.lower_bound = lower_bounds
        self.upper_bound = upper_bounds

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x, y)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_variables(self) -> int:
        return 2

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    @abstractmethod
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        pass

    @abstractmethod
    def name(self) -> str:
        pass


class McCormick(Base2DProblem):
    def __init__(self):
        super(McCormick, self).__init__([-1.5, -3.0], [4.0, 4.0])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables

        result = math.sin(x + y) + (x - y)**2 - 1.5 * x + 2.5 * y + 1
        solution.objectives[0] = result

        return solution

    def name(self) -> str:
        return "McCormick"
