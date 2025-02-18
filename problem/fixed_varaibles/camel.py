import math
from abc import ABC, abstractmethod
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class BaseCamelFunction(FloatProblem, ABC):
    def __init__(self, lower_bounds, upper_bounds):
        super(BaseCamelFunction, self).__init__()

        self.lower_bound = lower_bounds
        self.upper_bound = upper_bounds

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

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


class ThreeHumpCamel(BaseCamelFunction):
    def __init__(self):
        super().__init__([-5.0, -5.0], [5.0, 5.0])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables
        result = (2.0 * x**2) - (1.05 * x**4) + (x**6 / 6.0) + (x * y) + (y**2)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Three-Hump Camel"


class SixHumpCamel(BaseCamelFunction):
    def __init__(self):
        super().__init__([-3.0, -2.0], [3.0, 2.0])

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x, y = solution.variables
        result = (4.0 - 2.1 * x**2 + (x**4) / 3.0) * x**2
        result += x * y
        result += (-4.0 + 4.0 * y**2) * y**2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Six-Hump Camel"
