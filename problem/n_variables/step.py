import math
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class BaseStepFunction(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(BaseStepFunction, self).__init__()
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
        raise NotImplementedError("Subclasses must override evaluate().")

    def name(self) -> str:
        raise NotImplementedError("Subclasses must override name().")


class StepN1(BaseStepFunction):
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        result = sum(math.floor(abs(xi)) for xi in solution.variables)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Step N1"


class StepN2(BaseStepFunction):
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        result = sum(math.floor(abs(xi + 0.5)) ** 2 for xi in solution.variables)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Step N2"


class StepN3(BaseStepFunction):
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        result = sum(math.floor(xi ** 2) for xi in solution.variables)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Step N3"
