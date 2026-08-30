import math
import random

from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class SchwefelN26(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(SchwefelN26, self).__init__()
        self.lower_bound = [-500.0] * number_of_variables
        self.upper_bound = [500.0] * number_of_variables

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
        solution.objectives[0] = 418.9829 * len(x) - sum(xi * math.sin(math.sqrt(abs(xi))) for xi in x)
        return solution

    def name(self) -> str:
        return 'Schwefel N26'


class SchwefelN21(FloatProblem):
    """
    Schwefel 2.21 function:

      f(x) = max_{i=1,...,d} |x_i|

    Domain: x_i ∈ [-100, 100]
    Global optimum: f(0) = 0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(SchwefelN21, self).__init__()
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    # def create_solution(self) -> FloatSolution:
    #     new_solution = FloatSolution(
    #         self.lower_bound,
    #         self.upper_bound,
    #         self.number_of_objectives(),
    #         self.number_of_constraints())
    #     new_solution.variables = [
    #         random.uniform(lb, ub)
    #         for lb, ub in zip(self.lower_bound, self.upper_bound)]
    #     return new_solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = max(abs(xi) for xi in solution.variables)
        return solution

    def name(self) -> str:
        return "Schwefel N21"


# 4. Schwefel 2.22 Function
class SchwefelN22(FloatProblem):
    """
    Schwefel 2.22 function:

      f(x) = sum_{i=1}^{d} |x_i| + prod_{i=1}^{d} |x_i|

    Domain: x_i ∈ [-10, 10]
    Global optimum: f(0) = 0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(SchwefelN22, self).__init__()
        self.lower_bound = [-10.0] * number_of_variables
        self.upper_bound = [10.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    # def create_solution(self) -> FloatSolution:
    #     new_solution = FloatSolution(
    #         self.lower_bound,
    #         self.upper_bound,
    #         self.number_of_objectives(),
    #         self.number_of_constraints())
    #     new_solution.variables = [
    #         random.uniform(lb, ub)
    #         for lb, ub in zip(self.lower_bound, self.upper_bound)]
    #     return new_solution

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        s = sum(abs(xi) for xi in solution.variables)
        p = 1.0
        for xi in solution.variables:
            p *= abs(xi)
        solution.objectives[0] = s + p
        return solution

    def name(self) -> str:
        return "Schwefel N22"


class SchwefelN6(FloatProblem):
    """
    Schwefel6 function:

      f(x) = | sum_{i=1}^{d} x_i | + sum_{i=1}^{d} |x_i|

    Domain: x_i ∈ [-100, 100]
    Global optimum: f(0) = 0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(SchwefelN6, self).__init__()
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0


    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        sum_x = sum(solution.variables)
        sum_abs = sum(abs(xi) for xi in solution.variables)
        solution.objectives[0] = abs(sum_x) + sum_abs
        return solution

    def name(self) -> str:
        return "Schwefel N6"


# Schwefel20 Function
class SchwefelN20(FloatProblem):
    """
    Schwefel20 function:

      f(x) = max_{i=1,...,d} | sum_{j=1}^{i} x_j |

    Domain: x_i ∈ [-100, 100]
    Global optimum: f(0) = 0.
    """

    def __init__(self, number_of_variables: int = 10):
        super(SchwefelN20, self).__init__()
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        cumulative = 0.0
        max_val = -float("inf")
        for xi in solution.variables:
            cumulative += xi
            max_val = max(max_val, abs(cumulative))
        solution.objectives[0] = max_val
        return solution

    def name(self) -> str:
        return "Schwefel N20"


# Schwefel36 Function
class SchwefelN36(FloatProblem):
    """
    Schwefel36 function:

      f(x) = sum_{i=1}^{d} ( 418.9829 - x_i * sin(sqrt(|x_i|)) )^2

    Domain: x_i ∈ [-500, 500]
    Global optimum: Approximately f(x) = 0 when x_i ≈ 420.9687 for all i.
    """

    def __init__(self, number_of_variables: int = 10):
        super(SchwefelN36, self).__init__()
        self.lower_bound = [-500.0] * number_of_variables
        self.upper_bound = [500.0] * number_of_variables

        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']

        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0


    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        solution.objectives[0] = sum(
            (418.9829 - xi * math.sin(math.sqrt(abs(xi)))) ** 2
            for xi in solution.variables
        )
        return solution

    def name(self) -> str:
        return "Schwefel N36"