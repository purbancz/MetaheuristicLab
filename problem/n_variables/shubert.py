import math
import random
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


# ----------------------------
# Shubert 1: Generalized to n dimensions
# ----------------------------

class ShubertN1(FloatProblem):
    """
    Generalized Shubert 1 test objective function.

    We define:
        f(x) = ∏(d=1 to n) S(x_d),
    where
        S(x) = ∑(i=1 to 5) [ i · cos((i+1)*x + i) ].

    Domain: x_d ∈ [-10, 10] for d = 1, …, n.

    (For the original 2D case, one known global minimizer is approximately
     x ≈ [-7.0835, 4.8580] with f(x) ≈ -186.7309, among many others.)
    """

    def __init__(self, number_of_variables: int = 2):
        super(ShubertN1, self).__init__()
        self.lower_bound = [-10.0 for _ in range(number_of_variables)]
        self.upper_bound = [10.0 for _ in range(number_of_variables)]
        self.obj_directions = [self.MINIMIZE]
        # Set class-level bounds for FloatSolution
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        product = 1.0
        for d in range(self.number_of_variables()):
            x_d = solution.variables[d]
            # Compute the one-dimensional Shubert component for x_d
            component = sum(i * math.cos((i + 1) * x_d + i) for i in range(1, 6))
            product *= component
        solution.objectives[0] = product
        return solution

    def name(self) -> str:
        return "Shubert N1"


# ----------------------------
# Shubert 3: Generalized (separable sum over dimensions)
# ----------------------------

class ShubertN3(FloatProblem):
    """
    Generalized Shubert 3 test objective function.

    We define:
      f(x) = ∑(d=1 to n) S3(x_d),
    where
      S3(x) = ∑(j=1 to 5) [ j*sin((j+1)*x) + j ].

    Domain: x_d ∈ [-10, 10] for all d.

    (For the 2D case, one reported global optimum is approximately -24.062499.)
    """

    def __init__(self, number_of_variables: int = 2):
        super(ShubertN3, self).__init__()
        self.lower_bound = [-10.0 for _ in range(number_of_variables)]
        self.upper_bound = [10.0 for _ in range(number_of_variables)]
        self.obj_directions = [self.MINIMIZE]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            component = sum(j * math.sin((j + 1) * x) + j for j in range(1, 6))
            total += component
        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return "Shubert N3"


# ----------------------------
# Shubert 4: Generalized (separable sum over dimensions)
# ----------------------------

class ShubertN4(FloatProblem):
    """
    Generalized Shubert 4 test objective function.

    We define:
      f(x) = ∑(d=1 to n) S4(x_d),
    where
      S4(x) = ∑(j=1 to 5) [ j*cos((j+1)*x) + j ].

    Domain: x_d ∈ [-10, 10] for d = 1, …, n.

    (For the 2D case, one reported global optimum is approximately -29.016015.)
    """

    def __init__(self, number_of_variables: int = 2):
        super(ShubertN4, self).__init__()
        self.lower_bound = [-10.0 for _ in range(number_of_variables)]
        self.upper_bound = [10.0 for _ in range(number_of_variables)]
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["Shubert 4"]
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        total = 0.0
        for x in solution.variables:
            component = sum(j * math.cos((j + 1) * x) + j for j in range(1, 6))
            total += component
        solution.objectives[0] = total
        return solution

    def name(self) -> str:
        return "Shubert N4"
