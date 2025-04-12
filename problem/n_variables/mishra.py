import math
import random

import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution


class Mishra01(FloatProblem):
    """
    Mishra 1 test objective function.
    """

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def __init__(self, dimensions=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.zeros(dimensions)
        self.upper_bound = np.ones(dimensions)

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Objective function definition:
        # f(x) = (1 + x_n) ^ x_n where x_n = n - sum(x_i for i=1 to n-1)
        n = self.number_of_variables()
        x = solution.variables
        x_n = n - np.sum(x[:-1])  # x_n is computed as per the formula
        objective_value = (1 + x_n) ** x_n
        solution.objectives[0] = objective_value
        return solution

    def name(self) -> str:
        return "Mishra N1"


class Mishra02(FloatProblem):
    """
    Mishra 2 test objective function.
    """
    def __init__(self, dimensions=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.zeros(dimensions)
        self.upper_bound = np.ones(dimensions)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Objective function definition:
        # f(x) = (1 + x_n) ^ x_n where x_n = n - sum((x_i + x_(i+1)) / 2 for i=1 to n-1)
        n = self.number_of_variables()
        x = solution.variables
        x_n = n - np.sum((np.array(x[:-1]) + np.array(x[1:])) / 2)
        objective_value = (1 + x_n) ** x_n
        solution.objectives[0] = objective_value
        return solution

    def name(self) -> str:
        return "Mishra N2"


class Mishra03(FloatProblem):
    """
    Mishra 3 test objective function.
    """
    def __init__(self, dimensions=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.full(dimensions, -10)
        self.upper_bound = np.full(dimensions, 10)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Objective function definition:
        # f(x) = sqrt(|cos(sqrt(|x_1^2 + x_2^2|))|) + 0.01*(x_1 + x_2)
        x = solution.variables
        objective_value = np.sqrt(np.abs(np.cos(np.sqrt(np.abs(np.sum(np.array(x)**2)))))) + 0.01 * np.sum(x)
        solution.objectives[0] = objective_value
        return solution

    def name(self) -> str:
        return "Mishra N3"


class Mishra04(FloatProblem):
    """
    Mishra 4 test objective function.
    """
    def __init__(self, dimensions=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.full(dimensions, -10)
        self.upper_bound = np.full(dimensions, 10)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Convert x to a NumPy array (if it's a list)
        x = np.array(solution.variables)

        # Now, we can perform element-wise operations on x
        objective_value = np.sqrt(np.abs(np.sin(np.sqrt(np.abs(np.sum(x ** 2)))))) + 0.01 * np.sum(x)

        solution.objectives[0] = objective_value
        return solution

    def name(self) -> str:
        return "Mishra N4"


class Mishra05(FloatProblem):
    """
    Mishra 5 test objective function.
    """
    def __init__(self, dimensions=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.full(dimensions, -10)
        self.upper_bound = np.full(dimensions, 10)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Objective function definition:
        # f(x) = [sin^2((cos(x_1) + cos(x_2))^2) + cos^2((sin(x_1) + sin(x_2))^2) + x_1]^2 + 0.01(x_1 + x_2)
        x = solution.variables
        term1 = np.sin((np.cos(x[0]) + np.cos(x[1]))**2)**2
        term2 = np.cos((np.sin(x[0]) + np.sin(x[1]))**2)**2
        objective_value = (term1 + term2 + x[0])**2 + 0.01 * np.sum(x)
        solution.objectives[0] = objective_value
        return solution

    def name(self) -> str:
        return "Mishra N5"


class Mishra06(FloatProblem):
    """
    Mishra 6 test objective function.
    """
    def __init__(self, dimensions=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.full(dimensions, -10)
        self.upper_bound = np.full(dimensions, 10)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Objective function definition:
        # f(x) = -log(sin^2((cos(x_1) + cos(x_2))^2) - cos^2((sin(x_1) + sin(x_2))^2) + x_1)^2) + 0.01 * ((x_1 - 1)^2 + (x_2 - 1)^2)
        x = solution.variables
        term1 = np.sin((np.cos(x[0]) + np.cos(x[1]))**2)**2
        term2 = np.cos((np.sin(x[0]) + np.sin(x[1]))**2)**2
        objective_value = -np.log(np.abs(term1 - term2 + x[0])**2) + 0.01 * ((x[0] - 1)**2 + (x[1] - 1)**2)
        solution.objectives[0] = objective_value
        return solution

    def name(self) -> str:
        return "Mishra N6"


class Mishra07(FloatProblem):
    """
    Mishra 7 test objective function.
    """
    def __init__(self, dimensions=3):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.full(dimensions, -10)
        self.upper_bound = np.full(dimensions, 10)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        try:
            product_val = np.prod(solution.variables)
            n_factorial = math.factorial(self.number_of_variables())
            result = (product_val - n_factorial) ** 2
        except OverflowError:
            result = float('inf')
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Mishra N7"


class Mishra11(FloatProblem):
    """
    Mishra 11 test objective function.
    """
    def __init__(self, number_of_variables=2):
        super().__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.lower_bound = np.full(number_of_variables, -10)
        self.upper_bound = np.full(number_of_variables, 10)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Objective function definition:
        # f(x) = [1/n * sum(|x_i|) - (prod(|x_i|))^(1/n)]^2
        x = solution.variables
        sum_abs = np.sum(np.abs(x))
        prod_abs = np.prod(np.abs(x))
        term = (sum_abs / len(x) - prod_abs**(1/len(x)))**2
        solution.objectives[0] = term
        return solution

    def name(self) -> str:
        return "Mishra N11"



