import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class Stochastic(FloatProblem):
    def __init__(self, number_of_variables: int, seed: int | None = None,):
        super(Stochastic, self).__init__()
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ["f(x)"]
        self.rng = np.random.default_rng(seed)
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def set_seed(self, seed: int) -> None:
        self.rng = np.random.default_rng(seed)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self.number_of_variables()
        epsilons = self.rng.uniform(0, 1, size=d)
        # f(x) = sum_{i=1}^d ε_i * | x_i - 1/d |
        result = np.sum(epsilons * np.abs(x - (1.0 / d)))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Stochastic"
