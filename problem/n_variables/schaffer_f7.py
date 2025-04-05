import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

class SchafferF7(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(SchafferF7, self).__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self.number_of_variables()
        z = x
        s = np.array([z[i]**2 + z[i+1]**2 for i in range(d - 1)])
        inner = np.sum(s + s * (np.sin(50 * (s ** (1/5)))**2)) / (d - 1)
        result = inner**2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Schaffer F7"
