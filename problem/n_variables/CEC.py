import math
import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution

# =============================
# Group A: Unimodal Functions
# =============================

# shift
def generate_shift(problem: FloatProblem):
    lower_bound = np.array(problem.lower_bound)
    upper_bound = np.array(problem.upper_bound)
    dim = problem.number_of_variables()
    shift = np.random.uniform(lower_bound, upper_bound, size=dim)
    return shift

# rotation
def generate_random_orthogonal_matrix(problem: FloatProblem):
    random_matrix = np.random.randn(problem.number_of_variables(), problem.number_of_variables())
    q, _ = np.linalg.qr(random_matrix)
    return q

class RotatedHighConditionedElliptic(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(RotatedHighConditionedElliptic, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        # Rotacja – generujemy macierz ortogonalną
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = sum_{i=1}^{D} 10^(6*(i-1)/(D-1)) * z_i^2, gdzie z = M*x
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, x)
        d = self._number_of_variables
        weights = np.power(10, 6 * np.linspace(0, 1, d))
        result = np.sum(weights * (z ** 2))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Rotated High Conditioned Elliptic"

class RotatedBentCigar(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(RotatedBentCigar, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = x1^2 + 10^6 * sum_{i=2}^{D} x_i^2, z = M*x
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, x)
        result = z[0]**2 + (10**6) * np.sum(z[1:]**2)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Rotated Bent Cigar"

class RotatedDiscus(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(RotatedDiscus, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = 10^6 * x1^2 + sum_{i=2}^{D} x_i^2, z = M*x
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, x)
        result = (10**6) * z[0]**2 + np.sum(z[1:]**2)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Rotated Discus"

# =============================
# Group B: Simple Multimodal Functions
# =============================

class ShiftedRotatedRosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedRosenbrock, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = sum_{i=1}^{D-1} [100*(z_i^2 - z_{i+1})^2 + (z_i - 1)^2],
        # gdzie z = M*(x - shift)
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        result = 0.0
        for i in range(self._number_of_variables - 1):
            result += 100 * (z[i]**2 - z[i+1])**2 + (z[i] - 1)**2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Rosenbrock"

class ShiftedRotatedAckley(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedAckley, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-32.768] * number_of_variables
        self.upper_bound = [32.768] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = -20*exp(-0.2*sqrt((1/d)*sum(z_i^2))) - exp((1/d)*sum(cos(2*pi*z_i))) + 20 + e
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        term1 = -20 * math.exp(-0.2 * math.sqrt(np.sum(z**2)/d))
        term2 = -math.exp(np.sum(np.cos(2*math.pi*z))/d)
        result = term1 + term2 + 20 + math.e
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Ackley"

class ShiftedRotatedWeierstrass(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedWeierstrass, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-0.5] * number_of_variables
        self.upper_bound = [0.5] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)
        self.a = 0.5
        self.b = 3.0
        self.k_max = 20

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = sum_{i=1}^{d} sum_{k=0}^{k_max} a^k*cos(2*pi*b^k*(z_i+0.5)) - d*sum_{k=0}^{k_max} a^k*cos(2*pi*b^k*0.5)
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        sum1 = 0.0
        for i in range(d):
            for k in range(self.k_max+1):
                sum1 += (self.a**k) * math.cos(2*math.pi*(self.b**k)*(z[i]+0.5))
        sum2 = 0.0
        for k in range(self.k_max+1):
            sum2 += (self.a**k) * math.cos(2*math.pi*(self.b**k)*0.5)
        result = sum1 - d * sum2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Weierstrass"

class ShiftedRastrigin(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRastrigin, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.12] * number_of_variables
        self.upper_bound = [5.12] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound        
        self.shift = generate_shift(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = 10*d + sum_{i=1}^{d}[z_i^2 - 10*cos(2*pi*z_i)]
        x = np.array(solution.variables)
        z = x - self.shift
        d = self._number_of_variables
        result = 10 * d + np.sum(z**2 - 10*np.cos(2*math.pi*z))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted Rastrigin"

class ShiftedRotatedRastrigin(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedRastrigin, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.12] * number_of_variables
        self.upper_bound = [5.12] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = 10*D + sum_{i=1}^{D}[z_i^2 - 10*cos(2*pi*z_i)], z = M*(x-shift)
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        result = 10 * d + np.sum(z**2 - 10*np.cos(2*math.pi*z))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Rastrigin"

class ShiftedSchwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedSchwefel, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-500] * number_of_variables
        self.upper_bound = [500] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = 418.9829*d - sum_{i=1}^{d}[z_i*sin(sqrt(|z_i|))], z = x - shift
        x = np.array(solution.variables)
        z = x - self.shift
        d = self._number_of_variables
        result = 418.9829 * d - np.sum(z * np.sin(np.sqrt(np.abs(z))))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted Schwefel"

class ShiftedRotatedSchwefel(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedSchwefel, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-500] * number_of_variables
        self.upper_bound = [500] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # f(x) = 418.9829*d - sum_{i=1}^{d}[z_i*sin(sqrt(|z_i|))], z = M*(x-shift)
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        result = 418.9829 * d - np.sum(z * np.sin(np.sqrt(np.abs(z))))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Schwefel"

class ShiftedRotatedKatsuura(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedKatsuura, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        prod = 1.0
        for i in range(d):
            temp = 0.0
            for j in range(1, d+1):
                temp += abs(2**j * z[i] - round(2**j * z[i])) / (2**j)
            prod *= (1 + (i+1) * (temp ** (2.0 / d)))
        result = (1.2 / (d ** 2.2)) * prod - (1.2 / (d ** 2.2))
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Expanded Katsuura"

class ShiftedRotatedHappyCat(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedHappyCat, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # HappyCat function (przybliżenie):
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        sum_z2 = np.sum(z**2)
        sum_z = np.sum(z)
        result = abs(sum_z2 - d)**0.25 + (0.5 * sum_z2 + sum_z)/d + 0.5
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated HappyCat"

class ShiftedRotatedHGBat(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedHGBat, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-5.0] * number_of_variables
        self.upper_bound = [5.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        d = self._number_of_variables
        S2 = np.sum(z**2)
        S1 = np.sum(z)
        val1 = math.sqrt(abs(S2**2 - S1**2))
        val2 = (0.5 * S2 + S1) / d
        result = val1 + val2 + 0.5
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated HGBat"

class ShiftedRotatedExpandedGriewankPlusRosenbrock(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedExpandedGriewankPlusRosenbrock, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Przybliżona wersja: f(x) = sum_{i=1}^{D-1}[100*( (z_i^2 - z_{i+1} + 1)^2 ) + z_i^2]
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        result = 0.0
        for i in range(self._number_of_variables - 1):
            result += 100 * ((z[i]**2 - z[i+1] + 1)**2) + (z[i]**2)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Expanded Griewank plus Rosenbrock"

class ShiftedRotatedSchafferF7(FloatProblem):
    def __init__(self, number_of_variables: int = 2):
        super(ShiftedRotatedSchafferF7, self).__init__()
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = np.array(solution.variables)
        d = self.number_of_variables()
        z = np.dot(self.rotation_matrix, (x - self.shift))
        z = 10 * z
        s = np.array([z[i]**2 + z[i+1]**2 for i in range(d - 1)])
        inner = np.sum(s + s * (np.sin(50 * (s ** (1/5)))**2)) / (d - 1)
        result = inner**2
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Schaffer F7"

class ShiftedRotatedExpandedScafferF6(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(ShiftedRotatedExpandedScafferF6, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.shift = generate_shift(self)
        self.rotation_matrix = generate_random_orthogonal_matrix(self)

    def number_of_variables(self) -> int:
        return self._number_of_variables

    def number_of_objectives(self) -> int:
        return 1

    def number_of_constraints(self) -> int:
        return 0

    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Przybliżona wersja Expanded Scaffer’s F6: f(x) = sum_{i=1}^{D-1} sqrt(z_i^2+z_{i+1}^2)
        x = np.array(solution.variables)
        z = np.dot(self.rotation_matrix, (x - self.shift))
        result = 0.0
        for i in range(self._number_of_variables - 1):
            result += math.sqrt(z[i]**2 + z[i+1]**2)
        solution.objectives[0] = result
        return solution

    def name(self) -> str:
        return "Shifted and Rotated Expanded Scaffer’s F6"

# =============================
# Group C: Hybrid Functions
# =============================

class HybridFunction1(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(HybridFunction1, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.p = [0.3, 0.3, 0.4]  # podział procentowy
        self.shift = generate_shift(self)
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Hybrid Function 1 łączy:
        # - Modified Schwefel’s Function (proxy: ShiftedSchwefel)
        # - Rastrigin’s Function (ShiftedRastrigin)
        # - High Conditioned Elliptic Function (RotatedHighConditionedElliptic)
        x = np.array(solution.variables) - self.shift
        d = self._number_of_variables
        idx1 = int(self.p[0] * d)
        idx2 = idx1 + int(self.p[1] * d)
        part1 = x[:idx1]
        part2 = x[idx1:idx2]
        part3 = x[idx2:]
        val1 = 418.9829 * len(part1) - np.sum(part1 * np.sin(np.sqrt(np.abs(part1))))
        val2 = 10 * len(part2) + np.sum(part2**2 - 10 * np.cos(2*math.pi*part2))
        weights = np.power(10, 6 * np.linspace(0, 1, len(part3)))
        val3 = np.sum(weights * (part3**2))
        result = val1 + val2 + val3
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Hybrid Function 1"

class HybridFunction2(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(HybridFunction2, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.p = [0.3, 0.3, 0.4]
        self.shift = generate_shift(self)
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Hybrid Function 2 łączy:
        # - Bent Cigar (RotatedBentCigar)
        # - HGBat (ShiftedRotatedHGBat)
        # - Rastrigin (ShiftedRastrigin)
        x = np.array(solution.variables) - self.shift
        d = self._number_of_variables
        idx1 = int(self.p[0] * d)
        idx2 = idx1 + int(self.p[1] * d)
        part1 = x[:idx1]
        part2 = x[idx1:idx2]
        part3 = x[idx2:]
        val1 = part1[0]**2 + (10**6)*np.sum(part1[1:]**2)
        val2 = abs(np.sum(part2**2) - len(part2))**(1/8) + (0.5*np.sum(part2**2)+np.sum(part2))/len(part2) + 0.5
        val3 = 10 * len(part3) + np.sum(part3**2 - 10*np.cos(2*math.pi*part3))
        result = val1 + val2 + val3
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Hybrid Function 2"

class HybridFunction3(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(HybridFunction3, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.p = [0.2, 0.2, 0.3, 0.3]
        self.shift = generate_shift(self)
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Hybrid Function 3 łączy:
        # - Griewank (przykładowo: standardowa Griewank)
        # - Weierstrass (ShiftedRotatedWeierstrass)
        # - Rosenbrock (ShiftedRotatedRosenbrock)
        # - Scaffer’s F6 (uproszczona wersja)
        x = np.array(solution.variables) - self.shift
        d = self._number_of_variables
        idx1 = int(self.p[0] * d)
        idx2 = idx1 + int(self.p[1] * d)
        idx3 = idx2 + int(self.p[2] * d)
        part1 = x[:idx1]
        part2 = x[idx1:idx2]
        part3 = x[idx2:idx3]
        part4 = x[idx3:]
        # Uproszczone funkcje:
        val1 = np.sum(part1**2)/4000 - np.prod(np.cos(part1/np.sqrt(np.arange(1, len(part1)+1)))) + 1
        val2 = 0.0
        for i in range(len(part2)):
            for k in range(21):
                val2 += (0.5**k)*math.cos(2*math.pi*(3**k)*(part2[i]+0.5))
        val3 = 0.0
        for i in range(len(part3)-1):
            val3 += 100*(part3[i]**2 - part3[i+1])**2 + (part3[i]-1)**2
        val4 = 0.0
        for i in range(len(part4)-1):
            val4 += math.sqrt(part4[i]**2+part4[i+1]**2)
        result = val1 + val2 + val3 + val4
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Hybrid Function 3"

class HybridFunction4(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(HybridFunction4, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.p = [0.2, 0.2, 0.3, 0.3]
        self.shift = generate_shift(self)
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Hybrid Function 4 łączy:
        # - HGBat (ShiftedRotatedHGBat)
        # - Discus (RotatedDiscus)
        # - Expanded Griewank+Rosenbrock (ShiftedRotatedExpandedGriewankPlusRosenbrock)
        # - Rastrigin (ShiftedRastrigin)
        x = np.array(solution.variables) - self.shift
        d = self._number_of_variables
        idx1 = int(self.p[0] * d)
        idx2 = idx1 + int(self.p[1] * d)
        idx3 = idx2 + int(self.p[2] * d)
        part1 = x[:idx1]
        part2 = x[idx1:idx2]
        part3 = x[idx2:idx3]
        part4 = x[idx3:]
        val1 = abs(np.sum(part1**2)-len(part1))**(1/8) + (0.5*np.sum(part1**2)+np.sum(part1))/len(part1) + 0.5
        val2 = (10**6)*(part2[0]**2) + np.sum(part2[1:]**2)
        val3 = 0.0
        for i in range(len(part3)-1):
            val3 += 100*(part3[i]**2 - part3[i+1] + 1)**2 + part3[i]**2
        val4 = 10 * len(part4) + np.sum(part4**2 - 10*np.cos(2*math.pi*part4))
        result = val1 + val2 + val3 + val4
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Hybrid Function 4"

class HybridFunction5(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(HybridFunction5, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.p = [0.1, 0.2, 0.2, 0.2, 0.3]
        self.shift = generate_shift(self)
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Hybrid Function 5 łączy:
        # - Scaffer’s F6 (uproszczona)
        # - HGBat (ShiftedRotatedHGBat)
        # - Rosenbrock (ShiftedRotatedRosenbrock)
        # - Modified Schwefel (ShiftedSchwefel)
        # - High Conditioned Elliptic (RotatedHighConditionedElliptic)
        x = np.array(solution.variables) - self.shift
        d = self._number_of_variables
        idx1 = int(self.p[0] * d)
        idx2 = idx1 + int(self.p[1] * d)
        idx3 = idx2 + int(self.p[2] * d)
        idx4 = idx3 + int(self.p[3] * d)
        part1 = x[:idx1]
        part2 = x[idx1:idx2]
        part3 = x[idx2:idx3]
        part4 = x[idx3:idx4]
        part5 = x[idx4:]
        val1 = 0.0
        for i in range(len(part1)-1):
            val1 += math.sqrt(part1[i]**2 + part1[i+1]**2)
        val2 = abs(np.sum(part2**2)-len(part2))**(1/8) + (0.5*np.sum(part2**2)+np.sum(part2))/len(part2) + 0.5
        val3 = 0.0
        for i in range(len(part3)-1):
            val3 += 100*(part3[i]**2 - part3[i+1])**2 + (part3[i]-1)**2
        val4 = 418.9829*len(part4) - np.sum(part4*np.sin(np.sqrt(np.abs(part4))))
        weights = np.power(10, 6 * np.linspace(0, 1, len(part5)))
        val5 = np.sum(weights * (part5**2))
        result = val1 + val2 + val3 + val4 + val5
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Hybrid Function 5"

class HybridFunction6(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(HybridFunction6, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.p = [0.1, 0.2, 0.2, 0.2, 0.3]
        self.shift = generate_shift(self)
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        # Hybrid Function 6 łączy:
        # - Katsuura (ShiftedRotatedKatsuura)
        # - HappyCat (ShiftedRotatedHappyCat)
        # - Expanded Griewank+Rosenbrock (ShiftedRotatedExpandedGriewankPlusRosenbrock)
        # - Modified Schwefel (ShiftedSchwefel)
        # - Ackley (ShiftedRotatedAckley)
        x = np.array(solution.variables) - self.shift
        d = self._number_of_variables
        idx1 = int(self.p[0] * d)
        idx2 = idx1 + int(self.p[1] * d)
        idx3 = idx2 + int(self.p[2] * d)
        idx4 = idx3 + int(self.p[3] * d)
        part1 = x[:idx1]  # Katsuura
        part2 = x[idx1:idx2]  # HappyCat
        part3 = x[idx2:idx3]  # Expanded Griewank+Rosenbrock
        part4 = x[idx3:idx4]  # Modified Schwefel
        part5 = x[idx4:]  # Ackley
        # Uproszczone oceny:
        val1 = 0.0
        for i in range(len(part1)):
            prod = 1.0
            for j in range(len(part1)):
                prod *= (1 + (j+1)* (abs(part1[i]) ** (1+4*(j+1)/len(part1))))
            val1 += prod
        val1 = (1.2 / (len(part1) ** 2.2)) * val1 - (1.2 / (len(part1) ** 2.2))
        val2 = abs(np.sum(part2**2)-len(part2))**0.25 + (0.5*np.sum(part2**2)+np.sum(part2))/len(part2) + 0.5
        val3 = 0.0
        for i in range(len(part3)-1):
            val3 += 100*(part3[i]**2 - part3[i+1])**2 + (part3[i]-1)**2
        val4 = 418.9829*len(part4) - np.sum(part4*np.sin(np.sqrt(np.abs(part4))))
        val5 = -20*math.exp(-0.2*math.sqrt(np.sum(part5**2)/len(part5))) - math.exp(np.sum(np.cos(2*math.pi*part5))/len(part5)) + 20 + math.e
        result = val1 + val2 + val3 + val4 + val5
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Hybrid Function 6"

# =============================
# Group D: Composition Functions
# =============================

def clone_solution(problem: FloatProblem, variables: list) -> FloatSolution:
    """ Tworzy nowy obiekt FloatSolution z takimi samymi parametrami, co problem,
        ale z zadanym wektorem variables. """
    new_sol = FloatSolution(
        lower_bound=problem.lower_bound,
        upper_bound=problem.upper_bound,
        number_of_objectives=problem.number_of_objectives(),
        number_of_constraints=problem.number_of_constraints()
    )
    new_sol.variables = list(variables)
    return new_sol


class CompositionFunction1(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction1, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        # Parametry
        self.N = 5
        self.sigma = [10, 20, 30, 40, 50]
        self.lambd = [1, 1e-6, 1e-26, 1e-6, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        # Wybieramy 5 podstawowych funkcji jako proxy
        self.basic_functions = [
            ShiftedRotatedRosenbrock(),
            RotatedHighConditionedElliptic(),
            RotatedBentCigar(),
            RotatedDiscus(),
            ShiftedRastrigin()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))  # uproszczenie: odległość od zera
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]
            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 1"

class CompositionFunction2(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction2, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 3
        self.sigma = [20, 20, 20]
        self.lambd = [1, 1, 1]
        self.bias = [0, 100, 200]
        self.basic_functions = [
            ShiftedSchwefel(),
            ShiftedRotatedRastrigin(),
            ShiftedRotatedHGBat()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]
            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 2"

class CompositionFunction3(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction3, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 3
        self.sigma = [10, 30, 50]
        self.lambd = [0.25, 1, 1e-7]
        self.bias = [0, 100, 200]
        self.basic_functions = [
            ShiftedRotatedSchwefel(),
            ShiftedRotatedRastrigin(),
            RotatedHighConditionedElliptic()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]

            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 3"

class CompositionFunction4(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction4, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 5
        self.sigma = [10, 10, 10, 10, 10]
        self.lambd = [0.25, 1, 1e-7, 2.5, 10]
        self.bias = [0, 100, 200, 300, 400]
        self.basic_functions = [
            ShiftedRotatedSchwefel(),
            ShiftedRotatedHappyCat(),
            RotatedHighConditionedElliptic(),
            ShiftedRotatedWeierstrass(),
            ShiftedRastrigin()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]

            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 4"

class CompositionFunction5(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction5, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 5
        self.sigma = [10, 10, 10, 20, 20]
        self.lambd = [10, 10, 2.5, 25, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.basic_functions = [
            ShiftedRotatedHGBat(),
            ShiftedRotatedRastrigin(),
            ShiftedRotatedSchwefel(),
            ShiftedRotatedWeierstrass(),
            RotatedHighConditionedElliptic()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]

            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 5"

class CompositionFunction6(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction6, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 5
        self.sigma = [10, 20, 30, 40, 50]
        self.lambd = [2.5, 10, 2.5, 5e-4, 1e-6]
        self.bias = [0, 100, 200, 300, 400]
        self.basic_functions = [
            ShiftedRotatedExpandedGriewankPlusRosenbrock(),
            ShiftedRotatedHappyCat(),
            ShiftedRotatedSchwefel(),
            ShiftedRotatedExpandedScafferF6(),
            RotatedHighConditionedElliptic()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]

            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 6"

class CompositionFunction7(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction7, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 3
        self.sigma = [10, 30, 50]
        self.lambd = [1, 1, 1]
        self.bias = [0, 100, 200]
        self.basic_functions = [
            HybridFunction1(),
            HybridFunction2(),
            HybridFunction3()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]

            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 7"

class CompositionFunction8(FloatProblem):
    def __init__(self, number_of_variables: int = 30):
        super(CompositionFunction8, self).__init__()
        self._number_of_variables = number_of_variables
        self.obj_directions = [self.MINIMIZE]
        self.obj_labels = ['f(x)']
        self.lower_bound = [-100.0] * number_of_variables
        self.upper_bound = [100.0] * number_of_variables
        FloatSolution.lower_bound = self.lower_bound
        FloatSolution.upper_bound = self.upper_bound
        self.N = 3
        self.sigma = [10, 30, 50]
        self.lambd = [1, 1, 1]
        self.bias = [0, 100, 200]
        self.basic_functions = [
            HybridFunction4(),
            HybridFunction5(),
            HybridFunction6()
        ]
    def number_of_variables(self) -> int:
        return self._number_of_variables
    def number_of_objectives(self) -> int:
        return 1
    def number_of_constraints(self) -> int:
        return 0
    def evaluate(self, solution: FloatSolution) -> FloatSolution:
        x = solution.variables
        weights = []
        for i in range(self.N):
            diff = np.linalg.norm(np.array(x))
            w = math.exp(-diff / (2*(self.sigma[i]**2)))
            weights.append(w)
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        f_vals = []
        for i in range(self.N):
            sol_copy = clone_solution(self.basic_functions[i], x)
            f_i = self.basic_functions[i].evaluate(sol_copy).objectives[0]

            f_vals.append(f_i)
        f_vals = np.array(f_vals)
        result = np.sum(weights * (f_vals / np.array(self.lambd) + np.array(self.bias)))
        solution.objectives[0] = result
        return solution
    def name(self) -> str:
        return "Composition Function 8"
