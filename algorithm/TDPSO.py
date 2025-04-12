import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.single_objective_PSO import SingleObjectivePSO


class TDPSO(SingleObjectivePSO):
    """
    Thermodynamic PSO (TDPSO)
    Concept: Applies thermodynamic principles to swarm dynamics
    Key Features:
      - Temperature Metric: Controls exploration/exploitation balance
      - Entropy Monitoring: Triggers phase transitions (solid/liquid/gas)
      - Brownian Motion: Adds controlled random perturbations
    """
    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion, temperature = 1.0, cooling_rate = 0.99):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.temperature = temperature
        self.cooling_rate = cooling_rate

    def swarm_diversity(self) -> float:
        positions = np.array([p.variables for p in self.solutions])
        centroid = np.mean(positions, axis=0)
        return np.mean(np.linalg.norm(positions - centroid, axis=1))

    def calculate_temperature(self):
        diversity = self.swarm_diversity()
        self.temperature = max(0.1, self.temperature * self.cooling_rate)
        if diversity < 0.05:
            self.temperature += 0.2  # Prevent freezing

    def brownian_motion(self):
        for particle in self.solutions:
            variables = np.array(particle.variables, dtype=float)
            perturbation = np.random.normal(0, self.temperature, size=variables.shape)
            new_variables = variables + perturbation
            new_variables = np.clip(new_variables, self.problem.lower_bound, self.problem.upper_bound)
            particle.variables = new_variables.tolist()

    def step(self):
        self.calculate_temperature()
        self.brownian_motion()
        super().step()


import numpy as np
import random
from copy import deepcopy
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion
from algorithm.single_objective_PSO import SingleObjectivePSO


class LangevinPSO(SingleObjectivePSO):
    """
    Langevin PSO: A physically grounded PSO variant using Langevin dynamics.

    In this algorithm, each particle is updated according to a Langevin-like equation:

      v_new = v_old + (-grad - gamma*v_old) * dt + sqrt(2*gamma*T*dt) * N(0, I)
      x_new = x_old + v_new * dt

    where grad is obtained by numerically estimating the gradient of the objective,
    gamma is a friction coefficient, T is the current temperature, dt is the time step,
    and N(0, I) is a standard normal noise vector.

    The temperature is cooled gradually over time, with a minimum value enforced.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion,
                 dt: float = 0.01,
                 gamma: float = 0.1,
                 T0: float = 1.0,
                 cooling_rate: float = 0.99,
                 T_min: float = 0.1):
        # Note: The parameters c1, c2, and w are maintained for compatibility
        # with the base PSO even though they are not directly used in this variant.
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.dt = dt
        self.gamma = gamma
        self.temperature = T0
        self.cooling_rate = cooling_rate
        self.T_min = T_min

    def _numerical_gradient(self, particle: FloatSolution, epsilon: float = 1e-6) -> np.ndarray:
        """Finite difference approximation of the gradient of the objective."""
        grad = np.zeros(self.problem.number_of_variables())
        x = np.array(particle.variables, dtype=float)
        for i in range(self.problem.number_of_variables()):
            x_plus = x.copy()
            x_plus[i] += epsilon
            sol_plus = self.problem.create_solution()
            sol_plus.variables = x_plus.tolist()
            self.problem.evaluate(sol_plus)

            x_minus = x.copy()
            x_minus[i] -= epsilon
            sol_minus = self.problem.create_solution()
            sol_minus.variables = x_minus.tolist()
            self.problem.evaluate(sol_minus)

            grad[i] = (sol_plus.objectives[0] - sol_minus.objectives[0]) / (2 * epsilon)
        return grad  # Note: The force is -grad (hence our update adds -grad)

    def step(self):
        # Update temperature using a cooling schedule.
        self.temperature = max(self.T_min, self.temperature * self.cooling_rate)

        for particle in self.solutions:
            # Compute gradient (approximation of force) at the current position.
            grad = self._numerical_gradient(particle)

            # Retrieve current velocity.
            v = np.array(particle.attributes['velocity'], dtype=float)

            # Langevin update: include force, friction, and a stochastic (thermal) term.
            noise = np.random.normal(0, 1, size=v.shape)
            v_new = v + (-grad - self.gamma * v) * self.dt \
                    + np.sqrt(2 * self.gamma * self.temperature * self.dt) * noise
            particle.attributes['velocity'] = v_new.tolist()

            # Update position based on the new velocity.
            x = np.array(particle.variables, dtype=float)
            x_new = x + v_new * self.dt
            # Ensure the new position is within the search bounds.
            lower_bound = np.array(self.problem.lower_bound, dtype=float)
            upper_bound = np.array(self.problem.upper_bound, dtype=float)
            x_new = np.clip(x_new, lower_bound, upper_bound)
            particle.variables = x_new.tolist()

        # Evaluate the updated swarm and update best solutions.
        self.solutions = self.evaluate(self.solutions)
        self.update_global_best(self.solutions)
        self.update_particle_best(self.solutions)
