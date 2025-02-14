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