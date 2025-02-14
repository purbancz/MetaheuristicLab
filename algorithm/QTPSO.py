import numpy as np
import random
from copy import deepcopy
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion
from algorithm.single_objective_PSO import SingleObjectivePSO

# =========================
# QTPSO: Quantum-Tunneling PSO
# =========================

class QTPSO(SingleObjectivePSO):
    """
    Quantum-Tunneling PSO (QT-PSO)
    Concept: Combines quantum mechanics with chaotic exploration
    Key Features:
      - Quantum Potential Wells: Particles occasionally tunnel through fitness barriers
      - Chaotic Local Search: Uses logistic map for intensive local exploitation
      - Energy-State Particles: Particles switch between "ground" (exploitation) and "excited" (exploration) states
    """
    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion, quantum_prob=0.1, chaos_strength=0.05):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.quantum_prob = quantum_prob  # Probability of quantum tunneling
        self.chaos_strength = chaos_strength

    def create_initial_solutions(self) -> [FloatSolution]:
        solutions = super().create_initial_solutions()
        # Initialize chaos seed for each particle
        for particle in solutions:
            particle.chaos_seed = np.random.rand()
        return solutions

    def quantum_tunnel(self, particle: FloatSolution):
        if np.random.rand() < self.quantum_prob:
            # Tunnel to a random position within search space
            particle.variables = np.random.uniform(
                self.problem.lower_bound,
                self.problem.upper_bound
            ).tolist()

    def chaotic_search(self, particle: FloatSolution):
        # Apply chaotic perturbation using logistic map
        variables = np.array(particle.variables, dtype=float)
        chaos = 4 * particle.chaos_seed * (1 - particle.chaos_seed)
        variables = variables + self.chaos_strength * (chaos - 0.5)
        particle.variables = variables.tolist()
        particle.chaos_seed = chaos

    def step(self):
        # Apply quantum and chaotic operators before the regular PSO step.
        for particle in self.solutions:
            self.quantum_tunnel(particle)
            self.chaotic_search(particle)
        super().step()