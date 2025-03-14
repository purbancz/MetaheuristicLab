import numpy as np
from jmetal.core.problem import FloatProblem
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.single_objective_PSO import SingleObjectivePSO


class Memory:
    def __init__(self):
        self.memory = []

    def store(self, variables):
        self.memory.append(variables)

    def retrieve(self, solutions):
        if self.memory:
            sample = self.memory[np.random.randint(len(self.memory))]
            for particle in solutions:
                if np.random.rand() < 0.1:
                    particle.variables = (0.9 * np.array(particle.variables) + 0.1 * np.array(sample)).tolist()

class NPSO(SingleObjectivePSO):
    """
    Neural PSO (NPSO)
    Concept: Implements neural dynamics in particle interactions
    Key Features:
      - Particle "Neurons": Communicate via simulated spike-timing dynamics
      - Plasticity Rules: Hebbian learning adjusts social/cognitive weights
      - Ensemble Memory: Shared memory remembers promising regions
    """
    def __init__(self, problem: FloatProblem, swarm_size: int, c1: float, c2: float, w: float,
                 termination_criterion: TerminationCriterion,
                 spike_threshold = 0.96,
                 max_c1: float = 7,
                 max_c2: float = 7):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.lstm = Memory()
        self.spike_threshold = spike_threshold
        self.max_c1 = max_c1
        self.max_c2 = max_c2

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def calculate_spikes(self):
        for particle in self.solutions:
            fitness = particle.objectives[0]
            firing_rate = self.sigmoid(-fitness)
            particle.firing_rate = firing_rate
            if firing_rate > self.spike_threshold:
                self.lstm.store(particle.variables)

    def hebbian_update(self):
        for particle in self.solutions:

            if particle.objectives[0] < particle.attributes.get('best_objective', float('inf')):
                particle.attributes['improved'] = True
                self.c1 = min(self.max_c1, self.c1 * 1.1)
            else:
                particle.attributes['improved'] = False
                self.c2 = min(self.max_c2, self.c2 * 1.1)

    def step(self):
        self.calculate_spikes()
        self.hebbian_update()
        super().step()
        self.lstm.retrieve(self.solutions)