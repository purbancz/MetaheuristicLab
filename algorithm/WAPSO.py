import random
import numpy as np
from typing import List, TypeVar
from jmetal.core.problem import FloatProblem
from jmetal.core.solution import FloatSolution
from jmetal.util.termination_criterion import TerminationCriterion
from algorithm.single_objective_PSO import SingleObjectivePSO

S = TypeVar("S")
R = TypeVar("R")


class WorstAwarePSO(SingleObjectivePSO):
    """
    Base class for PSO variants that maintain and update worst solution information.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.global_worst = None

    def create_initial_solutions(self) -> List[S]:
        solutions = super().create_initial_solutions()
        for sol in solutions:
            sol.attributes['worst_position'] = sol.variables.copy()
            sol.attributes['worst_objective'] = sol.objectives[0]
        self.global_worst = max(solutions, key=lambda s: s.objectives[0])
        return solutions

    @staticmethod
    def update_particle_worst(swarm: List[S]) -> None:
        for particle in swarm:
            if particle.objectives[0] > particle.attributes['worst_objective']:
                particle.attributes['worst_objective'] = particle.objectives[0]
                particle.attributes['worst_position'] = particle.variables.copy()

    def update_global_worst(self, swarm: List[S]) -> None:
        self.global_worst = max(swarm, key=lambda s: s.objectives[0])

class ReverseLearningPSO(WorstAwarePSO):
    """
    Reverse Learning PSO:
    Instead of following personal/global best, particles avoid their personal and global worst
    (particles avoid their personal and global worst by moving away from them).
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 b1: float,
                 b2: float,
                 w: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, b1, b2, w, termination_criterion)

    def update_velocity(self, swarm: List[S]) -> None:
        worst_global = np.array(self.global_worst.variables)
        for particle in swarm:
            r1, r2 = random.random(), random.random()
            current = np.array(particle.variables)
            worst_personal = np.array(particle.attributes['worst_position'])
            repulsion = (self.c1 * r1 * (current - worst_personal) +
                         self.c2 * r2 * (current - worst_global))
            new_velocity = (self.w * np.array(particle.attributes['velocity']) +
                            repulsion)
            particle.attributes['velocity'] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "ReverseLearningPSO"

class ReverseLearningGlobalAttractorPSO(WorstAwarePSO):
    """
    Reverse Learning PSO with global attraction.
    Particles avoid their personal and global worst while also being attracted to the global best.
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 a: float,
                 b1: float,
                 b2: float,
                 w: float,
                 termination_criterion: TerminationCriterion):
        super().__init__(problem, swarm_size, b1, b2, w, termination_criterion)
        self.a = a

    def update_velocity(self, swarm: List[S]) -> None:
        best_global = np.array(self.best_global.variables)
        worst_global = np.array(self.global_worst.variables)
        for particle in swarm:
            r1, r2, r3 = random.random(), random.random(), random.random()
            current = np.array(particle.variables)
            worst_personal = np.array(particle.attributes['worst_position'])
            attraction = self.a * r3 * (best_global - current)
            repulsion = (self.c1 * r1 * (current - worst_personal) +
                         self.c2 * r2 * (current - worst_global))
            new_velocity = (self.w * np.array(particle.attributes['velocity']) +
                            repulsion + attraction)
            particle.attributes['velocity'] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "ReverseLearningGlobalAttractorPSO"

class ReverseLearningPersonalAttractorPSO(ReverseLearningGlobalAttractorPSO):
    """
    Reverse Learning PSO with personal (local) attraction.
    Particles avoid their personal and global worst while being attracted to their personal best.
    """

    def update_velocity(self, swarm: List[S]) -> None:
        worst_global = np.array(self.global_worst.variables)
        for particle in swarm:
            r1, r2, r3 = random.random(), random.random(), random.random()
            current = np.array(particle.variables)
            best_personal = np.array(particle.attributes['best_position'])
            worst_personal = np.array(particle.attributes['worst_position'])
            attraction = self.a * r3 * (best_personal - current)
            repulsion = (self.c1 * r1 * (current - worst_personal) +
                         self.c2 * r2 * (current - worst_global))
            new_velocity = (self.w * np.array(particle.attributes['velocity']) +
                            repulsion + attraction)
            particle.attributes['velocity'] = new_velocity.tolist()


    def get_name(self) -> str:
        return "ReverseLearningPersonalAttractorPSO"


class CombinedLearningPSO(WorstAwarePSO):
    """
    Combined Learning PSO:
    Particles are attracted to their personal and global bests while avoiding their personal and global worsts.
    Velocity update rule:
      v = w*v +
          b1*r1*(personal_best - current) +
          b2*r2*(global_best - current) +
          b1*r1*(current - personal_worst) +
          b2*r2*(current - global_worst)
    """

    def __init__(self,
                 problem: FloatProblem,
                 swarm_size: int,
                 c1: float,
                 c2: float,
                 w: float,
                 termination_criterion: TerminationCriterion,
                 b1: float,  # Coefficient for repulsion from personal worst
                 b2: float  # Coefficient for repulsion from global worst
                 ):
        super().__init__(problem, swarm_size, c1, c2, w, termination_criterion)
        self.b1 = b1
        self.b2 = b2

    def update_velocity(self, swarm: List[S]) -> None:
        self.update_global_worst(swarm)
        self.update_particle_worst(swarm)
        for particle in swarm:
            r1 = random.random()
            r2 = random.random()
            r3 = random.random()
            r4 = random.random()
            current = np.array(particle.variables)
            best_personal = np.array(particle.attributes['best_position'])
            best_global = np.array(self.best_global.variables)
            worst_personal = np.array(particle.attributes['worst_position'])
            worst_global = np.array(self.global_worst.variables)
            new_velocity = (self.w * np.array(particle.attributes['velocity']) +
                            self.c1 * r1 * (best_personal - current) +
                            self.c2 * r2 * (best_global - current) +
                            self.b1 * r3 * (current - worst_personal) +
                            self.b2 * r4 * (current - worst_global))
            particle.attributes['velocity'] = new_velocity.tolist()

    def update_particle_best(self, swarm: List[S]) -> None:
        super().update_particle_best(swarm)
        self.update_particle_worst(swarm)

    def update_global_best(self, swarm: List[FloatSolution]) -> None:
        super().update_global_best(swarm)
        self.update_global_worst(swarm)

    def get_name(self) -> str:
        return "CombinedLearningPSO"

