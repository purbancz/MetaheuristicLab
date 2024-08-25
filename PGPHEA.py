from jmetal.algorithm.singleobjective import GeneticAlgorithm
from jmetal.core.algorithm import Algorithm
from jmetal.core.problem import FloatProblem
from jmetal.operator import PolynomialMutation, SBXCrossover, BinaryTournamentSelection
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.generator import Generator
from jmetal.util.evaluator import Evaluator
from typing import TypeVar, List

from SingleObjectivePSO import SingleObjectivePSO

S = TypeVar("S")  # Solution type
R = TypeVar("R")  # Result type


class PGPHEA(Algorithm[S, R]):
    def __init__(self, problem: FloatProblem, population_size: int, switch_interval: int, max_iterations: int,
                 mutation: PolynomialMutation, crossover: SBXCrossover, selection: BinaryTournamentSelection,
                 termination_criterion: TerminationCriterion,
                 population_generator: Generator, population_evaluator: Evaluator):
        super().__init__()
        self.problem = problem
        self.population_size = population_size
        self.switch_interval = switch_interval
        self.max_iterations = max_iterations
        self.iterations = 0

        # Initialize GA and PSO
        self.ga = GeneticAlgorithm(problem, population_size, population_size,
                                   mutation, crossover, selection, termination_criterion,
                                   population_generator, population_evaluator)
        self.pso = SingleObjectivePSO(problem, population_size, max_iterations, 2.05, 2.05, 0.7)

        self.ga_population = []
        self.pso_swarm = []

    def create_initial_solutions(self) -> List[S]:
        self.ga_population = self.ga.create_initial_solutions()
        self.pso_swarm = self.pso.create_initial_solutions()
        return self.ga_population + self.pso_swarm  # Combined list, mainly for interface compliance

    def evaluate(self, solution_list: List[S]) -> List[S]:
        self.ga.evaluate(self.ga_population)
        self.pso.evaluate(self.pso_swarm)
        return solution_list  # This method might need adjustments based on your specific evaluation logic

    def stopping_condition_is_met(self) -> bool:
        return self.iterations >= self.max_iterations

    def update_progress(self) -> None:
        self.iterations += 1
        if self.iterations % self.switch_interval == 0:
            self.perform_exchange()

    def observable_data(self) -> dict:
        return {
            "GA_BEST": min(self.ga_population, key=lambda s: s.objectives[0]),
            "PSO_BEST": min(self.pso_swarm, key=lambda s: s.objectives[0])
        }

    def result(self) -> R:
        ga_best = min(self.ga_population, key=lambda s: s.objectives[0])
        pso_best = min(self.pso_swarm, key=lambda s: s.objectives[0])
        return ga_best if ga_best.objectives[0] < pso_best.objectives[0] else pso_best

    def perform_exchange(self):
        # Exchange logic to be defined based on your application needs
        pass
