import time
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import FloatProblem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import TerminationCriterion
from jmetal.util.comparator import ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from algorithm.custom_GA import GeneticAlgorithm
from algorithm.single_objective_PSO import SingleObjectivePSO

S = TypeVar("S")
R = TypeVar("R")


class PGPHEA(Algorithm[S, R]):
    def __init__(self, problem: FloatProblem, solutions_size: int,
                 c1: float, c2: float, w: float,
                 crossover: Crossover, mutation: Mutation,
                 exchange_interval: int, exchange_number: int,
                 selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
                 solution_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__()
        self.best_global = None
        self.solution_evaluator = solution_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.problem = problem
        self.solutions_size = solutions_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.exchange_interval = exchange_interval * solutions_size
        self.exchange_number = exchange_number
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        # self.start_computing_time = time.time()

        self.pso = SingleObjectivePSO(
            problem=problem, swarm_size=solutions_size, c1=c1, c2=c2, w=w,
            termination_criterion=termination_criterion
        )
        self.ga = GeneticAlgorithm(
            problem=problem, population_size=solutions_size,
            offspring_population_size=solutions_size,
            crossover=crossover, mutation=mutation, selection=selection,
            termination_criterion=termination_criterion
        )

    def create_initial_solutions(self) -> List[S]:
        pso_solutions = self.pso.create_initial_solutions()
        ga_solutions = self.ga.create_initial_solutions()
        print(f"PSO Solutions Count: {len(pso_solutions)}, GA Solutions Count: {len(ga_solutions)}")
        self.synchronize_best_global()
        self.ga.set_solutions(ga_solutions)
        self.pso.set_solutions(pso_solutions)
        return pso_solutions + ga_solutions

    def synchronize_best_global(self):
        all_solutions = self.pso.solutions + self.ga.solutions
        self.best_global = min(all_solutions, key=lambda s: s.objectives[0])

    def step(self):
        self.pso.step()
        self.ga.step()
        self.synchronize_best_global()

        if self.evaluations % self.exchange_interval == 0:
            self.exchange_solutions()

    def exchange_solutions(self):
        pso_top = sorted(self.pso.solutions, key=lambda s: s.objectives[0])[:self.exchange_number]
        ga_top = sorted(self.ga.solutions, key=lambda s: s.objectives[0])[:self.exchange_number]

        self.pso.solutions[:self.exchange_number] = ga_top
        self.ga.solutions[:self.exchange_number] = pso_top

        self.ga.set_solutions(self.ga.solutions)
        self.pso.set_solutions(self.pso.solutions)

        print(f"Exchanged {self.exchange_number} solutions at evaluation {self.evaluations}")

    def update_progress(self) -> None:
        self.evaluations += self.solutions_size * 2

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def evaluate(self, solution_list: List[S]):
        return self.solution_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self):
        self.evaluations = self.solutions_size * 2

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def result(self) -> R:
        return self.best_global

    def get_name(self) -> str:
        return "PGPHEA - Parallel Hybrid Evolutionary Algorithm"
