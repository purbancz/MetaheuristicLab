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


class PGSHEA(Algorithm[S, R]):
    def __init__(self, problem: FloatProblem, solutions_size: int,
                 c1: float, c2: float, w: float,
                 crossover: Crossover, mutation: Mutation,
                 swap_interval: int,
                 starting_algorithm: str,
                 selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
                 solution_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__()
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.solution_evaluator = solution_evaluator
        self.problem = problem
        self.solutions_size = solutions_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.swap_limit = swap_interval * solutions_size
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.start_computing_time = time.time()
        self.current_algorithm = starting_algorithm
        self.best_global = None

        self.pso = SingleObjectivePSO(
            problem=problem, swarm_size=solutions_size, c1=c1, c2=c2, w=w,
            termination_criterion=termination_criterion
        )
        self.ga = GeneticAlgorithm(
            problem=problem, population_size=solutions_size, offspring_population_size=100,
            crossover=crossover, mutation=mutation, selection=selection,
            termination_criterion=termination_criterion
        )

    def create_initial_solutions(self) -> List[S]:
        if self.current_algorithm == 'PSO':
            solutions = self.pso.create_initial_solutions()
            self.best_global = min(solutions, key=lambda s: s.objectives[0])
        else:
            solutions = self.ga.create_initial_solutions()
            self.best_global = min(solutions, key=lambda s: s.objectives[0])

        self.ga.set_solutions(solutions)
        self.pso.set_solutions(solutions)

        return solutions

    def update_progress(self) -> None:
        self.evaluations += self.solutions_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def evaluate(self, solution_list: List[S]):
        return self.solution_evaluator.evaluate(solution_list, self.problem)

    def init_progress(self):
        self.evaluations = self.solutions_size
        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        if self.current_algorithm == 'GA':
            self.ga.step()
            if self.best_global is None or self.ga.solutions[0].objectives[0] < self.best_global.objectives[0]:
                self.best_global = self.ga.solutions[0]
        else:
            self.pso.step()
            if self.best_global is None or self.pso.best_global.objectives[0] < self.best_global.objectives[0]:
                self.best_global = self.pso.best_global

        if self.evaluations % self.swap_limit == 0:
            if self.current_algorithm == 'GA':
                self.switch_to_pso()
            else:
                self.switch_to_ga()

    def switch_to_pso(self):
        best_solutions = sorted(self.ga.solutions, key=lambda x: x.objectives[:self.solutions_size - 1])
        if self.best_global not in best_solutions:
            best_solutions[-1] = self.best_global
        self.pso.set_solutions(best_solutions)
        self.current_algorithm = 'PSO'

    def switch_to_ga(self):
        best_solutions = sorted(self.pso.solutions, key=lambda x: x.objectives[:self.solutions_size - 1])
        if self.best_global not in best_solutions:
            best_solutions[-1] = self.best_global
        self.ga.set_solutions(best_solutions)
        self.current_algorithm = 'GA'

    def result(self) -> R:
        return self.best_global

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def get_name(self) -> str:
        return "PGSHEA - Series Hybrid Evolutionary Algorithm"
