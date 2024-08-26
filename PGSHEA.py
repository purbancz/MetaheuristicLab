import time
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.algorithm import Algorithm
from jmetal.core.operator import Crossover, Mutation, Selection
from jmetal.core.problem import FloatProblem
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.termination_criterion import TerminationCriterion
from custom_genetic_algorithm import GeneticAlgorithm
from jmetal.util.comparator import ObjectiveComparator
from SingleObjectivePSO import SingleObjectivePSO

S = TypeVar("S")
R = TypeVar("R")


class PGSHEA(Algorithm[S, R]):
    def __init__(self, problem: FloatProblem, solutions_size: int,
                 c1: float, c2: float, w: float,
                 crossover: Crossover, mutation: Mutation,
                 swap_limit: int,
                 selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__()
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.problem = problem
        self.solutions_size = solutions_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.swap_limit = swap_limit * solutions_size
        self.crossover = crossover
        self.mutation = mutation
        self.selection = selection
        self.start_computing_time = time.time()
        self.current_algorithm = 'PSO'

        self.pso = SingleObjectivePSO(
            problem=problem, swarm_size=solutions_size, c1=c1, c2=c2, w=w,
            termination_criterion=termination_criterion
        )
        self.ga = GeneticAlgorithm(
            problem=problem, population_size=solutions_size, offspring_population_size=solutions_size,
            crossover=crossover, mutation=mutation, selection=selection,
            termination_criterion=termination_criterion
        )

    def create_initial_solutions(self) -> List[S]:
        if self.current_algorithm == 'PSO':
            solutions = self.pso.create_initial_solutions()
            best_solutions = sorted(self.pso.solutions, key=lambda x: x.objectives)[:self.solutions_size]
            self.ga.set_solutions(best_solutions)
        else:
            solutions = self.ga.create_initial_solutions()
            self.pso.set_solutions(solutions)
        return solutions

    def update_progress(self) -> None:
        self.evaluations += self.solutions_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def evaluate(self, solution_list: List[S]):
        self.pso.evaluate(solution_list)

    def init_progress(self):
        self.evaluations = self.solutions_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def step(self):
        if self.current_algorithm == 'GA':
            self.ga.step()
        else:
            self.pso.step()

        self.update_progress()

        if self.evaluations % self.swap_limit == 0:
            if self.current_algorithm == 'GA':
                self.switch_to_pso()
            else:
                self.switch_to_ga()

    def switch_to_pso(self):
        best_solutions = sorted(self.ga.solutions, key=lambda x: x.objectives)[:self.solutions_size]
        self.pso.set_solutions(best_solutions)
        self.current_algorithm = 'PSO'
        print(f"Switched to PSO with {len(best_solutions)} solutions")

    def switch_to_ga(self):
        best_solutions = sorted(self.pso.solutions, key=lambda x: x.objectives)[:self.solutions_size]
        self.ga.set_solutions(best_solutions)  # Assuming set_solutions properly initializes GA state
        self.current_algorithm = 'GA'
        print(f"Switched to GA with {len(best_solutions)} solutions")

    def result(self) -> R:
        pso_best = self.pso.result() if self.pso.solutions else None
        print(f"Pso best {pso_best}, fitness {pso_best.objectives[0]}")
        ga_best = self.ga.result() if self.ga.solutions else None
        print(f"Ga best {ga_best}, fitness {ga_best.objectives[0]}")
        if pso_best and ga_best:
            return min(pso_best, ga_best, key=lambda sol: sol.objectives[0])
        return pso_best or ga_best

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def get_name(self) -> str:
        return "PGPHEA - Parallel Hybrid Evolutionary Algorithm"
