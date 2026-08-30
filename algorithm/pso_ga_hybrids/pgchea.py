import time
from typing import TypeVar, List

from jmetal.config import store
from jmetal.core.operator import Selection, Crossover, Mutation
from jmetal.core.problem import FloatProblem
from jmetal.core.algorithm import Algorithm
from jmetal.operator import BinaryTournamentSelection
from jmetal.util.comparator import ObjectiveComparator
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion

from algorithm.basic.custom_ga import GeneticAlgorithm
from algorithm.basic.single_objective_pso import SingleObjectivePSO
from operator_wrapper.PSO_GA_wrapper import MutationWithPsoAttributes, CrossoverWithPsoAttributes

S = TypeVar("S")
R = TypeVar("R")


class PGCHEA(Algorithm[S, R]):
    def __init__(self, problem: FloatProblem, solutions_size: int,
                 c1: float, c2: float, w: float,
                 crossover: Crossover, mutation: Mutation,
                 starting_algorithm: str,
                 inherit_best: bool = True,
                 selection: Selection = BinaryTournamentSelection(ObjectiveComparator(0)),
                 solution_evaluator: Evaluator = store.default_evaluator,
                 termination_criterion: TerminationCriterion = store.default_termination_criteria):
        super().__init__()
        self.global_best = None
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)
        self.solution_evaluator = solution_evaluator
        self.problem = problem
        self.solutions_size = solutions_size
        self.c1 = c1
        self.c2 = c2
        self.w = w
        self.crossover = CrossoverWithPsoAttributes(crossover, probability=crossover.probability,
                                                    inherit_best=inherit_best)
        self.mutation = MutationWithPsoAttributes(mutation, probability=mutation.probability)
        self.selection = selection
        self.start_computing_time = time.time()
        self.current_algorithm = starting_algorithm
        self.best_global = None

        self.pso = SingleObjectivePSO(
            problem=problem, swarm_size=solutions_size, c1=c1, c2=c2, w=w, termination_criterion=termination_criterion
        )
        self.ga = GeneticAlgorithm(
            problem=problem, population_size=solutions_size, offspring_population_size=100,
            crossover=self.crossover, mutation=self.mutation, selection=selection
        )

    def create_initial_solutions(self) -> List[S]:
        if self.current_algorithm == 'PSO':
            solutions = self.pso.create_initial_solutions()
            self.best_global = min(solutions, key=lambda s: s.objectives[0])
            # print(f"Initial best from PSO: {self.best_global.objectives[0]}")
        else:
            solutions = self.ga.create_initial_solutions()
            self.best_global = min(solutions, key=lambda s: s.objectives[0])
            # print(f"Initial best from GA: {self.best_global.objectives[0]}")

        self.ga.set_solutions(solutions)
        self.pso.set_solutions(solutions)

        return solutions

    def step(self):
        if self.current_algorithm == 'GA':
            self.ga.step()
            if self.best_global is None or self.ga.solutions[0].objectives[0] < self.best_global.objectives[0]:
                self.best_global = self.ga.solutions[0]
            self.switch_to_pso()
        else:
            self.pso.step()
            if self.best_global is None or self.pso.best_global.objectives[0] < self.best_global.objectives[0]:
                self.best_global = self.pso.best_global
            self.switch_to_ga()

    def switch_to_pso(self):
        self.pso.set_solutions(self.update_attributes())
        self.current_algorithm = 'PSO'

    def switch_to_ga(self):
        self.ga.set_solutions(self.pso.solutions)
        self.current_algorithm = 'GA'

    def update_attributes(self):
        best_solutions = sorted(self.ga.solutions, key=lambda x: x.objectives[:self.solutions_size - 1])
        for sol in best_solutions:
            if 'best_objective' not in sol.attributes or sol.objectives[0] < sol.attributes['best_objective']:
                sol.attributes['best_position'] = sol.variables[:]
                sol.attributes['best_objective'] = sol.objectives[0]
        return best_solutions

    def result(self):
        return self.best_global

    def evaluate(self, solution_list: List[S]):
        return self.solution_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self):
        return self.termination_criterion.is_met

    def init_progress(self):
        self.evaluations = self.solutions_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def update_progress(self) -> None:
        self.evaluations += self.solutions_size

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def observable_data(self) -> dict:
        return {
            "PROBLEM": self.problem,
            "EVALUATIONS": self.evaluations,
            "SOLUTIONS": self.result(),
            "COMPUTING_TIME": time.time() - self.start_computing_time,
        }

    def get_name(self):
        return "PGCHEA: PSO-GA Consecutive Hybrid Evolutionary Algorithm"
