import numpy as np
from typing import List, TypeVar

from jmetal.config import store
from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")


class CMAES(EvolutionaryAlgorithm[FloatSolution, FloatSolution]):
    """
    A definitive, robust, and self-contained implementation of CMA-ES for jMetalPy.
    This version overrides the `step` method to ensure the correct workflow.
    """

    def __init__(
            self,
            problem: Problem,
            mu: int,
            lambda_: int,
            termination_criterion: TerminationCriterion = store.default_termination_criteria,
            population_evaluator: Evaluator = store.default_evaluator,
    ):
        super().__init__(
            problem=problem,
            population_size=mu,
            offspring_population_size=lambda_
        )
        self.mu = mu
        self.lambda_ = lambda_
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion

        self.observable.register(self.termination_criterion)
        self.best_solution_so_far = None

        # --- Initialize CMA-ES state ---
        self.n = self.problem.number_of_variables()
        self.lower_bound = np.array(self.problem.lower_bound)
        self.upper_bound = np.array(self.problem.upper_bound)

        self.sigma = 0.3 * (self.upper_bound[0] - self.lower_bound[0])
        self.mean = self.lower_bound + np.random.rand(self.n) * (self.upper_bound - self.lower_bound)

        self.C = np.identity(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)

        # --- Recombination weights and adaptation constants ---
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        self.chiN = np.sqrt(self.n) * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

    def step(self) -> None:
        """
        Overrides the base class `step` to implement the correct CMA-ES workflow.
        1. Sample lambda_ offspring.
        2. Evaluate them.
        3. Update the model state and select mu parents for the next generation.
        """
        # 1. Sample new population from the model
        offspring_population = self._sample(self.offspring_population_size)

        # 2. Evaluate the new population
        offspring_population = self.evaluate(offspring_population)

        # 3. Update model and select the mu best solutions for the next generation
        self.solutions = self.replacement(self.solutions, offspring_population)

    def _sample(self, count: int) -> List[FloatSolution]:
        """Helper function to sample points from the current model."""
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        D = np.diag(np.sqrt(np.maximum(eigenvals, 1e-20)))
        B = eigenvecs

        offspring = []
        for _ in range(count):
            z = np.random.randn(self.n)
            y = B @ D @ z
            x = self.mean + self.sigma * y
            x = np.clip(x, self.lower_bound, self.upper_bound)

            sol = self.problem.create_solution()
            sol.variables = x.tolist()
            offspring.append(sol)
        return offspring

    def create_initial_solutions(self) -> List[FloatSolution]:
        return self._sample(self.offspring_population_size)

    def init_progress(self) -> None:
        self.evaluations = len(self.solutions)

        self.best_solution_so_far = min(
            self.solutions,
            key=lambda solution: solution.objectives[0],
        )

        observable_data = self.observable_data()
        self.observable.notify_all(**observable_data)

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def result(self) -> FloatSolution:
        return self.best_solution_so_far

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        # Not used in our custom `step`, but must be implemented.
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:
        # Not used in our custom `step`, but must be implemented.
        return []

    def replacement(
            self, population: List[FloatSolution], offspring_population: List[FloatSolution]
    ) -> List[FloatSolution]:
        """Update the CMA-ES model and select the new population."""
        offspring_population.sort(key=lambda s: s.objectives[0])
        best_solutions = offspring_population[:self.mu]

        # Update the overall best solution found so far
        if self.best_solution_so_far is None or best_solutions[0].objectives[0] < self.best_solution_so_far.objectives[
            0]:
            self.best_solution_so_far = best_solutions[0]

        if self.sigma < 1e-12:
            return best_solutions

        old_mean = self.mean.copy()
        X = np.array([s.variables for s in best_solutions])
        self.mean = self.weights @ X

        Y = (X - old_mean) / self.sigma
        y_w = self.weights @ Y

        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        inv_sqrt = eigenvecs @ np.diag(1 / np.sqrt(np.maximum(eigenvals, 1e-20))) @ eigenvecs.T
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * (inv_sqrt @ y_w)

        h_sig = 1.0 if (np.linalg.norm(self.ps) / (
                    self.chiN * np.sqrt(1 - (1 - self.cs) ** (2 * self.evaluations / self.lambda_)))) < (
                                   1.4 + 2 / (self.n + 1)) else 0.0
        self.pc = (1 - self.cc) * self.pc + h_sig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

        rank_one = np.outer(self.pc, self.pc)
        rank_mu = Y.T @ np.diag(self.weights) @ Y
        self.C = ((1 - self.c1 - self.cmu) * self.C + self.c1 * (
                    rank_one + (1 - h_sig) * self.cc * (2 - self.cc) * self.C) + self.cmu * rank_mu)

        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        return best_solutions

    def get_name(self) -> str:
        return "CMAES"