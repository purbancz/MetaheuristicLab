import numpy as np
import math
from copy import copy
from typing import List, TypeVar

from jmetal.core.algorithm import EvolutionaryAlgorithm
from jmetal.core.problem import Problem
from jmetal.core.solution import FloatSolution
from jmetal.util.evaluator import Evaluator, SequentialEvaluator
from jmetal.util.termination_criterion import TerminationCriterion

S = TypeVar("S")
R = TypeVar("R")


class CMAES(EvolutionaryAlgorithm[FloatSolution, FloatSolution]):
    """
    Implementation of Covariance Matrix Adaptation Evolution Strategy (CMA-ES).
    """

    def __init__(
            self,
            problem: Problem,
            mu: int,
            lambda_: int,
            termination_criterion: TerminationCriterion,
            population_evaluator: Evaluator = SequentialEvaluator(),
    ):
        """
        :param problem: The problem to solve.
        :param mu: Number of parents selected.
        :param lambda_: Number of offspring generated.
        :param termination_criterion: The stopping condition.
        :param population_evaluator: The evaluator for the population.
        """
        super(CMAES, self).__init__(
            problem=problem, population_size=mu, offspring_population_size=lambda_
        )
        self.mu = mu
        self.lambda_ = lambda_

        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        # CMA-ES state variables
        self.n = problem.number_of_variables()
        self.mean = np.random.rand(self.n) * (self.problem.upper_bound[0] - self.problem.lower_bound[0]) + \
                    self.problem.lower_bound[0]
        self.sigma = 0.5
        self.C = np.identity(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)

        # Strategy parameters: Selection and recombination
        self.weights = np.array([np.log(self.mu + 0.5) - np.log(i + 1) for i in range(self.mu)])
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)

        # Strategy parameters: Adaptation
        self.cc = (4 + self.mueff / self.n) / (self.n + 4 + 2 * self.mueff / self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3) ** 2 + self.mueff)
        self.cmu = min(1 - self.c1, 2 * (self.mueff - 2 + 1 / self.mueff) / ((self.n + 2) ** 2 + self.mueff))
        self.damps = 1 + 2 * max(0, np.sqrt((self.mueff - 1) / (self.n + 1)) - 1) + self.cs
        self.chiN = self.n ** 0.5 * (1 - 1 / (4 * self.n) + 1 / (21 * self.n ** 2))

    def create_initial_solutions(self) -> List[FloatSolution]:
        """Creates the initial list of solutions (size mu)."""
        # These solutions are placeholders to fit the framework's flow.
        # The actual first generation is sampled in the first reproduction step.
        return [self.problem.create_solution() for _ in range(self.population_size)]

    def evaluate(self, solution_list: List[S]) -> List[S]:
        return self.population_evaluator.evaluate(solution_list, self.problem)

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        """Selection is a no-op; the model-based generation occurs in reproduction."""
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:
        """Generates lambda_ offspring from the current distribution."""
        offspring_population = []
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        D = np.diag(np.sqrt(eigenvals))
        B = eigenvecs

        for _ in range(self.offspring_population_size):
            z = np.random.randn(self.n)  # Sample from N(0, I)
            y = B.dot(D).dot(z)  # Convert to sample from N(0, C)
            x = self.mean + self.sigma * y  # Add mean and scale by sigma

            solution = self.problem.create_solution()
            solution.variables = x.tolist()
            offspring_population.append(solution)

        return offspring_population

    def replacement(
            self, population: List[FloatSolution], offspring_population: List[FloatSolution]
    ) -> List[FloatSolution]:
        """Updates the CMA-ES state and returns the mu best offspring."""
        # 1. Sort offspring by fitness
        offspring_population.sort(key=lambda s: s.objectives[0])
        best_solutions = offspring_population[: self.mu]

        # 2. Update state variables (mean, evolution paths, C, sigma)
        old_mean = self.mean
        x_k = np.array([s.variables for s in best_solutions])
        self.mean = self.weights.dot(x_k)

        # Update evolution paths
        y_k = (x_k - old_mean) / self.sigma
        y_w = self.weights.dot(y_k)

        # Update ps
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        C_inv_sqrt = eigenvecs.dot(np.diag(1.0 / np.sqrt(eigenvals))).dot(eigenvecs.T)
        self.ps = (1 - self.cs) * self.ps + np.sqrt(self.cs * (2 - self.cs) * self.mueff) * C_inv_sqrt.dot(y_w)

        # Update pc
        h_sig_val = (np.linalg.norm(self.ps) / np.sqrt(
            1 - (1 - self.cs) ** (2 * self.evaluations / self.lambda_)) / self.chiN) < (1.4 + 2 / (self.n + 1))
        h_sig = 1.0 if h_sig_val else 0.0

        self.pc = (1 - self.cc) * self.pc + h_sig * np.sqrt(self.cc * (2 - self.cc) * self.mueff) * y_w

        # Update covariance matrix C
        art_delta = (1 - h_sig) * self.cc * (2 - self.cc)
        rank_one_update = np.outer(self.pc, self.pc)
        rank_mu_update_y = y_k * np.sqrt(self.weights)[:, np.newaxis]
        rank_mu_update = rank_mu_update_y.T.dot(rank_mu_update_y)

        self.C = ((1 - self.c1 - self.cmu) * self.C +
                  self.c1 * (rank_one_update + art_delta * self.C) +
                  self.cmu * rank_mu_update)

        # Update step size sigma
        self.sigma *= np.exp((self.cs / self.damps) * (np.linalg.norm(self.ps) / self.chiN - 1))

        # Return the mu best solutions, which become the new population
        return best_solutions

    def result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return "CMAES"