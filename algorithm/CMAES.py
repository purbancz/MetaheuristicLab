import numpy as np
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
    Covariance Matrix Adaptation Evolution Strategy (CMA-ES) for jMetalPy.
    """
    def __init__(
        self,
        problem: Problem,
        mu: int,
        lambda_: int,
        termination_criterion: TerminationCriterion,
        population_evaluator: Evaluator = SequentialEvaluator(),
    ):
        super().__init__(problem=problem, population_size=mu, offspring_population_size=lambda_)
        self.mu = mu
        self.lambda_ = lambda_
        self.population_evaluator = population_evaluator
        self.termination_criterion = termination_criterion
        self.observable.register(termination_criterion)

        # CMA-ES parameters
        self.n = problem.number_of_variables()
        lb = np.array(problem.lower_bound)
        ub = np.array(problem.upper_bound)
        self.lower_bound = lb
        self.upper_bound = ub

        self.mean = lb + (ub - lb) * np.random.rand(self.n)
        self.sigma = 0.5
        self.C = np.identity(self.n)
        self.pc = np.zeros(self.n)
        self.ps = np.zeros(self.n)

        # Selection & recombination weights
        self.weights = np.log(self.mu + 0.5) - np.log(np.arange(1, self.mu + 1))
        self.weights /= np.sum(self.weights)
        self.mueff = np.sum(self.weights) ** 2 / np.sum(self.weights ** 2)

        # Adaptation parameters
        self.cc = (4 + self.mueff/self.n) / (self.n + 4 + 2*self.mueff/self.n)
        self.cs = (self.mueff + 2) / (self.n + self.mueff + 5)
        self.c1 = 2 / ((self.n + 1.3)**2 + self.mueff)
        self.cmu = min(1 - self.c1, 2*(self.mueff - 2 + 1/self.mueff)/((self.n+2)**2 + self.mueff))
        self.damps = 1 + 2*max(0, np.sqrt((self.mueff-1)/(self.n+1)) - 1) + self.cs
        self.chiN = np.sqrt(self.n) * (1 - 1/(4*self.n) + 1/(21*self.n**2))

    def create_initial_solutions(self) -> List[FloatSolution]:
        """Sample initial mu solutions from N(mean, sigma^2 C)"""
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        D = np.diag(np.sqrt(eigenvals))
        B = eigenvecs
        population = []
        for _ in range(self.population_size):
            z = np.random.randn(self.n)
            y = B.dot(D).dot(z)
            x = self.mean + self.sigma * y
            x = np.clip(x, self.lower_bound, self.upper_bound)
            sol = self.problem.create_solution()
            sol.variables = x.tolist()
            population.append(sol)
        return population

    def evaluate(self, solution_list: List[FloatSolution]) -> List[FloatSolution]:
        evaluated = self.population_evaluator.evaluate(solution_list, self.problem)
        self.evaluations += len(evaluated)
        self.observable.notify(self)
        return evaluated

    def stopping_condition_is_met(self) -> bool:
        return self.termination_criterion.is_met

    def selection(self, population: List[FloatSolution]) -> List[FloatSolution]:
        return population

    def reproduction(self, population: List[FloatSolution]) -> List[FloatSolution]:
        """Generate lambda_ offspring from current distribution"""
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        D = np.diag(np.sqrt(eigenvals))
        B = eigenvecs
        offspring = []
        for _ in range(self.offspring_population_size):
            z = np.random.randn(self.n)
            y = B.dot(D).dot(z)
            x = self.mean + self.sigma * y
            x = np.clip(x, self.lower_bound, self.upper_bound)
            sol = self.problem.create_solution()
            sol.variables = x.tolist()
            offspring.append(sol)
        return offspring

    def replacement(
        self,
        population: List[FloatSolution],
        offspring_population: List[FloatSolution]
    ) -> List[FloatSolution]:
        """Update CMA-ES state and select mu best offspring"""
        offspring_population.sort(key=lambda s: s.objectives[0])
        best = offspring_population[:self.mu]

        old_mean = self.mean.copy()
        x_k = np.array([s.variables for s in best])
        self.mean = self.weights.dot(x_k)

        y_k = (x_k - old_mean) / self.sigma
        y_w = self.weights.dot(y_k)

        # Update evolution paths
        eigenvals, eigenvecs = np.linalg.eigh(self.C)
        C_inv_sqrt = eigenvecs.dot(np.diag(1.0/np.sqrt(eigenvals))).dot(eigenvecs.T)
        self.ps = (1 - self.cs)*self.ps + np.sqrt(self.cs*(2-self.cs)*self.mueff)*C_inv_sqrt.dot(y_w)

        # Heaviside signal
        h_sig = 1.0 if (
            np.linalg.norm(self.ps) /
            (self.chiN * np.sqrt(1 - (1 - self.cs)**(2 * self.evaluations / self.lambda_)))
        ) < (1.4 + 2 / (self.n + 1)) else 0.0
        self.pc = (1 - self.cc)*self.pc + h_sig*np.sqrt(self.cc*(2-self.cc)*self.mueff)*y_w

        # Covariance matrix update
        art_delta = (1 - h_sig)*self.cc*(2-self.cc)
        rank_one = np.outer(self.pc, self.pc)
        rank_mu = (y_k * np.sqrt(self.weights)[:, None]).T.dot(y_k * np.sqrt(self.weights)[:, None])
        self.C = (
            (1 - self.c1 - self.cmu) * self.C
            + self.c1 * (rank_one + art_delta * self.C)
            + self.cmu * rank_mu
        )

        # Repair C to ensure positive-definiteness
        ev, evec = np.linalg.eigh(self.C)
        ev = np.maximum(ev, 1e-20)
        self.C = evec.dot(np.diag(ev)).dot(evec.T)

        # Step-size control
        self.sigma *= np.exp((self.cs/self.damps) * (np.linalg.norm(self.ps)/self.chiN - 1))

        return best

    def result(self) -> R:
        return self.solutions[0]

    def get_name(self) -> str:
        return "CMAES"
