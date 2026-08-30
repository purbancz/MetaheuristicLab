import pytest
from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.basic.custom_ga import GeneticAlgorithm
from algorithm.basic.differential_evolution import DifferentialEvolution
from algorithm.tests.counting_problem import CountingProblem

_POP = 4
_MAX_EVAL = 20


def test_ga_reported_evaluations_match_actual_calls():
    problem = CountingProblem()
    ga = GeneticAlgorithm(
        problem=problem,
        population_size=_POP,
        offspring_population_size=_POP,
        crossover=SBXCrossover(1.0, 5.0),
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        termination_criterion=StoppingByEvaluations(max_evaluations=_MAX_EVAL),
    )
    ga.run()
    assert problem.evaluation_count == ga.evaluations


def test_de_reported_evaluations_match_actual_calls():
    problem = CountingProblem()
    de = DifferentialEvolution(
        problem=problem,
        swarm_size=_POP,
        termination_criterion=StoppingByEvaluations(max_evaluations=_MAX_EVAL),
    )
    de.run()
    assert problem.evaluation_count == de.evaluations
