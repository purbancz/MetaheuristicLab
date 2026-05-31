from unittest.mock import Mock

from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem import Sphere
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.PGCHEA import PGCHEA


def test_pgchea_switch_to_ga_updates_ga_population() -> None:
    problem = Sphere(3)
    algorithm = PGCHEA(
        problem=problem,
        solutions_size=4,
        c1=1.0,
        c2=1.0,
        w=0.5,
        crossover=SBXCrossover(1.0, 5.0),
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        starting_algorithm="PSO",
        termination_criterion=StoppingByEvaluations(max_evaluations=20),
    )
    algorithm.create_initial_solutions()
    algorithm.ga.set_solutions = Mock()

    algorithm.switch_to_ga()

    assert algorithm.current_algorithm == "GA"
