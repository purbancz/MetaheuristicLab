from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from PGSHEA import PGSHEA

if __name__ == "__main__":
    problem = Rastrigin(1)

    algorithm = PGSHEA(
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
        problem=problem,
        solutions_size=100,
        mutation=PolynomialMutation(1.0 / problem.number_of_variables(), 20.0),
        crossover=SBXCrossover(0.9, 5.0),
        swap_limit=50,
        c1=1.7,
        c2=0.7,
        w=0.7
    )

    algorithm.run()
    result = algorithm.result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
