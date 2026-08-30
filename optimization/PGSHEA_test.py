from jmetal.operator.crossover import SBXCrossover
from jmetal.operator.mutation import PolynomialMutation
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.pso_ga_hybrids.pgshea import PGSHEA

if __name__ == "__main__":
    problem = Rastrigin(100)

    algorithm = PGSHEA(
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
        problem=problem,
        solutions_size=100,
        mutation=PolynomialMutation(0.38 / problem.number_of_variables(), 20.0),
        crossover=SBXCrossover(1, 5.0),
        swap_interval=125,
        c1=2.63,
        c2=0.21,
        w=0.01,
        starting_algorithm='PSO',
    )

    algorithm.run()
    result = algorithm.result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
