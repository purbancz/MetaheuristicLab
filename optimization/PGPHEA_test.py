from jmetal.operator import PolynomialMutation, SBXCrossover
from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.PGPHEA import PGPHEA
from observer.fitness_observer import FitnessObserver

if __name__ == "__main__":
    problem = Rastrigin(100)

    algorithm = PGPHEA(
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
        problem=problem,
        solutions_size=100,
        mutation=PolynomialMutation(0.26 / problem.number_of_variables(), 20.0),
        crossover=SBXCrossover(1, 5.0),
        exchange_interval=13,
        exchange_number=11,
        c1=0.001,
        c2=0.3,
        w=0.00001,
    )

    fitness_observer = FitnessObserver(interval=100)
    algorithm.observable.register(fitness_observer)

    algorithm.run()
    result = algorithm.result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))
    print("Best fitness history:", fitness_observer.best_fitness_history)
