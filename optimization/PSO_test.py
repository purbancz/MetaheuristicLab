from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from algorithm.single_objective_PSO import SingleObjectivePSO
from observer.fitness_observer import FitnessObserver

if __name__ == "__main__":
    problem = Rastrigin(100)

    algorithm = SingleObjectivePSO(
        problem=problem,
        swarm_size=100,
        c1=1.97,
        c2=0.94,
        w=0.56,
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
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
