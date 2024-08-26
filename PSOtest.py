from jmetal.problem.singleobjective.unconstrained import Rastrigin
from jmetal.util.termination_criterion import StoppingByEvaluations

from SingleObjectivePSO import SingleObjectivePSO

if __name__ == "__main__":
    problem = Rastrigin(100)

    algorithm = SingleObjectivePSO(
        problem=problem,
        swarm_size=100,
        c1=2.5,
        c2=0.3,
        w=0.17,
        termination_criterion=StoppingByEvaluations(max_evaluations=25000),
    )

    algorithm.run()
    result = algorithm.result()

    print("Algorithm: {}".format(algorithm.get_name()))
    print("Problem: {}".format(problem.name()))
    print("Solution: {}".format(result.variables))
    print("Fitness: {}".format(result.objectives[0]))
    print("Computing time: {}".format(algorithm.total_computing_time))