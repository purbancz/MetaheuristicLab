import random
from typing import List

from jmetal.core.operator import Mutation, Crossover
from jmetal.core.solution import FloatSolution


class MutationWithPsoAttributes(Mutation):
    def __init__(self, mutation_operator, probability: float):
        super().__init__(probability)
        self.mutation_operator = mutation_operator

    def execute(self, solution: FloatSolution) -> FloatSolution:
        original_attributes = solution.attributes.copy()
        mutated_solution = self.mutation_operator.execute(solution)
        mutated_solution.attributes.update(original_attributes)
        return mutated_solution

    def get_name(self):
        return f"{self.mutation_operator.get_name()} with PSO attributes"


class CrossoverWithPsoAttributes(Crossover):
    def __init__(self, crossover_operator, probability: float, inherit_best: bool = True):
        super().__init__(probability)
        self.crossover_operator = crossover_operator
        self.inherit_best = inherit_best

    def execute(self, parents: List[FloatSolution]) -> List[FloatSolution]:
        offspring = self.crossover_operator.execute(parents)
        if self.inherit_best:
            better_parent = min(parents, key=lambda p: p.attributes.get('best_objective', float('inf')))
        else:
            better_parent = random.choice(parents)

        for child in offspring:
            if 'best_position' in better_parent.attributes and 'best_objective' in better_parent.attributes:
                child.attributes['best_position'] = better_parent.attributes['best_position']
                child.attributes['best_objective'] = better_parent.attributes['best_objective']
            velocity = better_parent.attributes.get('velocity')
            if velocity is not None:
                child.attributes['velocity'] = velocity

        return offspring

    def get_number_of_parents(self) -> int:
        return self.crossover_operator.get_number_of_parents()

    def get_number_of_children(self) -> int:
        return self.crossover_operator.get_number_of_children()

    def get_name(self):
        mode = "better" if self.inherit_best else "random"
        return f"{self.crossover_operator.get_name()} with PSO attributes (inherit mode: {mode})"
