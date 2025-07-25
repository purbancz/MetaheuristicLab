from typing import List

from jmetal.core.observer import Observer


class FitnessObserver(Observer):
    def __init__(self, interval: int = 1) -> None:
        self.interval = interval
        # the next eval count at which we want to sample
        self.next_eval = interval
        self.best_fitness_history: List[float] = []

    def update(self, *args, **kwargs):
        solutions   = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]
        best_f       = solutions.objectives[0]

        # once we hit or exceed the next threshold, record and bump it
        if evaluations >= self.next_eval:
            self.best_fitness_history.append(best_f)
            self.next_eval += self.interval
