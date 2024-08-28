from jmetal.core.observer import Observer


class FitnessObserver(Observer):
    def __init__(self, interval: int = 1) -> None:
        """Initialize observer with a specific update frequency."""
        self.display_interval = interval
        self.best_fitness_history = []

    def update(self, *args, **kwargs):
        solutions = kwargs["SOLUTIONS"]
        evaluations = kwargs["EVALUATIONS"]

        if evaluations % self.display_interval == 0:
            self.best_fitness_history.append(solutions.objectives[0])
            # print(f"Evaluations: {evaluations}, Best Fitness: {solutions.objectives[0]}")
