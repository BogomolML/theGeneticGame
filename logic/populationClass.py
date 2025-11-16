
class Population:
    def __init__(self):
        self._population = []

    def add(self, agent):
        self._population.append(agent)

    @property
    def population(self):
        return self._population

    @population.setter
    def population(self, value):
        self._population = value

    def clear(self):
        self._population.clear()

    def evaluate_all(self) -> list:
        fitness_scores = []
        for agent in self._population:
            fitness = agent.simulate()
            fitness_scores.append((fitness, agent))
        return sorted(fitness_scores, key=lambda f: f[0], reverse=True)