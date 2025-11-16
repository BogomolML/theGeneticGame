from random import choices, randint, random
from dataclasses import dataclass
from copy import deepcopy

from agentClass import Agent
from populationClass import Population

from visualize.classVisualizer import GeneticGameVisualizr


# ============= CONFIG =============
@dataclass
class Config:
    MOVES = ['up', 'down', 'left', 'right']
    MAX_GENERATIONS = 251
    ENV_SIZE = 1000
    AGENTS_CNT = 500
    DNK_CNT = 500
    START_COORD = [ENV_SIZE // 2, ENV_SIZE // 2]
    FINE = 0.15
    PARENTS_PERSENT = 0.3
    ELITE_PERSENT = 0.1
    PARENTS_CNT = int(AGENTS_CNT * PARENTS_PERSENT)
    ELITE_CNT = int(PARENTS_CNT * ELITE_PERSENT)
    MUTATION_RATE = 0.15
    GOAL = (randint(0, ENV_SIZE), randint(0, ENV_SIZE))
    FREQUENCY_SAVE = 50
# ==================================


def create_first_population(p: Population):
    for _ in range(Config.AGENTS_CNT):
        dnk = []
        for __ in range(Config.DNK_CNT):
            dnk.append(choices(Config.MOVES, k=1)[0])
        p.add(Agent(dnk, Config.START_COORD, Config.ENV_SIZE, Config.GOAL, Config.FINE))

def crossover(p1, p2): # p - parent
    child_dnk = []
    for i in range(len(p1)):
        if random() > 0.5:
            child_dnk.append(p1[i])
        else:
            child_dnk.append(p2[i])
    return Agent(child_dnk, Config.START_COORD, Config.ENV_SIZE, Config.GOAL, Config.FINE)

def mutate(agent, gen):
    progress = (Config.MAX_GENERATIONS - gen) / Config.MAX_GENERATIONS
    mutation_rate = Config.MUTATION_RATE * progress

    for i in range(len(agent)):
        if random() < mutation_rate:
            agent[i] = choices(Config.MOVES, k=1)[0]

def create_new_population(p: Population, parents: list, gen):
    p.clear()
    p.population = parents[:Config.ELITE_CNT]
    while len(p.population) < Config.AGENTS_CNT:
        parent1, parent2 = choices(parents, k=2)
        child = crossover(parent1, parent2)
        mutate(child, gen)
        p.add(child)


if __name__ == '__main__':
    visualizer = GeneticGameVisualizr(Config.ENV_SIZE, Config.GOAL)

    population = Population()
    create_first_population(population)
    fitness_history = []
    best_agents_history = []
    initial_population = None

    for g in range(Config.MAX_GENERATIONS):
        fitness_agents = population.evaluate_all()
        fitness_scores = [fitness for fitness, _ in fitness_agents]
        fitness_history.append(fitness_scores)

        if g == 0:
            initial_population = deepcopy(population)

        if g % Config.FREQUENCY_SAVE == 0:
            visualizer.plot_generation_snapshot(population, g)

            best_agent = fitness_agents[0][1]
            visualizer.plot_agent_trajectory(best_agent, g)

        best_agents = [agent for fitness, agent in fitness_agents[:Config.PARENTS_CNT]]
        create_new_population(population, best_agents, g)

    visualizer.plot_fitness_evolution(fitness_history)
    visualizer.plot_comparison(initial_population, population)
    visualizer.save_experiment_info(Config, fitness_history)