from copy import deepcopy
from random import choices, random

from logic.agentClass import Agent
from logic.config import Config
from logic.populationClass import Population
from visualize.classVisualizer import GeneticGameVisualizr


def create_first_population(p: Population):
    for _ in range(Config.AGENTS_CNT):
        dnk = []
        for __ in range(Config.DNK_CNT):
            dnk.append(choices(Config.MOVES, k=1)[0])
        p.add(Agent(dnk, Config.START_COORD.copy(), Config.ENV_SIZE, Config.GOAL,
                    Config.STEP_PENALTY, Config.GOAL_BONUS))

def crossover(parent1, parent2):
    child_dnk = []
    for i in range(len(parent1)):
        if random() > 0.5:
            child_dnk.append(parent1[i])
        else:
            child_dnk.append(parent2[i])
    return Agent(child_dnk, Config.START_COORD.copy(), Config.ENV_SIZE, Config.GOAL,
                 Config.STEP_PENALTY, Config.GOAL_BONUS)

def create_new_population(p: Population, parents: list, gen):
    p.clear()
    p.population = parents[:Config.ELITE_CNT]
    while len(p.population) < Config.AGENTS_CNT:
        parent1, parent2 = [deepcopy(agent) for agent in choices(parents, k=2)]
        child = crossover(parent1, parent2)
        child.mutate(Config.MUTATION_RATE, gen, Config.MAX_GENERATIONS)
        p.add(child)


if __name__ == '__main__':
    visualizer = GeneticGameVisualizr(Config.ENV_SIZE, Config.GOAL)

    population = Population()
    create_first_population(population)
    fitness_history = []
    best_agents_history = []
    initial_population = None

    for g in range(Config.MAX_GENERATIONS):
        scored_agents = population.evaluate_all()
        fitness_scores = [fitness for fitness, _ in scored_agents]
        fitness_history.append(fitness_scores)

        if g == 0:
            initial_population = deepcopy(population)

        if g % Config.FREQUENCY_SAVE == 0:
            visualizer.plot_generation_snapshot(population, g)

            best_agent = scored_agents[0][1]
            visualizer.plot_agent_trajectory(best_agent, g)

        best_agents = [agent for fitness, agent in scored_agents[:Config.PARENTS_CNT]]
        create_new_population(population, best_agents, g)

    visualizer.plot_fitness_evolution(fitness_history)
    visualizer.plot_comparison(initial_population, population)
    visualizer.save_experiment_info(Config, fitness_history)