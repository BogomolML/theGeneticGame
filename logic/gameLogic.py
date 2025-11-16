from random import choices, randint, random
from token import MINUS

from agentClass import Agent
from populationClass import Population


'''Consts'''
MOVES = ['up', 'down', 'left', 'right']
MAX_GENERATIONS = 1000
ENV_SIZE = 50
AGENTS_CNT = 100
DNK_CNT = 200
FINE = 0.1
PARENTS_PERSENT = 0.3
ELITE_PERSENT = 0.2
PARENTS_CNT = int(AGENTS_CNT * PARENTS_PERSENT)
ELITE_CNT = int(PARENTS_CNT * ELITE_PERSENT)
BASE_MUTATION_RATE = 0.03
GOAL = (randint(0, ENV_SIZE), randint(0, ENV_SIZE))
FREQUENCY_SAVE = 50


def create_first_population(p: Population):
    for _ in range(AGENTS_CNT):
        dnk = []
        for __ in range(DNK_CNT):
            dnk.append(choices(MOVES, k=1)[0])
        p.add(Agent(dnk, ENV_SIZE, GOAL, FINE))

def crossover(p1, p2): # p - parent
    child_dnk = []
    for i in range(len(p1)):
        if random() > 0.5:
            child_dnk.append(p1[i])
        else:
            child_dnk.append(p2[i])
    return Agent(child_dnk, ENV_SIZE, GOAL, FINE)

def mutate(agent, gen):
    progress = (MAX_GENERATIONS - gen) / MAX_GENERATIONS
    mutation_rate = BASE_MUTATION_RATE * progress

    for i in range(len(agent)):
        if random() < mutation_rate:
            agent[i] = choices(MOVES, k=1)[0]

def create_new_population(p: Population, parents: list, gen):
    p.clear()
    p.population = parents[:ELITE_CNT]
    while len(p.population) < AGENTS_CNT:
        parent1, parent2 = choices(parents, k=2)
        child = crossover(parent1, parent2)
        mutate(child, gen)
        p.add(child)


if __name__ == '__main__':
    population = Population()
    create_first_population(population)
    test = Agent([0] * DNK_CNT, ENV_SIZE, GOAL, FINE)

    print('goal', GOAL)
    print('best fit', test.simulate(test=True))
    print('worse agent, first pop', min(population.evaluate_all(), key=lambda f: f[0])[0])
    print('best agent, first pop', max(population.evaluate_all(), key=lambda f: f[0])[0])

    for g in range(MAX_GENERATIONS):
        fitness_agents = population.evaluate_all()
        best_agents = [agent for fitness, agent in fitness_agents[:PARENTS_CNT]]
        create_new_population(population, best_agents, g)

    print('worse agent, last pop', min(population.evaluate_all(), key=lambda f: f[0])[0])
    print('best agent, last pop', max(population.evaluate_all(), key=lambda f: f[0])[0])
