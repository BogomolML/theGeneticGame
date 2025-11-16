from dataclasses import dataclass
from random import randint


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
