from dataclasses import dataclass
from random import randint


@dataclass
class Config:
    MOVES = [0, 1, 2, 3]
    MAX_GENERATIONS = 101
    ENV_SIZE = 200
    AGENTS_CNT = 100
    DNK_CNT = 150
    FREQUENCY_SAVE = 10
    START_COORD = [ENV_SIZE // 2, ENV_SIZE // 2]
    STEP_PENALTY = 0.15
    GOAL_BONUS = 250
    PARENTS_PERCENT = 0.3
    ELITE_PERCENT = 0.1
    PARENTS_CNT = int(AGENTS_CNT * PARENTS_PERCENT)
    ELITE_CNT = int(PARENTS_CNT * ELITE_PERCENT)
    MUTATION_RATE = 0.4
    GOAL = (randint(0, ENV_SIZE), randint(0, ENV_SIZE))