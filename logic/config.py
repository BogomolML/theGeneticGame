from dataclasses import dataclass
from random import randint


@dataclass
class Config:
    MOVES = [0, 1, 2, 3]
    MAX_GENERATIONS = 1001
    ENV_SIZE = 500
    AGENTS_CNT = 200
    DNK_CNT = 750
    FREQUENCY_SAVE = 100
    START_COORD = [ENV_SIZE // 2, ENV_SIZE // 2]
    STEP_PENALTY = 0.15
    GOAL_BONUS = 250
    PARENTS_PERSENT = 0.3
    ELITE_PERSENT = 0.1
    PARENTS_CNT = int(AGENTS_CNT * PARENTS_PERSENT)
    ELITE_CNT = int(PARENTS_CNT * ELITE_PERSENT)
    MUTATION_RATE = 0.4
    GOAL = (randint(0, ENV_SIZE), randint(0, ENV_SIZE))