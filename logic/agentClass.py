from random import random, randint

import numpy as np


class Agent:
    def __init__(self, dnk: list, coord: list, env_size: int, goal: tuple, step_penalty: float, goal_bonus: int):
        self._dnk = np.array(dnk)
        self._start_coord = np.array(coord)
        self._goal = goal
        self._env_size = env_size
        self._step_penalty = step_penalty
        self._goal_bonus = goal_bonus

        self._fitness_cache = None
        self._coords_history_cache = None
        self._final_coord_cache = None
        self._cache_valid = False

    def __len__(self):
        return len(self._dnk)

    def __getitem__(self, item):
        return self._dnk[item]

    def __setitem__(self, key, value):
        self._dnk[key] = value
        self._cache_valid = False

    @property
    def dnk(self):
        return self._dnk

    @property
    def coord(self):
        if not self._cache_valid:
            return self._start_coord
        return self._final_coord_cache

    def _move(self) -> tuple:
        coords_history = []
        current_coord = self._start_coord.copy()
        for move in self._dnk:
            new_x, new_y = current_coord[0], current_coord[1]

            match move:
                case 0: new_x += 1
                case 1: new_x -= 1
                case 2: new_y += 1
                case 3: new_y -= 1

            if 0 <= new_x < self._env_size :
                current_coord[0] = new_x
            if 0 <= new_y < self._env_size :
                current_coord[1] = new_y
            coords_history.append(current_coord.copy())
        return coords_history, current_coord

    def simulate(self) -> tuple:
        if not self._cache_valid:
            hypotenuse = lambda k1, k2: np.sqrt(k1 ** 2 + k2 ** 2)
            coords_history, final_coord = self._move()
            x, y = self._goal

            leg1 = abs(self._goal[0] - final_coord[0])
            leg2 = abs(self._goal[1] - final_coord[1])
            dist_to_goal = hypotenuse(leg1, leg2)
            max_distance = hypotenuse(max(x, self._env_size - x), max(y, self._env_size - y))

            fitness = self.fitness(dist_to_goal, max_distance)

            self._fitness_cache = fitness
            self._coords_history_cache = coords_history
            self._final_coord_cache = final_coord
            self._cache_valid = True

        return self._fitness_cache, self._coords_history_cache

    def fitness(self, now_distance: float, max_distance: float):
        base_fitness = float((max_distance - now_distance) - (len(self._dnk) * self._step_penalty))

        goal_bonus = 0
        if np.array_equal(self.coord, self._goal):
            goal_bonus = self._goal_bonus

        fitness = base_fitness + goal_bonus
        return round(fitness, 3)

    def mutate(self, base_mutation_rate: float, generation: int, max_generations: int):
        progress = (max_generations - generation) / max_generations
        mutation_rate = base_mutation_rate * progress

        for i in range(len(self)):
            if random() < mutation_rate:
                self[i] = randint(0, 3)