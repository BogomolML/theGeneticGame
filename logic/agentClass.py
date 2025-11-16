import numpy as np


class Agent:
    def __init__(self, dnk: list, coord: list, env_size: int, goal: tuple, fine: float):
        self._dnk = np.array(dnk)
        self._start_coord = coord
        self._coord = coord
        self._goal = goal
        self._env_size = env_size
        self._fine = fine

    def __len__(self):
        return len(self._dnk)

    def __getitem__(self, item):
        return self._dnk[item]

    def __setitem__(self, key, value):
        self._dnk[key] = value

    @property
    def dnk(self):
        return self._dnk

    @property
    def coord(self):
        return self._start_coord

    @coord.setter
    def coord(self, value):
        self._start_coord = value

    def _move(self):
        for move in self._dnk:
            new_x, new_y = self._coord[0], self._coord[1]

            match move:
                case 'right': new_x += 1
                case 'left': new_x -= 1
                case 'up': new_y += 1
                case 'down': new_y -= 1

            if 0 <= new_x < self._env_size:
                self._coord[0] = new_x
            if 0 <= new_y < self._env_size:
                self._coord[1] = new_y

    def simulate(self) -> float:
        hypotenuse = lambda k1, k2: np.sqrt(k1**2 + k2**2)
        self._coord = self._start_coord
        self._move()
        x, y = self._goal

        leg1 = abs(self._goal[0] - self.coord[0])
        leg2 = abs(self._goal[1] - self.coord[1])
        dist_to_goal = hypotenuse(leg1, leg2)
        max_distance = hypotenuse(max(x, self._env_size - x), max(y, self._env_size - y))

        fitness = self.fitness(dist_to_goal, max_distance)
        return fitness

    def fitness(self, now_distance: float, max_distance: float):
        fitness = float((max_distance - now_distance) - (len(self._dnk) * self._fine))
        return round(fitness, 3)
