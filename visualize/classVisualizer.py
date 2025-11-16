import os
from shutil import rmtree
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from logic.populationClass import Population
from logic.agentClass import Agent
from logic.config import Config


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLOTS_DIR = os.path.join(PROJECT_ROOT, "plots")

def ensure_plots_dir():
    if not os.path.exists(PLOTS_DIR):
        os.makedirs(PLOTS_DIR)

class GeneticGameVisualizr:
    def __init__(self, env_size: int, goal: tuple):
        self._env_size = env_size
        self._goal = goal
        self._fig = None

        try:
            rmtree(PLOTS_DIR)
        except FileNotFoundError:
            pass
        ensure_plots_dir()

    @staticmethod
    def plot_fitness_evolution(fitness_history: list[list[float]]):
        plt.figure(figsize=(12, 5))

        generations = range(len(fitness_history))

        best_fitness = [max(f) for f in fitness_history]
        avg_fitness = [np.mean(f) for f in fitness_history]
        worst_fitness = [min(f) for f in fitness_history]

        plt.subplot(1, 2, 1)
        plt.plot(generations, best_fitness, label='Лучший', color='green', linewidth=2)
        plt.plot(generations, avg_fitness, label='Средний', color='blue', alpha=0.7)
        plt.plot(generations, worst_fitness, label='Худший', color='red', alpha=0.7)
        plt.xlabel('Поколение')
        plt.ylabel('Фитнес')
        plt.title('Эволюция фитнеса')
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        diversity = [max(f) - min(f) for f in fitness_history]
        plt.plot(generations, diversity, color='purple', linewidth=2)
        plt.xlabel('Поколение')
        plt.ylabel('Разброс фитнеса')
        plt.title('Разнообразие популяции')
        plt.grid(True, alpha=0.3)

        plt.tight_layout()

        filename = os.path.join(PLOTS_DIR, "fitness_evolution.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"График сохранен: {filename}")

    def plot_generation_snapshot(self, population: Population, generation: int,
                                 top_n: int = 10):
        plt.figure(figsize=(10, 10))

        fitness_agents = population.evaluate_all()
        top_agents = [agent for _, agent in fitness_agents[:top_n]]
        other_agents = [agent for _, agent in fitness_agents[top_n:]]

        other_coords = []
        for agent in other_agents:
            agent.simulate()
            other_coords.append(agent.coord)

        if other_coords:
            other_coords = np.array(other_coords)
            plt.scatter(other_coords[:, 0], other_coords[:, 1],
                        alpha=0.3, color='gray', label=f'Остальные агенты')

        colors = plt.cm.viridis(np.linspace(0, 1, top_n))
        for i, agent in enumerate(top_agents):
            agent.simulate()
            plt.scatter(agent.coord[0], agent.coord[1],
                        color=colors[i], s=80, label=f'Топ {i + 1}')

        plt.scatter(0, 0, color='green', s=150, marker='s', label='Старт', edgecolors='black')
        plt.scatter(self._goal[0], self._goal[1], color='red', s=200,
                    marker='*', label='Цель', edgecolors='black')

        plt.xlim(-2, self._env_size + 2)
        plt.ylim(-2, self._env_size + 2)
        plt.xlabel('X координата')
        plt.ylabel('Y координата')
        plt.title(f'Поколение {generation}\nЦель: {self._goal}')
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(PLOTS_DIR, f"generation_{generation:04d}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Снимок сохранен: {filename}")

    def plot_agent_trajectory(self, agent: Agent, generation: int):
        plt.figure(figsize=(8, 8))

        agent.coord = Config.START_COORD
        trajectory = [Config.START_COORD]

        for move in agent.dnk:
            new_x, new_y = agent.coord[0], agent.coord[1]
            match move:
                case 'right':
                    new_x += 1
                case 'left':
                    new_x -= 1
                case 'up':
                    new_y += 1
                case 'down':
                    new_y -= 1

            if 0 <= new_x < self._env_size:
                agent.coord[0] = new_x
            if 0 <= new_y < self._env_size:
                agent.coord[1] = new_y

            trajectory.append([agent.coord[0], agent.coord[1]])

        trajectory = np.array(trajectory)

        plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', alpha=0.5, linewidth=1)
        plt.scatter(trajectory[:, 0], trajectory[:, 1], c=range(len(trajectory)),
                    cmap='viridis', s=20, alpha=0.6)

        plt.scatter(0, 0, color='green', s=100, marker='s', label='Старт', zorder=5)
        plt.scatter(self._goal[0], self._goal[1], color='red', s=150,
                    marker='*', label='Цель', zorder=5)
        plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='purple', s=80,
                    marker='o', label='Конец', zorder=5)

        plt.colorbar(label='Шаг движения')
        plt.xlim(-1, self._env_size + 1)
        plt.ylim(-1, self._env_size + 1)
        plt.xlabel('X координата')
        plt.ylabel('Y координата')
        plt.title(f'Траектория агента (Поколение {generation})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        filename = os.path.join(PLOTS_DIR, f"trajectory_gen_{generation:04d}.png")
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Траектория сохранена: {filename}")

    def plot_comparison(self, initial_population: Population, final_population: Population):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

        for idx, (population, title) in enumerate([
            (initial_population, "Начальная популяция"),
            (final_population, "Финальная популяция")
        ]):
            ax = ax1 if idx == 0 else ax2

            coords = []
            for agent in population.population:
                agent.simulate()
                coords.append(agent.coord)

            coords = np.array(coords)
            ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=30)
            ax.scatter(0, 0, color='green', s=100, marker='s', label='Старт')
            ax.scatter(self._goal[0], self._goal[1], color='red', s=150, marker='*', label='Цель')

            ax.set_xlim(-1, self._env_size + 1)
            ax.set_ylim(-1, self._env_size + 1)
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)

        plt.tight_layout()

    @staticmethod
    def save_experiment_info(config, fitness_history):
        info = f"""
        ЭКСПЕРИМЕНТ ГЕНЕТИЧЕСКОГО АЛГОРИТМА
        ====================================
        Дата: {datetime.now().strftime("%Y-%m-%d %H:%M")}
        Размер поля: {config.ENV_SIZE}
        Размер популяции: {config.AGENTS_CNT}
        Длина ДНК: {config.DNK_CNT}
        Цель: {config.GOAL}
        Мутация: {config.MUTATION_RATE}
        Поколений: {len(fitness_history)}

        РЕЗУЛЬТАТЫ:
        Начальный лучший фитнес: {max(fitness_history[0]):.2f}
        Финальный лучший фитнес: {max(fitness_history[-1]):.2f}
        Улучшение: {max(fitness_history[-1]) - max(fitness_history[0]):.2f}
        """
        with open(f"{PLOTS_DIR}/experiment_info.txt", "w", encoding="utf-8") as f:
            f.write(info)
