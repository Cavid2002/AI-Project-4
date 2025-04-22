import numpy as np
import random

# === Helper to display arrows for actions ===
def action_to_arrow(action):
    arrows = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'}
    return arrows.get(action, '.')

# === Environment ===
class GridWorldEnv:
    def __init__(self, grid_size, goal_state, terminal_states=None, stochastic=False, obstacles=None):
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.terminal_states = terminal_states if terminal_states else [goal_state]
        self.stochastic = stochastic
        self.obstacles = obstacles if obstacles else []
        self.reset()

    def reset(self):
        while True:
            self.agent_pos = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
            if self.agent_pos not in self.terminal_states and self.agent_pos not in self.obstacles:
                break
        return self.agent_pos

    def step(self, action):
        if self.stochastic and np.random.rand() < 0.2:
            action = random.choice(['up', 'down', 'left', 'right'])
        next_pos = self._move(self.agent_pos, action)
        if next_pos in self.obstacles:
            next_pos = self.agent_pos
        self.agent_pos = next_pos
        reward = self.get_reward(next_pos)
        done = next_pos in self.terminal_states
        return next_pos, reward, done

    def _move(self, position, action):
        x, y = position
        if action == 'up':
            x = max(0, x - 1)
        elif action == 'down':
            x = min(self.grid_size - 1, x + 1)
        elif action == 'left':
            y = max(0, y - 1)
        elif action == 'right':
            y = min(self.grid_size - 1, y + 1)
        return (x, y)

    def get_reward(self, position):
        return 1 if position == self.goal_state else -0.1

    def get_possible_actions(self):
        return ['up', 'down', 'left', 'right']

    def render(self, policy=None):
        grid = [['X' for _ in range(self.grid_size)] for _ in range(self.grid_size)]
        for terminal in self.terminal_states:
            grid[terminal[0]][terminal[1]] = 'G'
        for obs in self.obstacles:
            grid[obs[0]][obs[1]] = 'O'
        grid[self.agent_pos[0]][self.agent_pos[1]] = 'A'
        for row in grid:
            print(' '.join(row))
        if policy:
            print("\nPolicy:")
            for i in range(self.grid_size):
                print(" ".join(action_to_arrow(policy.get((i, j), '.')) for j in range(self.grid_size)))
        print()
