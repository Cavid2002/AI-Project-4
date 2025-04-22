import random
import matplotlib.pyplot as plt
from gridworld_env import GridWorldEnv
from double_q_learning import DoubleQLearning
from utils import extract_policy, evaluate_agent

if __name__ == "__main__":
    m = 5
    all_obstacles = [(1, 1), (2, 2), (3, 3), (1, 3), (4, 4)]
    for i in range(m):
        print(f"\n--- GridWorld Environment {i+1} ---")
        grid_size = random.choice([5, 7, 10])
        goal_state = (random.randint(0, grid_size - 1), random.randint(0, grid_size - 1))
        obstacles = random.sample([obs for obs in all_obstacles if obs != goal_state], 3)
        env = GridWorldEnv(grid_size=grid_size, goal_state=goal_state, stochastic=True, obstacles=obstacles)
        
        print("Initial Grid:")
        env.render()

        agent = DoubleQLearning(env)
        rewards = agent.train(episodes=5000)

        plt.plot(rewards)
        plt.title("Episode Rewards Over Time")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.grid(True)
        plt.show()

        policy = extract_policy(agent.Q1, env)
        print("Final Policy Grid:")
        env.render(policy=policy)
        evaluate_agent(env, policy)
