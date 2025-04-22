import random
from collections import deque

class DoubleQLearning:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1, decay=0.999, replay_buffer_size=10000, batch_size=32):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.decay = decay
        self.replay_buffer = deque(maxlen=replay_buffer_size)
        self.batch_size = batch_size
        self.Q1 = {}
        self.Q2 = {}

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.env.get_possible_actions())
        else:
            q_values_1 = [self.Q1.get((state, action), 0) for action in self.env.get_possible_actions()]
            q_values_2 = [self.Q2.get((state, action), 0) for action in self.env.get_possible_actions()]
            max_q_value = [q1 + q2 for q1, q2 in zip(q_values_1, q_values_2)]
            max_actions = [action for action, value in zip(self.env.get_possible_actions(), max_q_value) if value == max(max_q_value)]
            return random.choice(max_actions)

    def update_q_values(self, state, action, reward, next_state, done):
        next_actions = [self.select_action(next_state) for _ in range(2)]
        next_action = next_actions[0] if random.random() > 0.5 else next_actions[1]
        if random.random() > 0.5:
            best_next_action_q = self.Q1.get((next_state, next_action), 0)
            q_value = self.Q1.get((state, action), 0) + self.alpha * (reward + self.gamma * best_next_action_q - self.Q1.get((state, action), 0))
            self.Q1[(state, action)] = q_value
        else:
            best_next_action_q = self.Q2.get((next_state, next_action), 0)
            q_value = self.Q2.get((state, action), 0) + self.alpha * (reward + self.gamma * best_next_action_q - self.Q2.get((state, action), 0))
            self.Q2[(state, action)] = q_value

    def store_experience(self, state, action, reward, next_state, done):
        td_error = reward + self.gamma * max(self.Q1.get((next_state, a), 0) for a in self.env.get_possible_actions()) - self.Q1.get((state, action), 0)
        self.replay_buffer.append((state, action, reward, next_state, done, td_error))

    def sample_batch(self):
        sorted_buffer = sorted(self.replay_buffer, key=lambda x: abs(x[5]), reverse=True)
        batch = sorted_buffer[:self.batch_size]
        return [(s, a, r, ns, d) for s, a, r, ns, d, _ in batch]

    def train(self, episodes=5000):
        rewards = []
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            done = False
            while not done:
                action = self.select_action(state)
                next_state, reward, done = self.env.step(action)
                total_reward += reward
                self.store_experience(state, action, reward, next_state, done)
                self.update_q_values(state, action, reward, next_state, done)
                state = next_state
            if len(self.replay_buffer) > self.batch_size:
                batch = self.sample_batch()
                for state, action, reward, next_state, done in batch:
                    self.update_q_values(state, action, reward, next_state, done)
            self.epsilon *= self.decay
            rewards.append(total_reward)
        return rewards
