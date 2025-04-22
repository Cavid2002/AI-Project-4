import requests
import pickle
import time
import os
import random
import api

class GridworldAgent:
    def __init__(self, team_id, q_table_path="q_table.pkl"):
        self.team_id = str(team_id)
        self.q_table_path = q_table_path

        self.q_table = self.load_q_table()

        # self.actions = ["up", "down", "left", "right"]
        self.actions = ["N", "S", "W", "E"]
        self.alpha = 0.1
        self.gamma = 0.9
        self.epsilon = 0.2

        self.last_enter_time = 0
        self.run_id = None
        self.world_id = None

    def load_q_table(self):
        if os.path.exists(self.q_table_path):
            with open(self.q_table_path, "rb") as f:
                print("[INFO] Loaded Q-table from disk.")
                return pickle.load(f)
        else:
            print("[INFO] No Q-table found. Creating new one.")
            return [[0 for _ in range(4)] for _ in range(40 * 40)]

    def save_q_table(self):
        with open(self.q_table_path, "wb") as f:
            pickle.dump(self.q_table, f)
            print("[INFO] Q-table saved to disk.")

    
    def state_to_index(self, state):
        if state is None or ":" not in state:
            return None
        x, y = map(int, state.split(":"))
        return x * 40 + y

    def choose_action(self, state_idx):
        if random.random() < self.epsilon:
            return random.randint(0, 3)  # Explore
        return max(range(4), key=lambda a: self.q_table[state_idx][a])  # Exploit

    def learn(self, world_id, max_steps=100):
        world, state = api.get_location(self.team_id)

        if world != str(world_id):
            state = api.enter_world(self.team_id, world_id)
            if state is None:
                return

        for step in range(max_steps):
            state_idx = self.state_to_index(state)
            if state_idx is None:
                print("[WARN] Invalid state index.")
                break

            action_idx = self.choose_action(state_idx)
            move = self.actions[action_idx]

            reward, new_state = api.make_move(self.team_id, move, world_id)
            time.sleep(15)  # Respect move delay

            if reward is None or new_state is None:
                print("[WARN] Move failed, skipping step.")
                break

            new_state_idx = self.state_to_index(new_state)
            if new_state_idx is not None:
                old_q = self.q_table[state_idx][action_idx]
                max_future_q = max(self.q_table[new_state_idx])
                self.q_table[state_idx][action_idx] = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * max_future_q)

            state = new_state

        self.save_q_table()


# ---------- Usage Example ----------

if __name__ == "__main__":
    TEAM_ID = "1459"

    # agent = GridworldAgent(TEAM_ID)

    # agent.learn(world_id=0, max_steps=100)
    # api.get_runs(TEAM_ID, 5)
    # api.get_score(TEAM_ID)

    api.get_location(TEAM_ID)
    # api.enter_world(TEAM_ID, "0")
    # api.get_runs(TEAM_ID, 1)
    api.make_move(TEAM_ID, "N", 0)
    # for world in range(10):  # World 0 to 9
    #     print(f"\n=== Learning in World {world} ===")
    #     agent.learn(world_id=world, max_steps=100)
    #     agent.get_runs()
    #     agent.get_score()