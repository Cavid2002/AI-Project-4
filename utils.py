

def extract_policy(Q, env):
    policy = {}
    for x in range(env.grid_size):
        for y in range(env.grid_size):
            state = (x, y)
            if state in env.obstacles or state in env.terminal_states:
                continue
            q_values = {action: Q.get((state, action), 0) for action in env.get_possible_actions()}
            best_action = max(q_values, key=q_values.get)
            policy[state] = best_action
    return policy

def evaluate_agent(env, policy, episodes=100):
    success_count = 0
    total_steps = 0
    total_rewards = 0
    for episode in range(episodes):
        state = env.reset()
        done = False
        steps = 0
        total_reward = 0
        while not done:
            action = policy.get(state, 'up')
            next_state, reward, done = env.step(action)
            total_reward += reward
            steps += 1
            state = next_state
        total_rewards += total_reward
        total_steps += steps
        if state == env.goal_state:
            success_count += 1
    print("Evaluation Results:")
    print(f"Success Rate: {success_count / episodes * 100:.2f}%")
    print(f"Average Steps: {total_steps / episodes:.2f}")
    print(f"Average Reward: {total_rewards / episodes:.2f}")
