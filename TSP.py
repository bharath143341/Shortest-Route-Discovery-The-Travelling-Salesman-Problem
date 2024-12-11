import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.multiprocessing as mp
import matplotlib.pyplot as plt

# ============================
# TSP Environment
# ============================
class TSPEnvironment:
    def __init__(self, num_cities):
        self.num_cities = num_cities
        self.cities = np.random.rand(num_cities, 2)  # Randomly generate city coordinates
        self.reset()

    def reset(self):
        self.state = np.zeros(self.num_cities)
        return self.state

    def calculate_distance(self, city1, city2):
        return np.linalg.norm(self.cities[city1] - self.cities[city2])

    def step(self, current_city, action):
        distance = self.calculate_distance(current_city, action)
        reward = -distance
        self.state[action] = 1
        done = np.all(self.state)
        return reward, action, done


# ============================
# Q-Learning Implementation
# ============================
def initialize_q_values(num_cities):
    return np.zeros((num_cities, num_cities))

def select_action(q_values, state, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values[state]))  # Explore
    else:
        return np.argmax(q_values[state])  # Exploit

def update_q_values(q_values, state, action, next_state, reward, alpha, gamma):
    best_next_action = np.argmax(q_values[next_state])
    q_values[state, action] += alpha * (reward + gamma * q_values[next_state, best_next_action] - q_values[state, action])

def run_q_learning(env, num_cities, num_episodes, epsilon, alpha, gamma):
    q_values = initialize_q_values(num_cities)
    losses = []

    for episode in range(num_episodes):
        state = np.random.choice(num_cities)
        total_distance = 0

        for _ in range(num_cities - 1):
            action = select_action(q_values, state, epsilon)
            reward, next_state, done = env.step(state, action)
            total_distance += -reward
            update_q_values(q_values, state, action, next_state, reward, alpha, gamma)
            state = next_state

        # Return to the starting city
        total_distance += env.calculate_distance(state, 0)
        losses.append(total_distance)
        print(f"Q-Learning | Episode {episode + 1}, Total Distance: {total_distance:.4f}")

    return losses


# ============================
# A3C Implementation
# ============================
class ActorCriticNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(ActorCriticNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.policy_layer = nn.Linear(hidden_size, output_size)
        self.value_layer = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        policy = torch.softmax(self.policy_layer(x), dim=-1)
        value = self.value_layer(x)
        return policy, value

def worker(global_model, optimizer, env, input_size, gamma, global_ep, global_ep_max, lock, rewards_list):
    local_model = ActorCriticNetwork(input_size, hidden_size=32, output_size=env.num_cities)
    local_model.load_state_dict(global_model.state_dict())

    while True:
        with lock:
            if global_ep.value >= global_ep_max:
                break
            global_ep.value += 1
            episode = global_ep.value

        state = env.reset()
        log_probs, values, rewards = [], [], []
        current_city = np.random.choice(env.num_cities)
        total_reward, done = 0, False

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
            policy, value = local_model(state_tensor)
            action = torch.multinomial(policy, 1).item()

            log_probs.append(torch.log(policy[0, action]))
            values.append(value)
            reward, next_city, done = env.step(current_city, action)
            rewards.append(reward)
            total_reward += reward
            current_city = next_city

        # Compute returns
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns)
        values = torch.cat(values)
        advantages = returns - values

        # Loss
        policy_loss = -(torch.stack(log_probs) * advantages.detach()).mean()
        value_loss = advantages.pow(2).mean()
        loss = policy_loss + value_loss

        optimizer.zero_grad()
        loss.backward()
        for global_param, local_param in zip(global_model.parameters(), local_model.parameters()):
            global_param._grad = local_param.grad
        optimizer.step()
        local_model.load_state_dict(global_model.state_dict())

        # Append total reward to shared list
        rewards_list.append(-total_reward)  # Negative because rewards are negative distances

        print(f"A3C Worker {mp.current_process().name} | Episode {episode} | Total Distance: {-total_reward:.2f}")

def run_a3c(env, num_cities, global_ep_max, gamma):
    input_size = num_cities
    global_model = ActorCriticNetwork(input_size, hidden_size=32, output_size=num_cities)
    global_model.share_memory()
    optimizer = optim.Adam(global_model.parameters(), lr=0.001)

    global_ep = mp.Value('i', 0)
    lock = mp.Lock()
    manager = mp.Manager()
    rewards_list = manager.list()

    processes = []
    for _ in range(mp.cpu_count()):
        p = mp.Process(target=worker, args=(global_model, optimizer, env, input_size, gamma, global_ep, global_ep_max, lock, rewards_list))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    print("A3C Training Complete!")
    return list(rewards_list)


# ============================
# Main Execution
# ============================
if __name__ == "__main__":
    num_cities = 20
    num_episodes = 100  # Adjust for faster output
    epsilon = 0.1
    alpha = 0.1
    gamma = 0.99
    global_ep_max = 100

    env = TSPEnvironment(num_cities)

    # Run Q-Learning
    print("\nRunning Q-Learning...")
    q_losses = run_q_learning(env, num_cities, num_episodes, epsilon, alpha, gamma)

    # Plot Q-Learning results
    plt.figure(figsize=(10, 5))
    plt.plot(q_losses, label="Q-Learning")
    plt.xlabel("Episode")
    plt.ylabel("Total Distance")
    plt.title("Q-Learning Performance")
    plt.legend()
    plt.show()

    # Run A3C
    print("\nRunning A3C...")
    a3c_losses = run_a3c(env, num_cities, global_ep_max, gamma)

    # Plot A3C results
    plt.figure(figsize=(10, 5))
    plt.plot(a3c_losses, label="A3C")
    plt.xlabel("Episode")
    plt.ylabel("Total Distance")
    plt.title("A3C Performance")
    plt.legend()
    plt.show()
