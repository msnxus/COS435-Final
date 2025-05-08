import torch
from torch import nn
import numpy as np
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F
import seaborn as sns
import warnings

from new_env import CityEnv
from poly_matrix import create_poly_matrix

sns.set(style="darkgrid")
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Define device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Parameters
N = 20
time_horizon = 24
max_steps = 200
destinations = np.array([i % 2 for i in range(20)])
start_vertex = 1
start_time = 0

def unpack_obs(obs):
    current_time, dests, current_vertex = obs
    current_vertex_onehot = np.zeros(N)
    current_vertex_onehot[current_vertex] = 1
    flat_obs = np.concatenate(([current_time], dests.flatten(), current_vertex_onehot))
    state = torch.tensor(flat_obs, dtype=torch.float, device=device).unsqueeze(0)
    return state

def evaluate_policy(policy, poly_matrix, seed=42):
    env_test = CityEnv(poly_matrix, N=N, time_horizon=time_horizon, max_steps=max_steps)
    obs, _ = env_test.reset(destinations=destinations, start_time=start_time, start_vertex=start_vertex)
    state = unpack_obs(obs)
    done = False
    total_reward = 0
    actions = []
    for _ in range(env_test.max_steps):
        dist = policy(state)
        action = dist.sample().item()
        actions.append(action)
        obs, reward, done, _, _ = env_test.step(action)
        state = unpack_obs(obs)
        total_reward += reward
        if done:
            break
    print(actions)
    return total_reward

class PolicyNetwork(nn.Module):
    def __init__(self, N, hidden_dim=128):
        super().__init__()
        input_dim = 2 * N + 1
        output_dim = N
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        self.f4 = nn.Linear(hidden_dim, output_dim)

    def forward(self, state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        logits = self.f4(x)
        return Categorical(logits=logits)

class ValueNetwork(nn.Module):
    def __init__(self, N, hidden_dim=128):
        super().__init__()
        input_dim = 2 * N + 1
        self.f1 = nn.Linear(input_dim, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        self.f4 = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = F.relu(self.f1(state))
        x = F.relu(self.f2(x))
        x = F.relu(self.f3(x))
        return self.f4(x)

def compute_advantages(next_value, rewards, masks, values, gamma=0.99, lambda_gae=0.95):
    values = values + [next_value]
    advantages = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        advantages = delta + gamma * lambda_gae * advantages
        returns.insert(0, advantages + values[step])
    return returns

def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
    batch_size = states.size(0)
    for _ in range(batch_size // mini_batch_size):
        idx = torch.randint(0, batch_size, (mini_batch_size,), device=device)
        yield states[idx], actions[idx], log_probs[idx], returns[idx], advantages[idx]

def ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size,
               states, actions, log_probs, returns, advantages, clip_param=0.2):
    for _ in range(ppo_epochs):
        for state_b, action_b, old_log_b, return_b, adv_b in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist = policy_net(state_b)
            new_log_prob = dist.log_prob(action_b)
            ratio = (new_log_prob - old_log_b).exp()
            surr1 = ratio * adv_b
            surr2 = torch.clamp(ratio, 1 - clip_param, 1 + clip_param) * adv_b
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (value_net(state_b).squeeze(1) - return_b).pow(2).mean()
            loss = 0.5 * critic_loss + actor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def train(num_steps=1000, mini_batch_size=8, ppo_epochs=4, threshold=400):
    with open('output.csv', 'w') as f:
        f.write('step,reward\n')

    polymatrix = create_poly_matrix(N=N, time_horizon=time_horizon)
    env = CityEnv(polymatrix, N=N, time_horizon=time_horizon, max_steps=max_steps)

    policy_net = PolicyNetwork(N, hidden_dim=64).to(device)
    value_net  = ValueNetwork(N, hidden_dim=64).to(device)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-3)

    obs, _ = env.reset(destinations=destinations, start_time=start_time, start_vertex=start_vertex)
    state = unpack_obs(obs)

    reward_list = []
    for step in range(num_steps):
        log_probs, values, states, actions, rewards, masks = [], [], [], [], [], []

        for _ in range(2048):
            dist, value = policy_net(state), value_net(state)
            action = dist.sample()
            obs, reward, done, _, _ = env.step(int(action.item()))
            next_state = unpack_obs(obs)

            log_probs.append(dist.log_prob(action).unsqueeze(0))
            values.append(value.detach())
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
            masks.append(torch.tensor([1 - done], dtype=torch.float32, device=device))
            states.append(state)
            actions.append(action.unsqueeze(0))

            state = next_state
            if done:
                obs, _ = env.reset(destinations=destinations, start_time=start_time, start_vertex=start_vertex)
                state = unpack_obs(obs)

        with torch.no_grad():
            next_value = value_net(state)

        returns = compute_advantages(next_value, rewards, masks, values)
        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach().squeeze(1)
        states = torch.cat(states)
        actions = torch.cat(actions).squeeze(1)
        advantages = returns - values

        ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages)

        if step % 10 == 0:
            test_reward = np.mean([evaluate_policy(policy_net, poly_matrix=polymatrix) for _ in range(10)])
            print(f'Step: {step}\tReward: {test_reward}')
            with open('output.csv', 'a') as f:
                f.write(f'{step},{test_reward}\n')
            reward_list.append(test_reward)
            if test_reward > threshold:
                print("Solved!")
                break
    return reward_list

if __name__ == '__main__':
    rewards = train(num_steps=100, mini_batch_size=16, ppo_epochs=4, threshold=0)
