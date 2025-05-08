
import torch
from torch import nn
from new_env import CityEnv
from poly_matrix import create_poly_matrix
from torch.distributions import Categorical
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import torch.nn.functional as F

import seaborn as sns
sns.set(style="darkgrid")

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

N = 20
time_horizon = 24
max_steps = 200
M = 10
destinations = np.array([i%2 for i in range(20)])
start_vertex = 1
start_time = 0


def unpack_obs(obs):
    current_time, dests, current_vertex = obs
    current_vertex_onehot = np.zeros(N)
    current_vertex_onehot[current_vertex] = 1
    flat_obs = np.concatenate(([current_time], dests.flatten(), current_vertex_onehot))
    state = torch.tensor(flat_obs, dtype=torch.float).unsqueeze(0)
    return state
    


def evaluate_policy(policy, poly_matrix, seed=42):
    env_test = CityEnv(poly_matrix, N=N, time_horizon=time_horizon, max_steps=max_steps)
    obs, _ = env_test.reset(destinations=destinations, start_time=start_time, start_vertex=start_vertex)
    state = unpack_obs(obs)

    done = False
    total_reward = 0
    actions = []
    for i in range(env_test.max_steps):
        dist = policy(state)
        action = dist.sample().item()
        actions.append(action)
        obs, reward, done, _, _ = env_test.step(action)
        next_state = unpack_obs(obs)
        state = next_state
        total_reward += reward
        if done:
            break
    print(actions)
    return total_reward


class PolicyNetwork(nn.Module):
  def __init__(self, N, hidden_dim=128):
      input_dim = 2*N + 1
      output_dim = N
      super(PolicyNetwork, self).__init__()
      self.f1 = nn.Linear(input_dim, hidden_dim)
      self.f2 = nn.Linear(hidden_dim, hidden_dim)
      self.f3 = nn.Linear(hidden_dim, hidden_dim)
      self.f4 = nn.Linear(hidden_dim, output_dim)
      self.relu = torch.relu

  def forward(self, state):
    h1 = self.relu(self.f1(state))
    h2 = self.relu(self.f2(h1))
    h3 = self.relu(self.f3(h2))
    logits = self.f4(h3)
    return Categorical(logits=logits)


class ValueNetwork(nn.Module):

  def __init__(self, N, hidden_dim=128):
      super(ValueNetwork, self).__init__()
      input_dim = 2*N + 1
      output_dim = 1
      self.f1 = nn.Linear(input_dim, hidden_dim)
      self.f2 = nn.Linear(hidden_dim, hidden_dim)
      self.f3 = nn.Linear(hidden_dim, hidden_dim)
      self.f4 = nn.Linear(hidden_dim, output_dim)
      self.relu = torch.relu

  def forward(self, state):
    h1 = self.relu(self.f1(state))
    h2 = self.relu(self.f2(h1))
    h3 = self.relu(self.f3(h2))
    logit = self.f4(h3)
    return logit


def compute_advantages(next_value, rewards, masks, values, gamma=0.99, lambda_gae=0.95):
    values = values + [next_value]  # Append bootstrap value for last state
    advantages = 0
    returns = []

    for step in reversed(range(len(rewards))):  # Iterate in reverse (backward pass)
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]

        ##### Code implementation here #####
        advantages = delta + gamma * lambda_gae * advantages

        returns.insert(0, advantages + values[step])

    return returns


def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
    batch_size = states.size(0)
    mini_batches = []

    for _ in range(batch_size // mini_batch_size):
        rand_ids = torch.randint(0, batch_size, (mini_batch_size,))
        mini_batch = states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        mini_batches.append(mini_batch)

    return mini_batches



def ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):

    for _ in range(ppo_epochs):
        for states, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
            dist = policy_net(states)
            new_log_probs = dist.log_prob(action)


            r = (new_log_probs - old_log_probs).exp()
            clipped = torch.clamp(r , 1-clip_param, 1+clip_param)*advantage
            unclipped = r*advantage
            actor_loss = -torch.min(clipped, unclipped).mean()

            critic_loss = (value_net(states).squeeze(1) - return_).pow(2).mean()

            print(actor_loss, critic_loss)

            loss = 0.05 * critic_loss + actor_loss  # You can freely adjust the weight of the critic loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




def train(num_steps=1000, mini_batch_size=8, ppo_epochs=4, threshold=400):
    with open('output1.csv', 'w') as f:
        f.write('step,reward\n')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    polymatrix = create_poly_matrix(N=N, time_horizon=time_horizon)
    env = CityEnv(polymatrix, N=N, time_horizon=time_horizon, max_steps=max_steps)
    num_vertices = env.n_vertices

    policy_net = PolicyNetwork(N, hidden_dim=64).float()
    value_net  = ValueNetwork(N, hidden_dim=64).float()

    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-3)

    obs, _ = env.reset(destinations=destinations, start_time=start_time, start_vertex=start_vertex)

    state = unpack_obs(obs)


    early_stop = False
    reward_list = []

    for step in range(num_steps):
        log_probs = []
        values = []
        states = []
        actions = []
        rewards = []
        masks = []

        # Collect samples under the current policy
        for _ in range(2048):
            
            dist, value = policy_net(state), value_net(state)

            action = dist.sample()
            obs, reward, done, _, _ = env.step(int(action.item()))  # Ensure action is an int

            next_state = unpack_obs(obs)


            log_prob = dist.log_prob(action)

            log_probs.append(log_prob.unsqueeze(0))
            values.append(value.unsqueeze(0))
            rewards.append(torch.tensor([reward], dtype=torch.float32))
            masks.append(torch.tensor([1 - done], dtype=torch.float32))
            states.append(state.unsqueeze(0))
            actions.append(action.unsqueeze(0))  # Fix for actions


            state = next_state
            if done:
                obs, _ = env.reset(destinations=destinations, start_time=start_time, start_vertex=start_vertex) 
                state = unpack_obs(obs)

        next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0)  # Ensure proper conversion
        next_value = value_net(next_state)
        returns = compute_advantages(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values

        # Run PPO update for policy and value networks
        ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

        if step % 10 == 0:
            test_reward = np.mean([evaluate_policy(policy_net, poly_matrix=polymatrix) for _ in range(10)])
            print(f'Step: {step}\tReward: {test_reward}')
            with open('output1.csv', 'a') as f:
                f.write(f'{step},{test_reward}\n')
            reward_list.append(test_reward)
            if test_reward > threshold:
                print("Solved!")
                early_stop = True
                break
    return early_stop, reward_list


if __name__ == '__main__':
    early_stop, reward_list = train(num_steps=100, mini_batch_size=16, ppo_epochs=4, threshold=0)