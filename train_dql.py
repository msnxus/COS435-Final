
import torch
from torch import nn
from simulator2 import TrafficEnv
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



def unpack_obs(obs, env, device):
    # traffic_tensor = torch.tensor(obs[0], dtype=torch.float32, device=device)
    next_paths = torch.tensor(obs[0], dtype=torch.float32, device=device)
    visited = torch.tensor(obs[1], dtype=torch.float32, device=device)
    current_vertex = torch.tensor([obs[2]], dtype=torch.long, device=device)
    current_vertex_onehot = F.one_hot(torch.tensor(current_vertex), num_classes=env.n_vertices).float().to(device).squeeze(0)
    state = torch.cat([next_paths, visited, current_vertex_onehot], dim=0)
    state = state.to(device)
    return state

def print_next_paths(next_paths):
    print(next_paths.shape)
    for val in next_paths:
        print(val, end=' ')

def evaluate_policy(policy, seed=42):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env_test = TrafficEnv(trafficmap_dir='traffic_maps2', time_per_step=2000, max_steps=8)
    
    obs, _ = env_test.reset()
 
    state = unpack_obs(obs, env_test, device)
    done = False
    total_reward = torch.zeros(1, device=device).unsqueeze(0)
    actions = []
    for i in range(env_test.max_steps):
        state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)

        dist = policy(state)
        action = dist.sample().item()
        # action = torch.argmax(dist.probs, dim=-1).item()
    
        actions.append(action)
        obs, reward, done, _, _ = env_test.step(action)
        visited = obs[1]
   
        
        next_state = unpack_obs(obs, env_test, device)

        state = next_state
        
        print(reward, end=' ')
        total_reward += reward
        if done:
            break
    print()
    print(actions)
    return total_reward

class PolicyNetwork(nn.Module):
    def __init__(self, n_V, n_T, hidden_dim=128):
        self.V = n_V
        self.T = n_T
        super(PolicyNetwork, self).__init__()
        self.f1 = nn.Linear(3*self.V, hidden_dim)
        self.f2 = nn.Linear(hidden_dim, hidden_dim)
        self.f3 = nn.Linear(hidden_dim, hidden_dim)
        self.f4 = nn.Linear(hidden_dim, self.V)
        self.relu = torch.relu

    def forward(self, state):
        h1 = self.relu(self.f1(state))
        h2 = self.relu(self.f2(h1))
        h3 = self.relu(self.f3(h2))
        logits = self.f4(h3)

        # Extract visited part of the state
        if state.dim() == 1:
            state = state.unsqueeze(0)
        visited = state[:,self.V:self.V*2]

        # Mask visited nodes with a large negative number so they are not chosen
        mask = (visited > 0).float()
        logits = logits.masked_fill(mask.bool(), -1e10)

        return Categorical(logits=logits)



class ValueNetwork(nn.Module):

  def __init__(self, n_V, n_T, hidden_dim=128):
      super(ValueNetwork, self).__init__()
      self.V = n_V
      self.T = n_T
      self.f1 = nn.Linear(3*self.V, hidden_dim)
      self.f2 = nn.Linear(hidden_dim, hidden_dim)
      self.f3 = nn.Linear(hidden_dim, hidden_dim)
      self.f4 = nn.Linear(hidden_dim, 1)
      self.relu = torch.relu

  def forward(self, state):
    h1 = self.relu(self.f1(state))
    h2 = self.relu(self.f2(h1))
    h3 = self.relu(self.f3(h2))
    logit = self.f4(h3)
    return logit


def compute_advantages(next_value, rewards, masks, values, gamma=0.99, lambda_gae=0.95):
    values = values + [next_value]  # Append bootstrap value for last state
    advantages = []
    gae = 0

    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lambda_gae * masks[step] * gae
        advantages.insert(0, gae)

    returns = [adv + val for adv, val in zip(advantages, values[:-1])]
    return returns



def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage, device):
    batch_size = states.size(0)
    mini_batches = []

    for _ in range(batch_size // mini_batch_size):
        rand_ids = torch.randint(0, batch_size, (mini_batch_size,), device=device)
        mini_batch = states[rand_ids], actions[rand_ids], log_probs[rand_ids], returns[rand_ids], advantage[rand_ids]
        mini_batches.append(mini_batch)

    return mini_batches



def ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, device, clip_param=0.2):

    for _ in range(ppo_epochs):
        for states, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages, device):
            dist = policy_net(states)
            new_log_probs = dist.log_prob(action)


            r = (new_log_probs - old_log_probs).exp()
            clipped = torch.clamp(r , 1-clip_param, 1+clip_param)*advantage
            unclipped = r*advantage
            actor_loss = -torch.min(clipped, unclipped).mean()

            critic_loss = (value_net(states) - return_).pow(2).mean()
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.1*entropy


            # loss = 0.5 * critic_loss + actor_loss  # You can freely adjust the weight of the critic loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()




def train(num_steps=1000, mini_batch_size=8, ppo_epochs=4, threshold=400):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = TrafficEnv(trafficmap_dir='traffic_maps2', time_per_step=2000, max_steps=8)
    num_vertices = env.n_vertices
    num_time_steps = env.n_timesteps

    policy_net = PolicyNetwork(n_T=num_time_steps, n_V=num_vertices, hidden_dim=64).to(device)
    value_net = ValueNetwork(n_T=num_time_steps, n_V=num_vertices, hidden_dim=64).to(device)
    optimizer = optim.Adam(list(policy_net.parameters()) + list(value_net.parameters()), lr=3e-4)

    obs, _ = env.reset()
    state = unpack_obs(obs, env, device)


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
        for _ in range(500):
            dist, value = policy_net(state), value_net(state)

            action = dist.sample()
            # action = torch.argmax(dist.probs, dim=-1)
            action = action.to(device)
            obs, reward, done, _, _ = env.step(int(action.item()))  # Ensure action is an int
            next_state = unpack_obs(obs, env, device)


            log_prob = dist.log_prob(action)

            log_probs.append(log_prob.unsqueeze(0))
            values.append(value.unsqueeze(0))
            rewards.append(torch.tensor([reward], dtype=torch.float32, device=device))
            masks.append(torch.tensor([1 - int(done)], dtype=torch.float32, device=device))
            states.append(state.unsqueeze(0))
            actions.append(action.unsqueeze(0))  # Fix for actions


            state = next_state
            if done:
                obs, _ = env.reset()  # Ensure proper Gym reset handling
                state = unpack_obs(obs, env, device)


        # next_state = torch.tensor(np.array(next_state), dtype=torch.float32).unsqueeze(0)  # Ensure proper conversion
        # next_value = value_net(next_state)
        # REPLACED ABOVE CODE
        with torch.no_grad():
            # Use the current state after the rollout ends
            next_value = value_net(state) if not done else torch.tensor([0.0], device=device)

        returns = compute_advantages(next_value, rewards, masks, values)

        returns = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values = torch.cat(values).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantage = returns - values
        # advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
        # advantage = advantage.detach()


        # Run PPO update for policy and value networks
        ppo_update(policy_net, value_net, optimizer, ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage, device)

        if step % 5 == 0:
            test_reward = np.mean([evaluate_policy(policy_net).item() for _ in range(10)])

            print(f'Step: {step}\tReward: {test_reward}')
            reward_list.append(test_reward)
            with open('output2.csv', 'a') as f:
                f.write(f"{step},{test_reward}\n")
    return early_stop, reward_list


if __name__ == '__main__':
    from trainer_ppo_class import train
    import os
    with open('output2.csv', 'w') as f:
        f.write('step, reward\n')
    early_stop, reward_list = train(num_steps=10000, mini_batch_size=16, ppo_epochs=4, threshold=0)