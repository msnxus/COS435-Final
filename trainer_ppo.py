import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
from model import TrafficNet
from simulator import TrafficEnv
import numpy as np

def print_state(obs, n):
    traffic_tensor, dijkstra_tensor = obs
    for t in range(n):
        print('-------------------------')
        print('t =', t)
        print(traffic_tensor[:, :, t])


def compute_returns(rewards, gamma, last_value):
    returns = []
    R = last_value
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return returns


def train_ppo(
    traffic_dir,
    total_episodes=500,
    gamma=0.99,
    clip_epsilon=0.2,
    update_epochs=4,
    batch_size=5,
    lr=2.5e-4
):
    env = TrafficEnv(traffic_dir)
    model = TrafficNet(env.n_vertices, env.n_timesteps)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for episode in range(total_episodes):
        traffic_tensor, next_paths = env.reset()[0]
        
        traffic_tensor = torch.tensor(traffic_tensor, dtype=torch.float32)
        next_paths = torch.tensor(next_paths, dtype=torch.float32)

        log_probs = []
        values = []
        rewards = []
        states = []
        actions = []

        done = False
        while not done:
            logits, value = model(traffic_tensor, next_paths)
            dist = Categorical(logits=logits)
            action = dist.sample()

            (next_traffic, next_paths), reward, done, _, _ = env.step(action.item())
            next_traffic = torch.tensor(next_traffic, dtype=torch.float32)
            next_paths = torch.tensor(next_paths, dtype=torch.float32)

            log_probs.append(dist.log_prob(action))
            values.append(value)
            rewards.append(torch.tensor(reward, dtype=torch.float32))
            actions.append(action)
            states.append((traffic_tensor, next_paths))

            traffic_tensor = next_traffic

        with torch.no_grad():
            _, last_value = model(traffic_tensor, next_paths)
        returns = compute_returns(rewards, gamma, last_value)

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        returns = torch.stack(returns)
        advantages = returns - values.detach()

        for _ in range(update_epochs):
            for i in range(0, len(states), batch_size):
                batch = states[i:i + batch_size]
                batch_actions = actions[i:i + batch_size]
                batch_advantages = advantages[i:i + batch_size]
                batch_returns = returns[i:i + batch_size]
                batch_log_probs = log_probs[i:i + batch_size]

                new_log_probs = []
                new_values = []

                for (traffic_tensor, next_paths), action in zip(batch, batch_actions):
                    logits, value = model(traffic_tensor, next_paths)
                    dist = Categorical(logits=logits)
                    new_log_probs.append(dist.log_prob(action))
                    new_values.append(value)

                new_log_probs = torch.stack(new_log_probs)
                new_values = torch.stack(new_values)
                ratio = (new_log_probs - batch_log_probs).exp()

                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = nn.functional.mse_loss(new_values, batch_returns)

                loss = policy_loss + 0.5 * value_loss
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        print(f"Episode {episode+1}/{total_episodes}, Total reward: {sum(rewards).item():.2f}")
