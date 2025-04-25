# initialize environment given a tensor of data
# M = tensor[i][j][k] = travel_time
# axi (i,j,k): i->j at cummulative travel time k

# States: |K|-k travel maps where the current map shows the shortest distances between current i and all j, also cummulative travel time k
# Actions: which vertex to travel to j
# Reward: -M[i, action, k]


import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import torch
import random


class TrafficEnv(gym.Env):
    def __init__(self, trafficmap_dir, time_per_step=30, max_steps=50):
        super().__init__()
        self.trafficmap_dir = trafficmap_dir
        self.files = [f for f in os.listdir(trafficmap_dir)]
        file = os.path.join(trafficmap_dir,self.files[0])
        self.trafficmap_tensor = torch.load(file)

        self.max_steps = max_steps
        self.num_steps = 0

        self.n_vertices=self.trafficmap_tensor.shape[0]
        self.n_timesteps = self.trafficmap_tensor.shape[2]
        self.cum_time = 0
        self.cum_time_step = 0
        self.time_per_step = time_per_step
        self.current_vertex = None
        self.visited_vertices = torch.zeros(self.n_vertices, dtype=torch.bool)
        self.next_optimal_paths = None
        
        self.completion_reward = 10
        self.visited_weight = 10

        # self.inf = 10**9
        self.inf = 100
    
        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_vertices)
        self.observation_space = spaces.Tuple((
            # spaces.Box(low=0, high=self.inf, shape=(self.n_vertices, self.n_vertices, self.n_timesteps), dtype=np.float32),
            spaces.Box(low=0, high=self.inf, shape=(self.n_vertices,), dtype=np.float32),  # next_optimal_paths
            spaces.MultiBinary(self.n_vertices),  # visited_vertices
            spaces.Discrete(self.n_vertices)  # current_vertex
        ))


    # Resets the agents current travel time
    # Randomly chooses the tensor that represents the traffic map
    def reset(self, *, seed=None, ):
        super().reset(seed=seed)
        randomly_selected_file = random.choice(self.files)
        file_path = os.path.join(self.trafficmap_dir, randomly_selected_file)
        # self.trafficmap_tensor = torch.load(file_path)
        self.cum_time = 0
        self.cum_time_step = 0
        self.current_time_step = 0
        self.current_vertex = random.randrange(0, self.n_vertices)
        self.visited_vertices = torch.zeros(self.n_vertices, dtype=torch.bool)

        # Compute the shortest distances between current point and every other point in this time step
        # self.trafficmap_tensor[self.current_vertex, :, 0] = dijkstra(self.trafficmap_tensor[:,:,0], self.current_vertex)
        # self.next_optimal_paths = dijkstra(self.trafficmap_tensor[:,:,0], self.current_vertex)
        self.next_optimal_paths = self.trafficmap_tensor[self.current_vertex,:,0]

        observation = (
            self.trafficmap_tensor.cpu().numpy(),
            self.next_optimal_paths.cpu().numpy(),
            self.visited_vertices.cpu().numpy(),
            self.current_vertex
        )

        

        # observation = (self.trafficmap_tensor.clone().float(), self.next_optimal_paths.clone().float())
        info = {}
        self.visited_vertices[self.current_vertex] = 1
        self.num_steps = 0
        return observation, info


    # The action is the i, j, k 
    def step(self, action):

        i = self.current_vertex
        j = action
        k = 0

        # travel_time = self.trafficmap_tensor[i, j, k]
        travel_time = self.next_optimal_paths[j]
        
        self.cum_time += travel_time
        reward = -travel_time
        

        num_shifts = int(self.cum_time // self.time_per_step)
        self.current_time_step += num_shifts

        self.cum_time = self.cum_time % self.time_per_step

        # Create the next state
        if num_shifts > 0:
            self.trafficmap_tensor = torch.roll(self.trafficmap_tensor, shifts=-num_shifts, dims=2)
            self.trafficmap_tensor[:, :, -num_shifts:] = 0


        
        
        # Compute the shortest distances between current point and every other point in this time step
        # self.trafficmap_tensor[self.current_vertex, :, 0] = dijkstra(self.trafficmap_tensor[:,:,0], self.current_vertex)
        # self.next_optimal_paths = dijkstra(self.trafficmap_tensor[:,:,0], self.current_vertex)
        self.next_optimal_paths = self.trafficmap_tensor[self.current_vertex,:,0]
        
        
        # self.trafficmap_tensor[i,:, :] = self.inf
        self.trafficmap_tensor[:,i, :] = self.inf
        
        
        
        observation = (
            self.trafficmap_tensor.cpu().numpy(),
            self.next_optimal_paths.cpu().numpy(),
            self.visited_vertices.cpu().numpy(),
            self.current_vertex
        )

        

        
        self.current_vertex = j
        self.visited_vertices[self.current_vertex] = 1
        self.num_steps += 1
        num_visited= self.visited_vertices.sum()
        done = num_visited >= self.n_vertices or self.num_steps >= self.max_steps
        reward += num_visited*self.visited_weight
        info = {}
        return observation, reward, done, False, info

    def render(self):
        # Implement visualization if needed
        pass

    def close(self):
        pass


# Shortest All-Paths Algorithm
def floyd_warshall(adj):
    n = adj.size(0)
    dist = adj.clone()

    for k in range(n):
        dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))

    return dist


# Dijstra's
def dijkstra(adj, start):
    LARGE = 1e9
    n = adj.size(0)
    visited = torch.zeros(n, dtype=torch.bool)
    dist = torch.full((n,), LARGE)
    dist[start] = 0

    for _ in range(n):
        # Find the unvisited node with the smallest distance
        min_dist = LARGE
        u = -1
        for v in range(n):
            if not visited[v] and dist[v] < min_dist:
                min_dist = dist[v]
                u = v

        if u == -1:
            break  # Remaining vertices are unreachable

        visited[u] = True

        # Update distances to neighbors of u
        for v in range(n):
            if not visited[v] and adj[u, v] != float('inf'):
                if dist[u] + adj[u, v] < dist[v]:
                    dist[v] = dist[u] + adj[u, v]
    dist[start] = LARGE #not allowed to revisit
    return dist