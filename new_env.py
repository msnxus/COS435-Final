import gymnasium as gym
from gymnasium import spaces
import numpy as np
import os
import torch
import random



# hyperparameters that define the city environment
# the number of nodes in the graph (possible locations between which the agent can travel)
# window of allowed times for edge evaluation

# we want a polynomial matrix indexed by (i, j) such that the poly_matrix[i, j] returns a 
# polynomial in time

# time_horizon denotes the time horizon of the environment/wrap-around period

class CityEnv(gym.Env):
    def __init__(self, poly_matrix, N, time_horizon, max_steps=50):
        super().__init__()
        self.num_steps = 0
        # Define the environment parameters
        self.max_steps = max_steps
        self.poly_matrix = poly_matrix
        self.n_vertices = N
        self.time_horizon = time_horizon

        # Define the current observation space
        self.destinations = None
        self.current_time = None
        self.current_vertex = None
    
        # Define action and observation space
        self.action_space = spaces.Discrete(self.n_vertices)
        self.observation_space = spaces.Tuple((
            spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32), # current time
            spaces.MultiBinary(self.n_vertices),  # vertices to visit
            spaces.Discrete(self.n_vertices)  # current vertex
        ))

    # Resets the agents current travel time
    # Randomly chooses the tensor that represents the traffic map
    def reset(self, *, seed=None, destinations, start_time, start_vertex):
        super().reset(seed=seed)
        self.destinations = destinations
        self.current_time = start_time
        self.current_vertex = start_vertex
        self.num_steps = 0

        observation = (
            (self.current_time % self.time_horizon).cpu().numpy(),
            self.destinations.cpu().numpy(),
            self.current_vertex.cpu().numpy()
        )
        info = {}
        return observation, info


    # The action is the i, j, k 
    def step(self, action):

        i = self.current_vertex
        j = action

        travel_time = self.poly_matrix[i, j].eval(self.current_time % self.time_horizon)
        
        # update the destinations
        self.destinations[j] = 0
        # updarte the current vertex
        self.current_vertex = j
        # update the current time
        self.current_time += travel_time
        observation = (
            (self.current_time % self.time_horizon).cpu().numpy(),
            self.destinations.cpu().numpy(),
            self.current_vertex.cpu().numpy()
        )

        self.num_steps += 1
        done = not self.destinations.any() or self.num_steps >= self.max_steps
        reward = -travel_time
        # consider adding auxiliary rewards for visiting new (needed) destinations
        info = {}
        return observation, reward, done, False, info

    def render(self):
        # Implement visualization if needed
        pass

    def close(self):
        pass

