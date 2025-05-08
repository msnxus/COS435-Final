import gymnasium as gym
from gymnasium import spaces
import numpy as np


class sb3_CityEnv(gym.Env):
    def __init__(self, poly_matrix, N, time_horizon, max_steps=50):
        super().__init__()
        self.poly_matrix = poly_matrix
        self.n_vertices = N
        self.time_horizon = time_horizon
        self.max_steps = max_steps

        # State variables
        self.destinations = None
        self.current_time = None
        self.current_vertex = None
        self.num_steps = 0

        # Action space: move to one of the vertices
        self.action_space = spaces.Discrete(self.n_vertices)

        # Observation space: [current_time, destinations (binary vector), current_vertex (one-hot or int)]
        # We'll flatten everything into a 1D array of shape (1 + N + 1,)
        obs_dim = 1 + self.n_vertices + 1
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(obs_dim,), dtype=np.float32)

    def _get_obs(self):
        time_obs = np.array([self.current_time % self.time_horizon], dtype=np.float32)
        dest_obs = self.destinations.astype(np.float32)
        vertex_obs = np.array([self.current_vertex], dtype=np.float32)
        return np.concatenate([time_obs, dest_obs, vertex_obs], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)

        # Support seeding and Gymnasium options
        options = options or {}

        self.destinations = options.get("destinations", np.random.choice([0, 1], size=self.n_vertices))
        self.current_time = options.get("start_time", 0)
        self.current_vertex = options.get("start_vertex", 0)
        self.num_steps = 0
        #print(self.destinations)
        return self._get_obs(), {}

    def step(self, action):
        i = self.current_vertex
        j = action
        reward = 0.0

        if i == j:
            reward -= 5.0

        travel_time = self.poly_matrix[i, j].eval(self.current_time % self.time_horizon)
        self.current_time += travel_time

        if self.destinations[j] > 0:
            reward += 1.0
            if self.destinations.sum() == 1:
                reward += 10.0
        self.destinations[j] = 0  # Mark destination as visited

        self.current_vertex = j
        self.num_steps += 1
        reward -= travel_time

        done = not self.destinations.any()
        truncated = self.num_steps >= self.max_steps

        return self._get_obs(), reward, done, truncated, {}

    def render(self):
        print(f"Time: {self.current_time}, Vertex: {self.current_vertex}, Destinations: {self.destinations}")

    def close(self):
        pass
