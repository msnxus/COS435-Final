import torch
import numpy as np
import os
import tempfile
from simulator import TrafficEnv  # Replace with your actual file/module name

def generate_dummy_tensor(n_vertices=4, n_timesteps=3):
    # tensor = torch.randint(0, 80, (n_vertices, n_vertices, n_timesteps), dtype=torch.float32)
    tensor = torch.zeros((n_vertices, n_vertices, n_timesteps), dtype=torch.float32)
    for i in range(n_vertices):
        tensor[i, i, :] = float('inf')  # No self-loops
    return tensor

def save_dummy_tensor_to_dir(directory, filename="dummy.pt"):
    tensor = generate_dummy_tensor()
    path = os.path.join(directory, filename)
    torch.save(tensor, path)
    return path

def print_state(obs, n):
    traffic_tensor, dijkstra_tensor = obs
    for t in range(n):
        print('-------------------------')
        print('t =', t)
        print(traffic_tensor[:, :, t])
    print('Dijkstra:', dijkstra_tensor)

def test_traffic_env_behavior():
    with tempfile.TemporaryDirectory() as tmp_dir:
        save_dummy_tensor_to_dir(tmp_dir)
        env = TrafficEnv(trafficmap_dir=tmp_dir)
        
        obs, info = env.reset()
        print_state(obs, env.n_timesteps)

        # Check that observation is a tuple of correct shape
        assert isinstance(obs, tuple), "Observation should be a tuple"
        assert len(obs) == 2, "Observation should have 2 elements"
        traffic_tensor, dijkstra_tensor = obs
        assert isinstance(traffic_tensor, torch.Tensor)
        assert isinstance(dijkstra_tensor, torch.Tensor)
        assert traffic_tensor.shape == (env.n_vertices, env.n_vertices, env.n_timesteps)
        assert dijkstra_tensor.shape == (env.n_vertices,)
        assert traffic_tensor.dtype == torch.float32
        assert dijkstra_tensor.dtype == torch.float32

        visited = set([env.current_vertex])
        vertex_count = env.n_vertices

        print("Initial vertex:", env.current_vertex)
        print("Starting test...")

        for step_count in range(vertex_count - 1):  # Since we already visited 1
            valid_actions = [v for v in range(vertex_count) if v not in visited]
            assert len(valid_actions) > 0, "No valid actions left but done=False"

            action = valid_actions[0]  # Just pick the first unvisited vertex
            print('action', action)
            obs, reward, done, _, info = env.step(action)
            print_state(obs, env.n_timesteps)

            # Unpack observation again
            traffic_tensor, dijkstra_tensor = obs

            # Test 1: current_vertex is updated
            assert env.current_vertex == action, "current_vertex not updated correctly"

            # Test 2: visited_vertices updates correctly
            assert action in env.visited_vertices, "visited_vertices missing the action"
            assert len(env.visited_vertices) == len(visited) + 1, "visited_vertices size incorrect"

            # Test 3: cum_time is valid
            assert env.cum_time >= 0, "cum_time should not be negative"

            # Test 4: observation components are valid
            assert isinstance(traffic_tensor, torch.Tensor)
            assert isinstance(dijkstra_tensor, torch.Tensor)
            assert traffic_tensor.shape == (env.n_vertices, env.n_vertices, env.n_timesteps)
            assert dijkstra_tensor.shape == (env.n_vertices,)
            assert traffic_tensor.dtype == torch.float32
            assert dijkstra_tensor.dtype == torch.float32

            # Update local visited tracker
            visited.add(action)

            if step_count < vertex_count - 2:
                assert not done, "Episode finished too early"
            else:
                assert done, "Episode should be done after visiting all vertices"

        print("All checks passed!")

if __name__ == "__main__":
    test_traffic_env_behavior()
