import torch
from torch.distributions import Beta
from scipy.sparse.csgraph import shortest_path
import os
from tqdm import tqdm

# Synthetic data generation for TSP solver
# Creates an (i, n, k) tensor which represents the time to get from i->j at a timestep k
# when i=j, the value is inf
# For all i,j pairs, a random Beta curve is created (alpha and beta on 0.5 - 4.5)
# For all k in that i,j trajectory, the PDF value of the curve is taken and noise is added
# Therefore trajectories follow a basic structure but are still randomized

def generate(n_destinations: int, n_timesteps: int) -> torch.Tensor:
    noise_scale = 0.5
    noise_mean = 20

    x = torch.linspace(1e-4, 1-1e-4, n_timesteps)

    alphas = torch.rand(n_destinations, n_destinations) * 9.5 + 0.1
    betas = torch.rand(n_destinations, n_destinations) * 9.5 + 0.1

    # EXPAND FOR BROADCASTING
    x = x.view(1, 1, n_timesteps).expand(n_destinations, n_destinations, n_timesteps)

    dist = Beta(alphas.unsqueeze(-1), betas.unsqueeze(-1))
    y = dist.log_prob(x).exp()  # shape: (n_dest, n_dest, n_points)

    noise = (torch.randn_like(y) + noise_mean) * noise_scale

    i = torch.arange(n_destinations)
    y[i, i, :] = float('inf')

    return y + noise

# Apply bellman ford to each timestep of the synthetic data and return as a tensor
# im worried this might change the shape shape of the tensor to be the wrong order of dims
def shortest_paths_by_slice(a: torch.Tensor) -> torch.Tensor:
    optim_slices = []
    for k in range(a.shape[2]):
        slice = a[:, :, k].numpy()
        optim_slice = shortest_path(slice, method='BF', directed=True, unweighted=False)
        optim_slices.append(optim_slice)

    return torch.Tensor(optim_slices)


def floyd_warshall(adj):
    n = adj.size(0)
    dist = adj.clone()
    for k in range(n):
        dist = torch.minimum(dist, dist[:, k].unsqueeze(1) + dist[k, :].unsqueeze(0))
    return dist


if __name__ == '__main__':
    outdir = 'traffic_maps2'
    os.makedirs(outdir, exist_ok=True)
    N = 10
    T = 10
    num_graphs = 1
    for i in tqdm(range(num_graphs)):
        adj = generate(N, T)
        graph = floyd_warshall(adj)
        torch.save(graph, os.path.join(outdir, f"traffic_{i}.pt"))

