import torch
from torch.distributions import Beta
from scipy.sparse.csgraph import shortest_path

# Synthetic data generation for TSP solver
# Creates an (i, n, k) tensor which represents the time to get from i->j at a timestep k
# when i=j, the value is inf
# For all i,j pairs, a random Beta curve is created (alpha and beta on 0.5 - 4.5)
# For all k in that i,j trajectory, the PDF value of the curve is taken and noise is added
# Therefore trajectories follow a basic structure but are still randomized

def generate(n_destinations: int, n_timesteps: int) -> torch.Tensor:
    noise_scale = 0.5
    noise_mean = 5

    x = torch.linspace(1e-4, 1-1e-4, n_timesteps)

    alphas = torch.rand(n_destinations, n_destinations) * 4 + 0.5
    betas = torch.rand(n_destinations, n_destinations) * 4 + 0.5

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