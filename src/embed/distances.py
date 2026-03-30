import torch
from torch.linalg import vector_norm
import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import dijkstra
import scipy
from typing import Callable, Union

EPS = 1e-5



def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance between x and y."""
    return vector_norm(x - y, dim=-1)

def poincare_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Poincare distance between x and y."""
    euc_dist = ((x - y)**2).sum(-1)
#     norm_x = 1 - (x**2).sum(-1)
#     norm_y = 1 - (y**2).sum(-1)
    norm_x = torch.clamp(1 - (x**2).sum(-1), min=EPS)
    norm_y = torch.clamp(1 - (y**2).sum(-1), min=EPS)
    x = torch.clamp(1 + 2 * torch.div(euc_dist, (norm_x * norm_y)), min=1 + EPS)
    return torch.acosh(x)


def klein_distance(x,y): #with p and q in euclidean coordinates and their norm<1
    """Compute Klein-Beltrami distance between x and y."""

    norm_sq_x = (x**2).sum(-1)
    norm_sq_y = (y**2).sum(-1)
    dot_xy = (x*y).sum(-1)

    if isinstance(x, np.ndarray) and isinstance(y, np.ndarray):
        return np.arccosh((1-dot_xy)/np.sqrt((1-norm_sq_x)*(1-norm_sq_y)))  

    elif isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor):
        return torch.arccosh((1-dot_xy)/torch.sqrt((1-norm_sq_x)*(1-norm_sq_y)))  


# def distance_matrix(data: torch.Tensor, distance_func: Callable) -> torch.Tensor:
#     """Builds a distance matrix given a vectorized distance function."""
#     preliminary = distance_func(data.unsqueeze(0), data.unsqueeze(1))
#     result = preliminary + torch.eye(data.shape[0], device=data.device)
#     return result.float()
def distance_matrix(points, dist_func=klein_distance):

    if isinstance(points, torch.Tensor):
        return dist_func(points.unsqueeze(0), points.unsqueeze(1))

    elif isinstance(points, np.ndarray):
        return dist_func(np.expand_dims(points, 0), np.expand_dims(points, 1)) 



def klein_to_poincare(x):
    # x: shape (..., 2)
    if isinstance(x, np.ndarray):
        norm_sq = np.sum(x**2, axis=-1, keepdims=True)
        return x / (1 + np.sqrt(1 - norm_sq ))
    elif isinstance(x, torch.Tensor):
        norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        return x / (1 + torch.sqrt(1 - norm_sq ))

def poincare_to_klein(x):
    # x: shape (..., 2)
    if isinstance(x, torch.Tensor):
        norm_sq = torch.sum(x**2, dim=-1, keepdim=True)
        return 2 * x / (1 + norm_sq)
    elif isinstance(x, np.ndarray):
        norm_sq = np.sum(x**2, axis=-1, keepdims=True)
        return 2 * x / (1 + norm_sq)


def hamming_distance(x: np.ndarray, y: np.ndarray) -> int:
    """Compute Hamming distance between x and y."""
    return (x.astype(np.int32) ^ y.astype(np.int32)).sum()

# def knn_geodesic_distance_matrix(data: np.ndarray, n_neighbors: int = 20) -> torch.Tensor:
#     """Compute geodesic distance matrix using k-nearest neighbors."""
#     data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='connectivity', include_self=False)
#     data_dist_matrix = data_nn_matrix.toarray()
#     # data_dist_matrix = dijkstra(data_nn_matrix)

#     data_dist_matrix = torch.FloatTensor(data_dist_matrix)
#     # data_dist_matrix = torch.where(data_dist_matrix == torch.inf, 1000 * torch.ones_like(data_dist_matrix), data_dist_matrix)
#     return data_dist_matrix

# def knn_graph_weighted_adjacency_matrix(data: np.ndarray, n_neighbors: int = 3, metric: str = 'minkowski') -> np.ndarray:
#     """Compute k-nearest neighbors graph weighted adjacency matrix."""
#     data_nn_matrix = kneighbors_graph(data, n_neighbors, mode='distance', include_self=False, metric=metric)
#     return data_nn_matrix.toarray()

# def hamming_distance_matrix(data: np.ndarray) -> np.ndarray:
#     """Compute Hamming distance matrix for binary data."""
#     return scipy.spatial.distance.cdist(data, data, metric='hamming') * data.shape[-1]
#