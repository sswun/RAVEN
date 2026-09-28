"""Stage 1 of the offline implementation: from teacher values to a codebook.

For every directed edge (sender j -> receiver i) we build

* the receiver's centred action-value profile Y_i(a) at the arrival state,
  averaging the teacher's joint value over the other agents' actions;
* a finite *sender type* T = tau(X_j) by k-means on the sender's normalised
  local observation;
* a finite *receiver condition* C = c(V_i) by k-means on the receiver's
  arrival observation after re-weighting each feature by how strongly it moves
  the receiver's action-value differences (the sensitivity metric);
* the shrunk conditional value table mu[T, C, a] and the exact codebook.

Receiver conditions organise the training statistics only. They are never
transmitted and never computed at execution time.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from raven.envs.navigation import N_ACTIONS
from raven.offline.codebook import shrunk_table, solve_codebook

CONDITIONS = ("sensitivity", "geometry", "full_local", "unconditional")


def kmeans(values: np.ndarray, n_clusters: int, seed: int, iterations: int = 40):
    """k-means++ initialisation followed by a fixed number of Lloyd steps."""
    rng = np.random.default_rng(seed)
    centres = [values[int(rng.integers(len(values)))]]
    while len(centres) < n_clusters:
        distance = np.min(np.square(values[:, None] - np.stack(centres)[None]).sum(-1), axis=1)
        centres.append(values[rng.choice(len(values), p=distance / distance.sum())])
    centres = np.stack(centres).copy()
    for _ in range(iterations):
        labels = assign(values, centres)
        for k in range(n_clusters):
            if np.any(labels == k):
                centres[k] = values[labels == k].mean(0)
    labels = assign(values, centres)
    if len(np.unique(labels)) != n_clusters:
        raise RuntimeError("k-means produced an empty cluster")
    return centres, labels


def assign(values: np.ndarray, centres: np.ndarray) -> np.ndarray:
    return np.square(values[:, None] - centres[None]).sum(-1).argmin(-1)


def receiver_marginal(q: torch.Tensor, n_agents: int, receiver: int) -> torch.Tensor:
    """Average a joint value table over every agent except ``receiver``."""
    shaped = q.reshape(len(q), *([N_ACTIONS] * n_agents))
    others = tuple(j + 1 for j in range(n_agents) if j != receiver)
    return shaped.mean(others) if others else shaped


def sensitivity_metric(teacher, arrival: np.ndarray, scale: np.ndarray, receiver: int,
                       batch: int = 512) -> np.ndarray:
    """Average outer product of d Y_i(a) / d V_i over data and receiver actions.

    ``arrival`` is the raw joint observation (rows, n_agents, obs_dim); the
    gradient is taken with respect to the receiver's own normalised features.
    """
    n_agents, dim = arrival.shape[1:]
    metric = np.zeros((dim, dim))
    device = next(teacher.parameters()).device
    for start in range(0, len(arrival), batch):
        x = torch.tensor(arrival[start:start + batch], dtype=torch.float32, device=device, requires_grad=True)
        profile = receiver_marginal(teacher(x.flatten(1)), n_agents, receiver)
        profile = profile - profile.mean(-1, keepdim=True)
        for a in range(N_ACTIONS):
            (grad,) = torch.autograd.grad(profile[:, a].sum(), x, retain_graph=a < N_ACTIONS - 1)
            g = grad[:, receiver].double().cpu().numpy() * scale[receiver]
            metric += g.T @ g
    return metric / (N_ACTIONS * len(arrival))


def metric_transform(metric: np.ndarray, ridge: float = 0.05, diagonal: bool = True) -> np.ndarray:
    """Linear map L with L L^T = G, where G is the trace-normalised metric plus a ridge."""
    g = (metric + metric.T) / 2
    if not np.trace(g) > 0:
        raise ValueError("sensitivity metric is degenerate")
    g = g * (len(g) / np.trace(g))
    if diagonal:
        g = np.diag(np.diag(g))
    g = g + ridge * np.eye(len(g))
    values, vectors = np.linalg.eigh(g)
    return vectors * np.sqrt(np.maximum(values, 0.0))[None]


@dataclass
class EdgeCode:
    sender: int
    receiver: int
    code: np.ndarray                 # symbol for each sender type
    counts: np.ndarray               # (types, conditions)
    distortion: float
    energy: float
    condition_transform: np.ndarray | None = None
    condition_centres: np.ndarray | None = None


@dataclass
class Construction:
    """Everything stage 2 needs; all arrays are over non-terminal construction rows."""

    mean: np.ndarray                 # (n_agents, obs_dim) normalisation
    scale: np.ndarray
    current: np.ndarray              # normalised sender inputs at t, (rows, n_agents, obs_dim)
    arrival: np.ndarray              # normalised receiver inputs at t + 1
    objective: np.ndarray            # centred, scaled joint teacher value at t + 1, (rows, 5**n)
    type_centres: np.ndarray         # (n_agents, types, obs_dim), in normalised units
    types: np.ndarray                # (rows, n_agents)
    edges: list[EdgeCode] = field(default_factory=list)

    def sender_labels(self) -> np.ndarray:
        """Target symbol of every sender on every row, shape (rows, n_agents)."""
        labels = np.zeros_like(self.types)
        for edge in self.edges:
            labels[:, edge.sender] = edge.code[self.types[:, edge.sender]]
        return labels


def build_construction(teacher, current_raw: np.ndarray, arrival_raw: np.ndarray,
                       edges: list[tuple[int, int]], *, seed: int, n_types: int = 12,
                       n_conditions: int = 12, n_symbols: int = 4, shrinkage: float = 32.0,
                       ridge: float = 0.05, condition: str = "sensitivity") -> Construction:
    if condition not in CONDITIONS:
        raise ValueError(f"condition must be one of {CONDITIONS}")
    n_agents = current_raw.shape[1]
    mean = current_raw.mean(0)
    scale = np.maximum(current_raw.std(0), 0.1)
    current = (current_raw - mean) / scale
    arrival = (arrival_raw - mean) / scale

    device = next(teacher.parameters()).device
    with torch.no_grad():
        q = torch.cat([teacher(torch.as_tensor(arrival_raw[i:i + 1024], device=device).flatten(1)).cpu()
                       for i in range(0, len(arrival_raw), 1024)]).numpy()
    q_scale = max(float(q.std(dtype=np.float64)), 1e-6)
    objective = ((q - q.mean(1, keepdims=True)) / q_scale).astype(np.float32)

    type_centres, types = [], []
    for j in range(n_agents):
        centres, labels = kmeans(current[:, j].astype(np.float64), n_types, seed + j)
        type_centres.append(centres)
        types.append(labels)
    types = np.stack(types, 1)

    result = Construction(mean=mean, scale=scale, current=current.astype(np.float32),
                          arrival=arrival.astype(np.float32), objective=objective,
                          type_centres=np.stack(type_centres), types=types)
    for sender, receiver in edges:
        profile = receiver_marginal(torch.as_tensor(objective), n_agents, receiver).double().numpy()
        profile = profile - profile.mean(-1, keepdims=True)
        z = arrival[:, receiver].astype(np.float64)
        transform = centres = None
        if condition == "unconditional":
            labels = np.zeros(len(z), dtype=np.int64)
            n_cond = 1
        else:
            if condition == "sensitivity":
                metric = sensitivity_metric(teacher, arrival_raw, scale, receiver)
                transform = metric_transform(metric, ridge)
                features = z @ transform
            elif condition == "geometry":
                features = z[:, :8]
            else:
                features = z
            centres, labels = kmeans(features, n_conditions, seed + 73000 + receiver)
            n_cond = n_conditions
        table, mass = shrunk_table(profile, types[:, sender], labels, n_types, n_cond, shrinkage)
        solved = solve_codebook(table, mass, n_symbols)
        counts = np.rint(mass * len(z)).astype(np.int64)
        result.edges.append(EdgeCode(sender, receiver, solved.code, counts, solved.distortion,
                                     solved.energy, transform, centres))
    return result
