"""Centralised Double-DQN teacher for the offline implementation.

The teacher sees the concatenated local observations of all agents and outputs
a value for every joint action. It exists only during training: its values
define the targets that the codebook and the receivers are fitted to.

Two heads are available:

* ``joint``    one output per joint action (5**N outputs); used for N = 2;
* ``additive`` Q(h, a) = (1/N) sum_k u_k(h, a_k); used for the larger rings.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from raven.envs.navigation import (
    EPISODE_LENGTH,
    LOCAL_DIM,
    N_ACTIONS,
    NavigationWorld,
    actions_to_joint,
    heuristic_ring_actions,
    joint_to_actions,
)


class JointTeacher(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int = LOCAL_DIM, hidden: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS ** n_agents),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        return self(x).argmax(-1)

    def value_of(self, x: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        return self(x).gather(1, joint[:, None]).squeeze(1)


class AdditiveTeacher(nn.Module):
    def __init__(self, n_agents: int, obs_dim: int = LOCAL_DIM, hidden: int = 128):
        super().__init__()
        self.n_agents = n_agents
        self.net = nn.Sequential(
            nn.Linear(obs_dim * n_agents, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, N_ACTIONS * n_agents),
        )
        powers = N_ACTIONS ** torch.arange(n_agents - 1, -1, -1)
        digits = (torch.arange(N_ACTIONS ** n_agents)[:, None] // powers[None]) % N_ACTIONS
        self.register_buffer("powers", powers, persistent=False)
        self.register_buffer("digits", digits, persistent=False)

    def utilities(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).reshape(len(x), self.n_agents, N_ACTIONS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = self.utilities(x)
        value = u[:, 0][:, self.digits[:, 0]]
        for k in range(1, self.n_agents):
            value = value + u[:, k][:, self.digits[:, k]]
        return value / self.n_agents

    def greedy(self, x: torch.Tensor) -> torch.Tensor:
        return (self.utilities(x).argmax(-1) * self.powers).sum(-1)

    def value_of(self, x: torch.Tensor, joint: torch.Tensor) -> torch.Tensor:
        u = self.utilities(x)
        digits = (joint[:, None] // self.powers) % N_ACTIONS
        return u.gather(2, digits[..., None]).squeeze(-1).mean(1)


def make_teacher(head: str, n_agents: int) -> nn.Module:
    if head == "joint":
        return JointTeacher(n_agents)
    if head == "additive":
        return AdditiveTeacher(n_agents)
    raise ValueError(f"unknown teacher head {head!r}")


@dataclass
class TeacherConfig:
    head: str = "joint"
    steps: int = 1_200_000
    parallel_envs: int = 16
    warmup: int = 5_000
    batch_size: int = 512
    replay: int = 200_000
    lr: float = 3e-4
    gamma: float = 0.95
    target_every: int = 1_000
    epsilon_decay_steps: int = 600_000
    epsilon_final: float = 0.05
    heuristic_steps: int = 0          # scripted exploration at the start (N = 2 setting)
    heuristic_noise: float = 0.2
    checkpoints: tuple[int, ...] = ()  # empty: keep the final network
    selection_episodes: int = 200
    construction_rows: int = 10_000    # first rows plus this many rows from the final replay


@torch.no_grad()
def greedy_return(teacher: nn.Module, n_agents: int, seeds: list[int], device) -> float:
    world = NavigationWorld("ring", n_agents, seeds)
    x = world.observation()
    total = np.zeros(len(seeds))
    for _ in range(EPISODE_LENGTH):
        joint = teacher.greedy(torch.as_tensor(x, device=device).flatten(1)).cpu().numpy()
        x, reward, _ = world.step(joint_to_actions(joint, n_agents))
        total += reward.mean(1)
    return float(total.mean())


def train_teacher(cfg: TeacherConfig, n_agents: int, seed: int, device="cpu", log=None):
    """Train the teacher and return it with the construction data.

    Returns ``(teacher, data)`` where ``data`` holds the raw joint observations
    ``current`` and ``arrival`` of the non-terminal construction transitions.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    net = make_teacher(cfg.head, n_agents).to(device)
    target = copy.deepcopy(net)
    optimiser = torch.optim.Adam(net.parameters(), lr=cfg.lr)

    b, cap = cfg.parallel_envs, cfg.replay
    replay = {
        "state": np.empty((cap, n_agents, LOCAL_DIM), np.float32),
        "next_state": np.empty((cap, n_agents, LOCAL_DIM), np.float32),
        "action": np.empty(cap, np.int64),
        "reward": np.empty(cap, np.float32),
        "done": np.empty(cap, bool),
    }
    first = {k: [] for k in ("state", "next_state", "done")}
    position = length = updates = 0
    next_seed = seed + 10_000
    world = NavigationWorld("ring", n_agents, range(next_seed, next_seed + b))
    next_seed += b
    x = world.observation()

    selection_seeds = list(range(seed + 900_000, seed + 900_000 + cfg.selection_episodes))
    snapshots = []

    def tensor(a):
        return torch.as_tensor(a, dtype=torch.float32, device=device)

    for step in range(0, cfg.steps, b):
        if step in cfg.checkpoints:
            snapshots.append((step, copy.deepcopy(net.state_dict()),
                              greedy_return(net, n_agents, selection_seeds, device)))

        with torch.no_grad():
            joint = net.greedy(tensor(x).flatten(1)).cpu().numpy()
        if step < cfg.heuristic_steps:
            joint = actions_to_joint(heuristic_ring_actions(x))
            explore = rng.random(b) < cfg.heuristic_noise
        else:
            epsilon = max(cfg.epsilon_final, 1 - (1 - cfg.epsilon_final) * step / cfg.epsilon_decay_steps)
            explore = rng.random(b) < epsilon
        joint[explore] = rng.integers(N_ACTIONS ** n_agents, size=int(explore.sum()))

        nx, reward, done = world.step(joint_to_actions(joint, n_agents))
        rows = (position + np.arange(b)) % cap
        batch = {"state": x, "next_state": nx, "action": joint,
                 "reward": reward.mean(1), "done": np.full(b, done)}
        for key, value in batch.items():
            replay[key][rows] = value
        if step < cfg.construction_rows:
            keep = min(b, cfg.construction_rows - step)
            for key in first:
                first[key].append(np.array(batch[key][:keep]))
        position = (position + b) % cap
        length = min(length + b, cap)

        if done:
            world = NavigationWorld("ring", n_agents, range(next_seed, next_seed + b))
            next_seed += b
            x = world.observation()
        else:
            x = nx

        if step + b >= cfg.warmup:
            idx = rng.integers(length, size=cfg.batch_size)
            s, s2 = tensor(replay["state"][idx]).flatten(1), tensor(replay["next_state"][idx]).flatten(1)
            a = torch.as_tensor(replay["action"][idx], device=device)
            with torch.no_grad():
                best = net.greedy(s2)
                not_done = tensor(~replay["done"][idx])
                y = tensor(replay["reward"][idx]) + cfg.gamma * not_done * target.value_of(s2, best)
            loss = nn.functional.smooth_l1_loss(net.value_of(s, a), y)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimiser.step()
            updates += 1
            if updates % cfg.target_every == 0:
                target.load_state_dict(net.state_dict())
            if log is not None and updates % 5000 == 0:
                log.log("teacher/loss", loss.item(), step + b)

    if cfg.checkpoints:
        if cfg.steps in cfg.checkpoints:
            snapshots.append((cfg.steps, copy.deepcopy(net.state_dict()),
                              greedy_return(net, n_agents, selection_seeds, device)))
        chosen_step, state, value = max(snapshots, key=lambda item: item[2])
        net.load_state_dict(state)
        if log is not None:
            log.info(f"teacher: selected step {chosen_step} (selection return {value:.3f})")
    net.eval().requires_grad_(False)

    sample = np.random.default_rng(seed + 96_000).choice(length, min(cfg.construction_rows, length), replace=False)
    collected = {k: np.concatenate([np.concatenate(first[k]), replay[k][sample]]) for k in first}
    valid = ~collected["done"]
    data = {"current": collected["state"][valid], "arrival": collected["next_state"][valid]}
    return net, data, {"updates": updates, "steps": cfg.steps}
