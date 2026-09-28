"""Stage 2 of the offline implementation: neural sender and receivers.

The sender of every edge is a small MLP distilled from the exact codebook with
cross-entropy and then frozen, together with its input normalisation. The
receivers read their own normalised observation and the one-hot symbol that
arrived from their predecessor; they are trained to maximise the teacher's
joint value in expectation over their joint action distribution, which can be
computed exactly because the joint action space is enumerable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from raven.envs.navigation import EPISODE_LENGTH, N_ACTIONS, NavigationWorld
from raven.offline.construction import Construction


class OfflinePolicy(nn.Module):
    """Senders and receivers for a ring of ``n_agents`` (agent i listens to i - 1)."""

    def __init__(self, n_agents: int, obs_dim: int, n_symbols: int, seed: int,
                 sender_hidden: int = 64, receiver_hidden: int = 32):
        super().__init__()
        self.n_agents, self.n_symbols = n_agents, n_symbols
        torch.manual_seed(seed + 100)
        self.receivers = nn.ModuleList(
            nn.Sequential(nn.Linear(obs_dim + n_symbols, receiver_hidden), nn.Tanh(),
                          nn.Linear(receiver_hidden, N_ACTIONS))
            for _ in range(n_agents))
        torch.manual_seed(seed + 800)
        self.senders = nn.ModuleList(
            nn.Sequential(nn.Linear(obs_dim, sender_hidden), nn.Tanh(),
                          nn.Linear(sender_hidden, sender_hidden), nn.Tanh(),
                          nn.Linear(sender_hidden, n_symbols))
            for _ in range(n_agents))
        self.register_buffer("mean", torch.zeros(n_agents, obs_dim))
        self.register_buffer("scale", torch.ones(n_agents, obs_dim))

    def normalise(self, local: torch.Tensor) -> torch.Tensor:
        return (local - self.mean) / self.scale

    def sender_logits(self, x: torch.Tensor) -> torch.Tensor:
        """x: normalised local observations (batch, n_agents, obs_dim)."""
        return torch.stack([self.senders[j](x[:, j]) for j in range(self.n_agents)], 1)

    def symbols(self, x: torch.Tensor) -> torch.Tensor:
        return self.sender_logits(x).argmax(-1)

    def receiver_logits(self, x: torch.Tensor, inbox: torch.Tensor) -> torch.Tensor:
        """inbox[:, j] is the one-hot symbol that agent j sent one step earlier."""
        return torch.stack([self.receivers[i](torch.cat((x[:, i], inbox[:, (i - 1) % self.n_agents]), -1))
                            for i in range(self.n_agents)], 1)


def joint_expectation(q: torch.Tensor, probs: torch.Tensor) -> torch.Tensor:
    """E_{a ~ prod_i probs[:, i]} q[:, a] for a joint table with agent 0 most significant."""
    n = probs.shape[1]
    value = q.reshape(len(q), *([N_ACTIONS] * n))
    for i in range(n - 1, -1, -1):
        shape = [len(q)] + [1] * i + [N_ACTIONS]
        value = (value * probs[:, i].reshape(shape)).sum(-1)
    return value


@dataclass
class PolicyConfig:
    sender_updates: int = 2048
    receiver_updates: int = 8192
    batch_size: int = 512
    lr: float = 3e-3
    grad_clip: float = 5.0


def fit_policy(construction: Construction, n_symbols: int, cfg: PolicyConfig, seed: int,
               device="cpu", log=None) -> OfflinePolicy:
    n_agents, obs_dim = construction.current.shape[1:]
    policy = OfflinePolicy(n_agents, obs_dim, n_symbols, seed).to(device)
    policy.mean.copy_(torch.as_tensor(construction.mean, dtype=torch.float32))
    policy.scale.copy_(torch.as_tensor(construction.scale, dtype=torch.float32))

    current = torch.as_tensor(construction.current, device=device)
    arrival = torch.as_tensor(construction.arrival, device=device)
    objective = torch.as_tensor(construction.objective, device=device)
    labels = torch.as_tensor(construction.sender_labels(), device=device)
    rows = len(current)

    # Distil the codebook into the sender networks, then freeze them.
    rng = np.random.default_rng(seed + 1600)
    optimiser = torch.optim.Adam(policy.senders.parameters(), lr=cfg.lr)
    for _ in range(cfg.sender_updates):
        idx = torch.as_tensor(rng.integers(rows, size=cfg.batch_size), device=device)
        logits = policy.sender_logits(current[idx])
        loss = F.cross_entropy(logits.flatten(0, 1), labels[idx].flatten())
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.senders.parameters(), cfg.grad_clip)
        optimiser.step()
    policy.senders.requires_grad_(False)
    with torch.no_grad():
        agreement = (policy.symbols(current) == labels).float().mean().item()
    if log is not None:
        log.info(f"sender distillation: agreement with the codebook {agreement:.4f}")

    # Train the receivers on the teacher's joint value with the frozen senders.
    with torch.no_grad():
        inbox = F.one_hot(policy.symbols(current), n_symbols).float()
    rng = np.random.default_rng(seed + 2000)
    optimiser = torch.optim.Adam(policy.receivers.parameters(), lr=cfg.lr)
    for update in range(cfg.receiver_updates):
        idx = torch.as_tensor(rng.integers(rows, size=cfg.batch_size), device=device)
        probs = policy.receiver_logits(arrival[idx], inbox[idx]).softmax(-1)
        loss = -joint_expectation(objective[idx], probs).mean()
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(policy.receivers.parameters(), cfg.grad_clip)
        optimiser.step()
        if log is not None and (update + 1) % 1024 == 0:
            log.log("receiver/expected_value", -loss.item(), update + 1)
    policy.requires_grad_(False)
    return policy


@torch.no_grad()
def evaluate_policy(policy: OfflinePolicy, seeds: list[int], drop_messages: bool = False,
                    every: int = 1, device="cpu") -> dict:
    """Greedy rollouts with a one-step-delayed mailbox.

    ``every`` > 1 sends a fresh symbol only every ``every`` steps; the receiver
    keeps the last symbol that arrived.
    """
    n = policy.n_agents
    world = NavigationWorld("ring", n, seeds)
    x = torch.as_tensor(world.observation(), device=device)
    inbox = torch.zeros(len(seeds), n, policy.n_symbols, device=device)
    returns = np.zeros(len(seeds))
    counts = np.zeros(policy.n_symbols, dtype=np.int64)
    for t in range(EPISODE_LENGTH):
        local = policy.normalise(x)
        actions = policy.receiver_logits(local, inbox).argmax(-1)
        sent = policy.symbols(local)
        if t % every == 0:
            counts += np.bincount(sent.flatten().cpu().numpy(), minlength=policy.n_symbols)
            if not drop_messages:
                inbox = F.one_hot(sent, policy.n_symbols).float()
        obs, reward, _ = world.step(actions.cpu().numpy())
        returns += reward.mean(1)
        x = torch.as_tensor(obs, device=device)
    return {
        "mean_return": float(returns.mean()),
        "episode_returns": returns.tolist(),
        "symbol_counts": counts.tolist(),
        "bits_per_message": int(np.ceil(np.log2(policy.n_symbols))),
    }
