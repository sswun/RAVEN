"""Episode storage for recurrent value-based training.

Every episode occupies ``episode_limit + 1`` time slots. Slot t holds the
observation, state and legal actions at time t together with the action taken,
the reward received and whether the episode terminated. The final slot only
holds the last observation. ``filled`` marks slots that were written.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

FIELDS = ("obs", "state", "avail_actions", "actions", "actions_onehot", "reward", "terminated", "filled")


@dataclass
class EpisodeBatch:
    obs: torch.Tensor             # (B, T, N, obs_dim)
    state: torch.Tensor           # (B, T, state_dim)
    avail_actions: torch.Tensor   # (B, T, N, n_actions), int
    actions: torch.Tensor         # (B, T, N, 1), long
    actions_onehot: torch.Tensor  # (B, T, N, n_actions)
    reward: torch.Tensor          # (B, T, 1)
    terminated: torch.Tensor      # (B, T, 1)
    filled: torch.Tensor          # (B, T, 1)

    @classmethod
    def empty(cls, batch: int, length: int, n_agents: int, obs_dim: int, state_dim: int,
              n_actions: int, device="cpu") -> EpisodeBatch:
        def z(*shape, dtype=torch.float32):
            return torch.zeros(batch, length, *shape, dtype=dtype, device=device)
        return cls(obs=z(n_agents, obs_dim), state=z(state_dim),
                   avail_actions=z(n_agents, n_actions, dtype=torch.int32),
                   actions=z(n_agents, 1, dtype=torch.long), actions_onehot=z(n_agents, n_actions),
                   reward=z(1), terminated=z(1), filled=z(1))

    def __getitem__(self, key: str) -> torch.Tensor:
        return getattr(self, key)

    @property
    def batch_size(self) -> int:
        return self.obs.shape[0]

    @property
    def max_seq_length(self) -> int:
        return self.obs.shape[1]

    @property
    def device(self) -> torch.device:
        return self.obs.device

    def record_inputs(self, t: int, obs, state, avail) -> None:
        self.obs[:, t] = torch.as_tensor(np.asarray(obs), dtype=torch.float32)
        self.state[:, t] = torch.as_tensor(np.asarray(state), dtype=torch.float32)
        self.avail_actions[:, t] = torch.as_tensor(np.asarray(avail), dtype=torch.int32)
        self.filled[:, t] = 1

    def record_actions(self, t: int, actions: torch.Tensor) -> None:
        self.actions[:, t, :, 0] = actions
        self.actions_onehot[:, t] = torch.nn.functional.one_hot(actions, self.actions_onehot.shape[-1]).float()

    def record_outcome(self, t: int, reward: float, terminated: bool) -> None:
        self.reward[:, t] = float(reward)
        self.terminated[:, t] = float(terminated)

    def truncate(self, length: int) -> EpisodeBatch:
        return EpisodeBatch(**{k: getattr(self, k)[:, :length] for k in FIELDS})

    def max_filled(self) -> int:
        return int(self.filled.squeeze(-1).sum(1).max().item())

    def to(self, device) -> EpisodeBatch:
        return EpisodeBatch(**{k: getattr(self, k).to(device) for k in FIELDS})

    def clone(self) -> EpisodeBatch:
        return EpisodeBatch(**{k: getattr(self, k).clone() for k in FIELDS})


class ReplayBuffer:
    """Circular buffer of whole episodes, sampled uniformly without replacement."""

    def __init__(self, capacity: int, length: int, n_agents: int, obs_dim: int, state_dim: int,
                 n_actions: int):
        self.capacity = capacity
        self.storage = EpisodeBatch.empty(capacity, length, n_agents, obs_dim, state_dim, n_actions)
        self.position = 0
        self.size = 0

    def insert(self, episode: EpisodeBatch) -> None:
        for i in range(episode.batch_size):
            for key in FIELDS:
                getattr(self.storage, key)[self.position] = getattr(episode, key)[i].cpu()
            self.position = (self.position + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def can_sample(self, batch: int) -> bool:
        return self.size >= batch

    def sample(self, batch: int) -> EpisodeBatch:
        if self.size == batch:
            index = np.arange(batch)
        else:
            index = np.random.choice(self.size, batch, replace=False)
        index = torch.as_tensor(index)
        sampled = EpisodeBatch(**{k: getattr(self.storage, k)[index] for k in FIELDS})
        return sampled.truncate(sampled.max_filled())
