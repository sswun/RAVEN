"""MPE2 tasks as fully cooperative problems.

All agents (including adversaries in Tag and Eve in Crypto) are controlled by
one team and the team reward is the sum of all agents' rewards. Observations
are zero-padded to a common size and the global state, used only by the
mixing network, is the concatenation of the padded observations.
"""

from __future__ import annotations

import importlib

import numpy as np

SCENARIOS = {
    "spread": ("simple_spread_v3", {"N": 3, "local_ratio": 0.5}),
    "tag": ("simple_tag_v3", {"num_good": 1, "num_adversaries": 3, "num_obstacles": 2}),
    "crypto": ("simple_crypto_v3", {}),
}


class MPETask:
    episode_limit = 25

    def __init__(self, name: str, seed: int = 0):
        if name not in SCENARIOS:
            raise ValueError(f"unknown MPE task {name!r}; choose from {sorted(SCENARIOS)}")
        module, kwargs = SCENARIOS[name]
        factory = importlib.import_module(f"mpe2.{module}")
        self.env = factory.parallel_env(max_cycles=self.episode_limit, continuous_actions=False,
                                        render_mode=None, **kwargs)
        self.agents = list(self.env.possible_agents)
        self.obs_sizes = [int(np.prod(self.env.observation_space(a).shape)) for a in self.agents]
        self.action_sizes = [int(self.env.action_space(a).n) for a in self.agents]
        self.n_agents = len(self.agents)
        self.obs_dim = max(self.obs_sizes)
        self.n_actions = max(self.action_sizes)
        self.state_dim = self.n_agents * self.obs_dim
        self.seed = int(seed)
        self._obs = None

    def reset(self, seed: int | None = None) -> None:
        self._obs, _ = self.env.reset(seed=self.seed if seed is None else int(seed))

    def get_obs(self) -> np.ndarray:
        padded = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        for i, name in enumerate(self.agents):
            value = np.asarray(self._obs[name], dtype=np.float32)
            padded[i, :len(value)] = value
        return padded

    def get_state(self) -> np.ndarray:
        return self.get_obs().reshape(-1)

    def get_avail_actions(self) -> np.ndarray:
        avail = np.zeros((self.n_agents, self.n_actions), dtype=np.int32)
        for i, size in enumerate(self.action_sizes):
            avail[i, :size] = 1
        return avail

    def step(self, actions):
        self._obs, rewards, terminated, truncated, _ = self.env.step(
            {name: int(a) for name, a in zip(self.agents, actions, strict=True)})
        done = all(terminated.values()) or all(truncated.values())
        return float(sum(rewards.values())), done, done, {}

    def close(self) -> None:
        self.env.close()
