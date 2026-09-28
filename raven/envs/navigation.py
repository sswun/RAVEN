"""Vectorised reference-navigation worlds used in the navigation experiments.

Two families are provided.

``ring``
    N agents and three landmarks. Every agent observes its own velocity, the
    relative positions of the three landmarks and the colour of the goal of its
    *successor*; its own goal colour is held by its predecessor. With N = 2 this
    is the collision-free MPE ``simple_reference`` task (``local_ratio=0``):
    local observations and team rewards match the MPE2 implementation.

``broadcast``
    One speaker that sees the goal colour but cannot move, and L listeners that
    see the landmarks but not the goal. With L = 1 this is MPE
    ``simple_speaker_listener``.

The world has no communication channel of its own; message timing is handled
by the policies (one-step delay).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

N_LANDMARKS = 3
N_ACTIONS = 5
EPISODE_LENGTH = 25
LOCAL_DIM = 11  # velocity (2) + landmark offsets (6) + goal colour (3)

# MPE movement constants: sensitivity 5, dt 0.1, damping 0.25, unit mass.
_DIRECTIONS = np.array([[0.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [0.0, -1.0], [0.0, 1.0]])
_DT, _DAMPING, _SENSITIVITY = 0.1, 0.25, 5.0


class NavigationWorld:
    """A batch of independent episodes, one per reset seed."""

    def __init__(self, family: str, n_agents: int, seeds: Sequence[int]):
        if family not in ("ring", "broadcast"):
            raise ValueError(f"unknown family {family!r}")
        self.family = family
        self.n_agents = int(n_agents)
        self.seeds = [int(s) for s in seeds]
        self.batch = len(self.seeds)
        self.t = 0

        n_goals = self.n_agents if family == "ring" else 1
        self.positions = np.empty((self.batch, self.n_agents, 2))
        self.velocity = np.zeros_like(self.positions)
        self.landmarks = np.empty((self.batch, N_LANDMARKS, 2))
        self.goals = np.empty((self.batch, n_goals), dtype=np.int64)
        self.speaker_position = np.empty((self.batch, 2))
        for b, seed in enumerate(self.seeds):
            rng = np.random.default_rng(seed)
            self.goals[b] = rng.integers(N_LANDMARKS, size=n_goals)
            if family == "broadcast":
                self.speaker_position[b] = rng.uniform(-1, 1, 2)
            for j in range(self.n_agents):
                self.positions[b, j] = rng.uniform(-1, 1, 2)
            for j in range(N_LANDMARKS):
                self.landmarks[b, j] = rng.uniform(-1, 1, 2)

    def observation(self) -> np.ndarray:
        """Local observations, shape ``(batch, n_agents, 11)`` (ring) or ``(..., 8)``."""
        offsets = self.landmarks[:, None] - self.positions[:, :, None]
        local = np.concatenate((self.velocity, offsets.reshape(self.batch, self.n_agents, 6)), -1)
        if self.family == "ring":
            colour = 0.25 + 0.5 * np.eye(N_LANDMARKS)[self.goals]
            local = np.concatenate((local, colour), -1)
        return local.astype(np.float32)

    def speaker_observation(self) -> np.ndarray:
        if self.family != "broadcast":
            raise RuntimeError("only the broadcast family has a speaker")
        return np.eye(N_LANDMARKS, dtype=np.float32)[self.goals[:, 0]]

    def step(self, actions: np.ndarray):
        """Advance every episode by one step.

        Returns ``(observation, per-agent reward, done)``. The team reward used
        in the paper is the mean over agents.
        """
        actions = np.asarray(actions, dtype=np.int64)
        if self.t >= EPISODE_LENGTH:
            raise RuntimeError("episode already finished")
        if actions.shape != (self.batch, self.n_agents):
            raise ValueError(f"expected actions of shape {(self.batch, self.n_agents)}")
        if np.any((actions < 0) | (actions >= N_ACTIONS)):
            raise ValueError("action out of range")

        self.positions += self.velocity * _DT
        self.velocity *= 1 - _DAMPING
        self.velocity += _DIRECTIONS[actions] * _SENSITIVITY * _DT

        if self.family == "ring":
            goals = np.roll(self.goals, 1, axis=1)  # agent i's goal is held by agent i-1
        else:
            goals = np.broadcast_to(self.goals, (self.batch, self.n_agents))
        target = self.landmarks[np.arange(self.batch)[:, None], goals]
        squared = np.square(self.positions - target).sum(-1)
        rewards = -np.sqrt(squared) if self.family == "ring" else -squared
        self.t += 1
        return self.observation(), rewards, self.t == EPISODE_LENGTH


def ring_edges(n_agents: int) -> list[tuple[int, int]]:
    """Directed (sender, receiver) pairs of the ring topology."""
    return [(i, (i + 1) % n_agents) for i in range(n_agents)]


def joint_to_actions(index: np.ndarray, n_agents: int) -> np.ndarray:
    """Decode joint-action indices; agent 0 is the most significant digit."""
    index = np.asarray(index)
    return np.stack([(index // N_ACTIONS ** (n_agents - 1 - j)) % N_ACTIONS for j in range(n_agents)], -1)


def actions_to_joint(actions: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions)
    n_agents = actions.shape[-1]
    powers = N_ACTIONS ** np.arange(n_agents - 1, -1, -1)
    return (actions * powers).sum(-1)


def heuristic_ring_actions(local: np.ndarray) -> np.ndarray:
    """Scripted centralised policy used only to seed the teacher's replay.

    Each agent reads its goal colour from its predecessor and moves along the
    dominant axis towards that landmark, damping its current velocity.
    """
    batch, n_agents, _ = local.shape
    actions = np.zeros((batch, n_agents), dtype=np.int64)
    for i in range(n_agents):
        goal = local[:, (i - 1) % n_agents, 8:11].argmax(1)
        offset = local[:, i, 2:8].reshape(batch, 3, 2)[np.arange(batch), goal]
        delta = offset - 0.8 * local[:, i, :2]
        axis = np.abs(delta).argmax(1)
        signed = delta[np.arange(batch), axis]
        actions[:, i] = np.where(axis == 0, np.where(signed < 0, 1, 2), np.where(signed < 0, 3, 4))
        actions[np.linalg.norm(delta, axis=1) < 0.03, i] = 0
    return actions


class NavigationTask:
    """Single-episode view of :class:`NavigationWorld` for the online learner.

    ``ring``: N agents; agent i may only hear agent i - 1.
    ``broadcast``: agent 0 is the speaker (observation slots 0-2, can only take
    the no-op action) and agents 1..L are listeners (slots 3-10).
    """

    episode_limit = EPISODE_LENGTH
    obs_dim = LOCAL_DIM
    n_actions = N_ACTIONS

    def __init__(self, family: str, n: int, seed: int = 0):
        self.family = family
        self.size = int(n)                      # ring agents or listeners
        self.n_agents = self.size if family == "ring" else self.size + 1
        self.state_dim = self.n_agents * self.obs_dim
        self.topology = "ring" if family == "ring" else "star"
        self.seed = int(seed)
        self.world = None

    def reset(self, seed: int | None = None) -> None:
        self.world = NavigationWorld(self.family, self.size, [self.seed if seed is None else int(seed)])

    def get_obs(self) -> np.ndarray:
        if self.family == "ring":
            return self.world.observation()[0]
        obs = np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)
        obs[0, :3] = self.world.speaker_observation()[0]
        obs[1:, 3:] = self.world.observation()[0]
        return obs

    def get_state(self) -> np.ndarray:
        return self.get_obs().reshape(-1)

    def get_avail_actions(self) -> np.ndarray:
        avail = np.ones((self.n_agents, self.n_actions), dtype=np.int32)
        if self.family == "broadcast":
            avail[0, 1:] = 0
        return avail

    def step(self, actions):
        actions = np.asarray(actions, dtype=np.int64)
        moving = actions if self.family == "ring" else actions[1:]
        _, reward, done = self.world.step(moving[None])
        return float(reward.mean()), bool(done), bool(done), {}

    def close(self) -> None:
        self.world = None
