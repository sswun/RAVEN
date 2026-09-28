"""Environment adapters.

Every task exposes ``n_agents``, ``obs_dim``, ``state_dim``, ``n_actions``,
``episode_limit`` and the methods ``reset``, ``get_obs``, ``get_state``,
``get_avail_actions``, ``step(actions) -> (reward, done, terminated, info)``
and ``close``. The global state is only ever given to the mixing network.
"""

from __future__ import annotations


def make_task(kind: str, name: str, seed: int = 0, **kwargs):
    if kind == "navigation":
        from raven.envs.navigation import NavigationTask

        family, _, size = name.partition("-")      # e.g. "ring-3" or "broadcast-4"
        return NavigationTask(family, int(size), seed)
    if kind == "mpe":
        from raven.envs.mpe import MPETask

        return MPETask(name, seed)
    if kind == "smac":
        from raven.envs.smac import SMACTask

        return SMACTask(name, seed, **kwargs)
    raise ValueError(f"unknown task kind {kind!r}")
