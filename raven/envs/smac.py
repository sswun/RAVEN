"""StarCraft Multi-Agent Challenge maps (requires ``smac`` and StarCraft II)."""

from __future__ import annotations

DEFAULT_ARGS = {
    "difficulty": "7",
    "step_mul": 8,
    "move_amount": 2,
    "continuing_episode": False,
    "obs_all_health": True,
    "obs_own_health": True,
    "obs_last_action": False,
    "obs_pathing_grid": False,
    "obs_terrain_height": False,
    "obs_timestep_number": False,
    "obs_instead_of_state": False,
    "state_last_action": False,
    "state_timestep_number": False,
    "reward_sparse": False,
    "reward_only_positive": True,
    "reward_death_value": 10,
    "reward_win": 200,
    "reward_defeat": 0,
    "reward_negative_scale": 0.5,
    "reward_scale": True,
    "reward_scale_rate": 20,
    "heuristic_ai": False,
    "heuristic_rest": False,
    "debug": False,
}


class SMACTask:
    def __init__(self, map_name: str, seed: int = 0, **overrides):
        from smac.env import StarCraft2Env

        args = dict(DEFAULT_ARGS, map_name=map_name, seed=int(seed), **overrides)
        self.env = StarCraft2Env(**args)
        info = self.env.get_env_info()
        self.n_agents = info["n_agents"]
        self.obs_dim = info["obs_shape"]
        self.state_dim = info["state_shape"]
        self.n_actions = info["n_actions"]
        self.episode_limit = info["episode_limit"]

    def reset(self, seed: int | None = None) -> None:
        # SMAC draws episodes from the seed given at construction.
        self.env.reset()

    def get_obs(self):
        return self.env.get_obs()

    def get_state(self):
        return self.env.get_state()

    def get_avail_actions(self):
        return self.env.get_avail_actions()

    def step(self, actions):
        reward, done, info = self.env.step([int(a) for a in actions])
        terminated = bool(done and not info.get("episode_limit", False))
        return float(reward), bool(done), terminated, info

    def close(self) -> None:
        self.env.close()
