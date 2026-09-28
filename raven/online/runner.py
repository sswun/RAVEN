"""Episode collection, evaluation and the online training loop."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch

from raven.envs import make_task
from raven.online.buffer import EpisodeBatch, ReplayBuffer
from raven.online.controller import EpsilonGreedy, RavenController
from raven.online.learner import RavenLearner
from raven.utils import MetricLogger, count_parameters, seed_everything, write_json


def rollout(task, mac: RavenController, t_env: int, greedy: bool = False, seed: int | None = None):
    """Play one episode; returns the episode batch, its length and a result dict."""
    batch = EpisodeBatch.empty(1, task.episode_limit + 1, task.n_agents, task.obs_dim,
                               task.state_dim, task.n_actions, device=mac.device)
    task.reset(seed)
    mac.init_hidden(1)
    episode_return, info, t = 0.0, {}, 0
    for t in range(task.episode_limit):
        batch.record_inputs(t, [task.get_obs()], [task.get_state()], [task.get_avail_actions()])
        actions = mac.select_actions(batch, t, t_env, greedy=greedy)
        reward, done, terminated, info = task.step(actions[0].cpu().numpy())
        batch.record_actions(t, actions)
        batch.record_outcome(t, reward, terminated)
        episode_return += reward
        if done:
            break
    else:
        raise RuntimeError("episode exceeded the task's episode limit")
    length = t + 1
    batch.record_inputs(length, [task.get_obs()], [task.get_state()], [task.get_avail_actions()])
    batch.record_actions(length, mac.select_actions(batch, length, t_env, greedy=greedy))
    won = info.get("battle_won")
    return batch, length, {"return": episode_return, "won": None if won is None else float(won)}


@torch.no_grad()
def evaluate(task, mac: RavenController, episodes: int, first_seed: int, drop_messages: bool = False) -> dict:
    previous = mac.drop_messages
    mac.drop_messages = drop_messages
    returns, wins = [], []
    try:
        for i in range(episodes):
            _, _, result = rollout(task, mac, 0, greedy=True, seed=first_seed + i)
            returns.append(result["return"])
            if result["won"] is not None:
                wins.append(result["won"])
    finally:
        mac.drop_messages = previous
    out = {"episodes": episodes, "mean_return": float(np.mean(returns))}
    if wins:
        out["win_rate"] = float(np.mean(wins))
    return out


def build(task, cfg: dict, seed: int, device: str = "cpu"):
    """Controller and learner with the initialisation order used in the paper."""
    variant = cfg.get("variant", "full")
    topology = cfg.get("topology") or getattr(task, "topology", "all")
    seed_everything(seed)
    mac = RavenController(
        task.n_agents, task.obs_dim, task.n_actions,
        hidden_dim=cfg.get("hidden_dim", 64), n_symbols=cfg.get("n_symbols", 4),
        top_k=cfg.get("top_k", 2), key_dim=cfg.get("key_dim", 32), topology=topology,
        communicate=variant != "no_comm",
        epsilon=EpsilonGreedy(cfg.get("epsilon_start", 1.0), cfg.get("epsilon_finish", 0.05),
                              cfg.get("epsilon_anneal_steps", 50_000)),
        device=device)
    learner = RavenLearner(
        mac, task.state_dim, variant=variant, lr=cfg.get("lr", 5e-4), comm_lr=cfg.get("comm_lr", 3e-4),
        gamma=cfg.get("gamma", 0.99), cv_coef=cfg.get("cv_coef", 0.1),
        probe_states=cfg.get("probe_states", 8),
        target_update_interval=cfg.get("target_update_interval", 200),
        grad_clip=cfg.get("grad_clip", 10.0), mixing_embed_dim=cfg.get("mixing_embed_dim", 32),
        hypernet_dim=cfg.get("hypernet_dim", 64),
        standardise_rewards=cfg.get("standardise_rewards", True), mixer_seed=seed + 1)
    return mac, learner


def train_online(config: dict, out_dir: str | Path, device: str = "cpu") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = MetricLogger(out / "metrics.jsonl")
    seed = int(config["seed"])
    task_cfg = config["task"]
    train_cfg = config.get("training", {})
    task = make_task(task_cfg["kind"], task_cfg["name"], seed, **task_cfg.get("args", {}))
    mac, learner = build(task, config.get("model", {}), seed, device)

    t_max = int(train_cfg.get("t_max", 2_050_000))
    batch_size = int(train_cfg.get("batch_size", 32))
    eval_interval = int(train_cfg.get("eval_interval", 50_000))
    eval_episodes = int(train_cfg.get("eval_episodes", 100))
    final_episodes = int(train_cfg.get("final_eval_episodes", 1000))
    log_interval = int(train_cfg.get("log_interval", 10_000))
    replay = ReplayBuffer(int(train_cfg.get("buffer_size", 5000)), task.episode_limit + 1,
                          task.n_agents, task.obs_dim, task.state_dim, task.n_actions)
    eval_task = make_task(task_cfg["kind"], task_cfg["name"], seed + 100_000, **task_cfg.get("args", {}))

    log.info(f"{task_cfg['kind']}:{task_cfg['name']}  agents={task.n_agents}  "
             f"variant={config.get('model', {}).get('variant', 'full')}  t_max={t_max}")
    t_env, episodes, next_eval, next_log = 0, 0, eval_interval, 0
    started = time.time()
    recent = []
    while t_env < t_max:
        batch, length, result = rollout(task, mac, t_env)
        t_env += length
        episodes += 1
        recent.append(result["return"])
        replay.insert(batch)
        if replay.can_sample(batch_size):
            stats = learner.train(replay.sample(batch_size).to(device))
            if t_env >= next_log:
                for key, value in stats.items():
                    log.log(f"train/{key}", value, t_env)
                log.log("train/return", float(np.mean(recent)), t_env)
                recent.clear()
                next_log = t_env + log_interval
        if t_env >= next_eval:
            result = evaluate(eval_task, mac, eval_episodes, first_seed=seed + 100_000 + t_env)
            for key, value in result.items():
                log.log(f"eval/{key}", value, t_env)
            log.info(f"t={t_env:>9}  episodes={episodes:>7}  eval return {result['mean_return']:.3f}"
                     + (f"  win rate {result['win_rate']:.3f}" if "win_rate" in result else ""))
            next_eval += eval_interval

    learner.save(out / "checkpoint")
    final_seed = seed + 3_000_000
    final = evaluate(eval_task, mac, final_episodes, first_seed=final_seed)
    summary = {"task": task_cfg, "seed": seed, "environment_steps": t_env, "episodes": episodes,
               "updates": learner.updates, "final": final,
               "deployed_parameters": sum(count_parameters(m) for m in mac.modules().values()),
               "bits_per_agent_step": int(np.ceil(np.log2(mac.n_symbols))) if mac.communicate else 0,
               "wall_seconds": time.time() - started, "config": config}
    if mac.communicate and train_cfg.get("evaluate_without_messages", True):
        summary["final_without_messages"] = evaluate(eval_task, mac, final_episodes,
                                                     first_seed=final_seed, drop_messages=True)
    write_json(out / "summary.json", summary)
    task.close()
    eval_task.close()
    return summary
