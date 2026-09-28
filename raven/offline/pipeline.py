"""End-to-end offline RAVEN on the ring navigation task."""

from __future__ import annotations

import time
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np
import torch

from raven.envs.navigation import ring_edges
from raven.offline.construction import build_construction
from raven.offline.policy import PolicyConfig, evaluate_policy, fit_policy
from raven.offline.teacher import TeacherConfig, train_teacher
from raven.utils import MetricLogger, count_parameters, seed_everything, write_json


def _dataclass_from(cls, values: dict):
    names = {f.name for f in fields(cls)}
    unknown = set(values) - names
    if unknown:
        raise ValueError(f"unknown {cls.__name__} options: {sorted(unknown)}")
    if "checkpoints" in values:
        values = dict(values, checkpoints=tuple(values["checkpoints"]))
    return cls(**values)


def run_offline(config: dict, out_dir: str | Path, device: str = "cpu") -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    log = MetricLogger(out / "metrics.jsonl")
    seed = int(config["seed"])
    n_agents = int(config["n_agents"])
    n_symbols = int(config.get("n_symbols", 4))
    seed_everything(seed)
    started = time.time()

    teacher_cfg = _dataclass_from(TeacherConfig, config.get("teacher", {}))
    log.info(f"[1/4] training the centralised teacher ({teacher_cfg.head} head, {teacher_cfg.steps} steps)")
    teacher, data, teacher_info = train_teacher(teacher_cfg, n_agents, seed, device, log)

    log.info(f"[2/4] building conditional value tables on {len(data['current'])} transitions")
    construction_cfg = config.get("construction", {})
    construction = build_construction(
        teacher, data["current"], data["arrival"], ring_edges(n_agents), seed=seed,
        n_symbols=n_symbols, **construction_cfg)
    del teacher

    log.info("[3/4] distilling the codebook and training the receivers")
    policy_cfg = _dataclass_from(PolicyConfig, config.get("policy", {}))
    policy = fit_policy(construction, n_symbols, policy_cfg, seed, device, log)
    torch.save({"model": policy.state_dict(), "n_agents": n_agents, "n_symbols": n_symbols,
                "obs_dim": construction.current.shape[2]}, out / "policy.pt")

    evaluation = config.get("evaluation", {})
    episodes = int(evaluation.get("episodes", 1000))
    first_seed = int(evaluation.get("seed", 10_000_000 + seed))
    seeds = list(range(first_seed, first_seed + episodes))
    log.info(f"[4/4] evaluating on {episodes} held-out episodes")
    normal = evaluate_policy(policy, seeds, device=device)
    silent = evaluate_policy(policy, seeds, drop_messages=True, device=device)

    summary = {
        "n_agents": n_agents,
        "seed": seed,
        "mean_return": normal["mean_return"],
        "mean_return_without_messages": silent["mean_return"],
        "symbol_counts": normal["symbol_counts"],
        "bits_per_message": normal["bits_per_message"],
        "deployed_parameters": count_parameters(policy),
        "teacher_updates": teacher_info["updates"],
        "construction_rows": int(len(construction.current)),
        "codebooks": [{"sender": e.sender, "receiver": e.receiver, "code": e.code.tolist(),
                       "distortion": e.distortion, "energy": e.energy} for e in construction.edges],
        "config": {"teacher": asdict(teacher_cfg), "policy": asdict(policy_cfg),
                   "construction": construction_cfg, "n_symbols": n_symbols},
        "wall_seconds": time.time() - started,
    }
    write_json(out / "summary.json", summary)
    np.save(out / "episode_returns.npy", np.asarray(normal["episode_returns"]))
    log.info(f"mean return {summary['mean_return']:.3f} "
             f"(without messages {summary['mean_return_without_messages']:.3f})")
    return summary
