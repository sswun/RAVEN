"""Training objective of the online implementation.

    L = 1/2 (L_TD(deployed) + L_TD(reference)) + beta * L_CV

Both TD losses are Double-Q QMIX losses and update every trainable module.
L_CV is the conditional value alignment: at a few sampled time steps, and for
every receiver, it compares the team value of *each* of the receiver's legal
actions under the deployed branch (symbols) and under the reference branch
(continuous states), with the partners' utilities held at their replayed
actions. The profiles are centred over legal actions, so only action
differences matter. Gradients of L_CV reach only the encoder and the symbol
embedding; the receiver, the local network and the mixer enter it frozen.
"""

from __future__ import annotations

import copy
import random
from pathlib import Path

import numpy as np
import torch
from torch.func import functional_call

from raven.online.buffer import EpisodeBatch
from raven.online.controller import RavenController
from raven.online.networks import QMixer

VARIANTS = ("full", "no_cv", "no_reference", "no_comm")


def frozen_call(module: torch.nn.Module, *args):
    """Call ``module`` with detached parameters (inputs still carry gradients)."""
    params = {name: p.detach() for name, p in module.named_parameters()}
    return functional_call(module, params, args)


class RunningMeanStd:
    """Running mean and variance (parallel algorithm of Chan et al.)."""

    def __init__(self, device="cpu", epsilon: float = 1e-4):
        self.mean = torch.zeros(1, device=device)
        self.var = torch.ones(1, device=device)
        self.count = epsilon

    def update(self, x: torch.Tensor) -> None:
        x = x.reshape(-1, 1)
        batch_mean, batch_var, n = x.mean(0), x.var(0), x.shape[0]
        delta = batch_mean - self.mean
        total = self.count + n
        self.mean = self.mean + delta * n / total
        self.var = (self.var * self.count + batch_var * n + delta.square() * self.count * n / total) / total
        self.count = total


class RavenLearner:
    def __init__(self, controller: RavenController, state_dim: int, *, variant: str = "full",
                 lr: float = 5e-4, comm_lr: float = 3e-4, gamma: float = 0.99, cv_coef: float = 0.1,
                 probe_states: int = 8, target_update_interval: int = 200, grad_clip: float = 10.0,
                 mixing_embed_dim: int = 32, hypernet_dim: int = 64, standardise_rewards: bool = True,
                 mixer_seed: int | None = None):
        if variant not in VARIANTS:
            raise ValueError(f"variant must be one of {VARIANTS}")
        if (variant == "no_comm") == controller.communicate:
            raise ValueError("variant 'no_comm' needs a controller built with communicate=False")
        self.mac = controller
        self.variant = variant
        self.gamma, self.cv_coef, self.probe_states = gamma, cv_coef, probe_states
        self.target_update_interval, self.grad_clip = target_update_interval, grad_clip
        self.standardise_rewards = standardise_rewards

        if mixer_seed is not None:
            torch.manual_seed(mixer_seed)
        self.mixer = QMixer(controller.n_agents, state_dim, mixing_embed_dim, hypernet_dim).to(controller.device)
        self.target_mac = copy.deepcopy(controller)
        self.target_mixer = copy.deepcopy(self.mixer)
        for p in self.target_mac.parameters() + list(self.target_mixer.parameters()):
            p.requires_grad_(False)

        groups = [{"params": list(controller.agent.parameters()) + list(self.mixer.parameters()), "lr": lr}]
        if controller.communicate:
            groups.append({"params": controller.communication_parameters(), "lr": comm_lr})
        self.params = [p for g in groups for p in g["params"]]
        self.optimiser = torch.optim.Adam(groups)
        self.reward_stats = RunningMeanStd(controller.device)
        self.updates = 0
        self.last_target_update = 0

    # ------------------------------------------------------------------ L_CV
    def _team_profiles(self, q: torch.Tensor, partner: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        """Team value of every action of every receiver, partners fixed.

        q: (P, N, A) receiver action values, partner: (P, N) partner utilities,
        state: (P, S). Returns (P, N, A).
        """
        p, n, a = q.shape
        utilities = partner[:, None, None, :].expand(p, n, a, n).clone()
        for j in range(n):
            utilities[:, j, :, j] = q[:, j]
        states = state[:, None, None, :].expand(p, n, a, state.shape[-1])
        return frozen_call(self.mixer, utilities.reshape(p, n * a, n), states.reshape(p, n * a, -1)).reshape(p, n, a)

    def conditional_value_loss(self, rollout: dict, mask: torch.Tensor, batch: EpisodeBatch) -> torch.Tensor:
        valid = mask.squeeze(-1).bool().clone()
        valid[:, 0] = False                         # nothing has arrived at t = 0
        candidates = valid.nonzero(as_tuple=False)
        if len(candidates) == 0:
            return rollout["q"].new_zeros(())
        pick = candidates[torch.randperm(len(candidates), device=candidates.device)[:self.probe_states]]
        b, t = pick[:, 0], pick[:, 1]

        h = rollout["h"][b, t].detach()
        previous = rollout["previous"][b, t].detach()
        payload, _ = self.mac.codec(previous)                  # the only path with gradients
        delta, _ = frozen_call(self.mac.receiver, h, payload, payload, self.mac.allowed)
        deployed = frozen_call(self.mac.agent.fc2, h) + delta
        with torch.no_grad():
            ref_delta, _ = self.mac.receiver(h, payload.detach(), previous, self.mac.allowed)
            reference = self.mac.agent.fc2(h) + ref_delta
            actions = batch.actions[b, t]
            partner = reference.gather(-1, actions).squeeze(-1)
            state = batch.state[b, t]
            reference = self._team_profiles(reference, partner, state)
        deployed = self._team_profiles(deployed, partner, state)

        legal = batch.avail_actions[b, t].to(deployed.dtype)
        count = legal.sum(-1, keepdim=True).clamp_min(1)
        reference = (reference - (reference * legal).sum(-1, keepdim=True) / count) * legal
        deployed = (deployed - (deployed * legal).sum(-1, keepdim=True) / count) * legal
        scale = (reference.square().sum(-1, keepdim=True) / count).clamp_min(1)
        weight = legal * (count > 1).to(legal.dtype)
        return ((deployed - reference).square() / scale * weight).sum() / weight.sum().clamp_min(1)

    # ------------------------------------------------------------------ update
    def _td_loss(self, live_q, target_q, batch, rewards, terminated, mask):
        actions = batch.actions[:, :-1]
        chosen = live_q[:, :-1].gather(-1, actions).squeeze(-1)
        q_tot = self.mixer(chosen, batch.state[:, :-1])
        with torch.no_grad():
            next_avail = batch.avail_actions[:, 1:]
            best = live_q[:, 1:].masked_fill(next_avail == 0, -torch.inf).argmax(-1, keepdim=True)
            next_q = target_q[:, 1:].gather(-1, best).squeeze(-1)
            target = rewards + self.gamma * (1 - terminated) * self.target_mixer(next_q, batch.state[:, 1:])
        return ((q_tot - target).square() * mask).sum() / mask.sum().clamp_min(1)

    def train(self, batch: EpisodeBatch) -> dict[str, float]:
        mask = batch.filled[:, :-1].clone()
        terminated = batch.terminated[:, :-1]
        mask[:, 1:] *= 1 - terminated[:, :-1]
        rewards = batch.reward[:, :-1]
        if self.standardise_rewards:
            self.reward_stats.update(rewards[mask.squeeze(-1).bool()])
            rewards = (rewards - self.reward_stats.mean) / self.reward_stats.var.sqrt().clamp_min(1e-8)

        use_reference = self.variant in ("full", "no_cv")
        live = self.mac.unroll(batch, reference=use_reference)
        with torch.no_grad():
            target = self.target_mac.unroll(batch, reference=use_reference)

        stats = {}
        td = self._td_loss(live["q"], target["q"], batch, rewards, terminated, mask)
        stats["td_deployed"] = td.item()
        if use_reference:
            td_ref = self._td_loss(live["reference"], target["reference"], batch, rewards, terminated, mask)
            stats["td_reference"] = td_ref.item()
            loss = 0.5 * (td + td_ref)
        else:
            loss = td
        if self.variant in ("full", "no_reference"):
            cv = self.conditional_value_loss(live, mask, batch)
            stats["cv"] = cv.item()
            loss = loss + self.cv_coef * cv

        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite loss")
        self.optimiser.zero_grad(set_to_none=True)
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(self.params, self.grad_clip)
        self.optimiser.step()
        self.updates += 1
        if self.updates - self.last_target_update >= self.target_update_interval:
            self.target_mac.load_state(self.mac)
            self.target_mixer.load_state_dict(self.mixer.state_dict())
            self.last_target_update = self.updates
        stats.update(loss=loss.item(), grad_norm=float(grad_norm))
        return stats

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        self.mac.save(path)
        torch.save(self.mixer.state_dict(), path / "mixer.pt")
        torch.save({
            "optimiser": self.optimiser.state_dict(),
            "target_mac": {k: m.state_dict() for k, m in self.target_mac.modules().items()},
            "target_mixer": self.target_mixer.state_dict(),
            "reward_stats": (self.reward_stats.mean, self.reward_stats.var, self.reward_stats.count),
            "updates": self.updates,
            "last_target_update": self.last_target_update,
            "rng": (random.getstate(), np.random.get_state(), torch.get_rng_state()),
        }, path / "training_state.pt")
