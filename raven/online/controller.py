"""Decentralised controller of the online implementation.

At time t every agent updates its recurrent state h_t from its own inputs,
emits a hard symbol m_t = argmax(W h_t) (2 bits for K = 4), and acts on

    Q_i(., h_t) = W_o h_t + Delta_i(h_t, symbols that arrived from t - 1).

Symbols emitted at t arrive at t + 1; at t = 0 the mailbox is empty. The
controller never reads the global state or another agent's recurrent state.
During training ``unroll(..., reference=True)`` also returns the reference
branch: the same receiver, with routing keys still computed from the symbols,
but aggregating the senders' continuous states h_{t-1} instead of symbols.
The reference branch is a training device only.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.distributions import Categorical

from raven.online.buffer import EpisodeBatch
from raven.online.networks import Receiver, RNNAgent, SymbolCodec


def communication_graph(n_agents: int, topology: str | list) -> torch.Tensor:
    """allowed[i, j] is True when receiver i may read sender j."""
    if topology == "all":
        return ~torch.eye(n_agents, dtype=torch.bool)
    allowed = torch.zeros(n_agents, n_agents, dtype=torch.bool)
    if topology == "ring":
        allowed[torch.arange(n_agents), (torch.arange(n_agents) - 1) % n_agents] = True
    elif topology == "star":  # agent 0 speaks, all others listen
        allowed[1:, 0] = True
    else:
        for sender, receiver in topology:
            allowed[receiver, sender] = True
    return allowed


class EpsilonGreedy:
    def __init__(self, start: float = 1.0, finish: float = 0.05, anneal_steps: int = 50_000):
        self.start, self.finish, self.anneal_steps = start, finish, anneal_steps

    def epsilon(self, t_env: int) -> float:
        return max(self.finish, self.start - (self.start - self.finish) * t_env / self.anneal_steps)

    def select(self, q: torch.Tensor, avail: torch.Tensor, t_env: int, greedy: bool) -> torch.Tensor:
        masked = q.masked_fill(avail == 0, -torch.inf)
        chosen = masked.argmax(-1)
        if greedy:
            return chosen
        explore = torch.rand(q.shape[:-1], device=q.device) < self.epsilon(t_env)
        random_actions = Categorical(avail.float()).sample()
        return torch.where(explore, random_actions, chosen)


class RavenController:
    def __init__(self, n_agents: int, obs_dim: int, n_actions: int, *, hidden_dim: int = 64,
                 n_symbols: int = 4, top_k: int = 2, key_dim: int = 32, topology="all",
                 communicate: bool = True, epsilon: EpsilonGreedy | None = None, device="cpu"):
        self.n_agents, self.obs_dim, self.n_actions = n_agents, obs_dim, n_actions
        self.hidden_dim, self.n_symbols = hidden_dim, n_symbols
        self.communicate = communicate
        self.device = torch.device(device)
        self.agent = RNNAgent(obs_dim + n_actions + n_agents, hidden_dim, n_actions).to(self.device)
        if communicate:
            self.codec = SymbolCodec(hidden_dim, n_symbols).to(self.device)
            self.receiver = Receiver(hidden_dim, n_actions, key_dim, top_k).to(self.device)
        else:
            self.codec = self.receiver = None
        self.allowed = communication_graph(n_agents, topology).to(self.device)
        self.topology = topology
        self.selector = epsilon or EpsilonGreedy()
        self.drop_messages = False
        self.hidden = None
        self.mailbox = None
        self.last_symbols = None
        self.last_weights = None

    # ------------------------------------------------------------------ modules
    def modules(self) -> dict[str, torch.nn.Module]:
        out = {"agent": self.agent}
        if self.communicate:
            out.update(codec=self.codec, receiver=self.receiver)
        return out

    def parameters(self) -> list[torch.nn.Parameter]:
        return [p for m in self.modules().values() for p in m.parameters()]

    def communication_parameters(self) -> list[torch.nn.Parameter]:
        if not self.communicate:
            return []
        return list(self.codec.parameters()) + list(self.receiver.parameters())

    def load_state(self, other: RavenController) -> None:
        for name, module in self.modules().items():
            module.load_state_dict(other.modules()[name].state_dict())

    # ------------------------------------------------------------------ inputs
    def build_inputs(self, batch: EpisodeBatch, t: int) -> torch.Tensor:
        b = batch.batch_size
        last_action = (torch.zeros_like(batch.actions_onehot[:, 0]) if t == 0
                       else batch.actions_onehot[:, t - 1])
        agent_id = torch.eye(self.n_agents, device=batch.device).expand(b, -1, -1)
        return torch.cat((batch.obs[:, t], last_action, agent_id), -1)

    # ------------------------------------------------------------------ execution
    def init_hidden(self, batch_size: int) -> None:
        self.hidden = self.agent.initial_state(batch_size, self.n_agents)
        self.mailbox = None
        self.last_symbols = None
        self.last_weights = None

    def forward(self, batch: EpisodeBatch, t: int) -> torch.Tensor:
        """One decentralised step; returns Q-values of shape (B, N, n_actions)."""
        self.hidden = self.agent(self.build_inputs(batch, t), self.hidden)
        q = self.agent.local_q(self.hidden)
        if not self.communicate:
            return q
        if self.mailbox is not None and not self.drop_messages:
            delta, self.last_weights = self.receiver(self.hidden, self.mailbox, self.mailbox, self.allowed)
            q = q + delta
        else:
            self.last_weights = None
        self.mailbox, self.last_symbols = self.codec(self.hidden)
        return q

    @torch.no_grad()
    def select_actions(self, batch: EpisodeBatch, t: int, t_env: int, greedy: bool = False) -> torch.Tensor:
        q = self.forward(batch, t)
        return self.selector.select(q, batch.avail_actions[:, t], t_env, greedy)

    # ------------------------------------------------------------------ training
    def unroll(self, batch: EpisodeBatch, reference: bool = False) -> dict[str, torch.Tensor]:
        """Recompute a whole batch of episodes; identical to stepping ``forward``."""
        hidden = self.agent.initial_state(batch.batch_size, self.n_agents)
        states = []
        for t in range(batch.max_seq_length):
            hidden = self.agent(self.build_inputs(batch, t), hidden)
            states.append(hidden)
        h = torch.stack(states, 1)                                   # (B, T, N, H)
        local = self.agent.local_q(h)
        out = {"q": local, "reference": None, "h": h}
        if not self.communicate:
            return out

        previous = torch.cat((torch.zeros_like(h[:, :1]), h[:, :-1]), 1)
        payload, symbols = self.codec(previous)
        delta, weights = self.receiver(h, payload, payload, self.allowed)
        arrived = torch.ones_like(delta[..., :1])
        arrived[:, 0] = 0
        if self.drop_messages:
            arrived.zero_()
        out.update(q=local + arrived * delta, previous=previous, payload=payload,
                   symbols=symbols, weights=weights)
        if reference:
            ref_delta, _ = self.receiver(h, payload, previous, self.allowed)
            out["reference"] = local + arrived * ref_delta
        return out

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save({name: m.state_dict() for name, m in self.modules().items()}, path / "controller.pt")
        meta = {"n_agents": self.n_agents, "obs_dim": self.obs_dim, "n_actions": self.n_actions,
                "hidden_dim": self.hidden_dim, "n_symbols": self.n_symbols,
                "communicate": self.communicate, "topology": self.topology, "message_delay": 1}
        (path / "controller.json").write_text(json.dumps(meta, indent=2) + "\n")

    def load(self, path: str | Path) -> None:
        path = Path(path)
        meta = json.loads((path / "controller.json").read_text())
        for key in ("n_agents", "obs_dim", "n_actions", "hidden_dim", "n_symbols", "communicate"):
            if meta[key] != getattr(self, key):
                raise ValueError(f"checkpoint mismatch for {key}: {meta[key]} != {getattr(self, key)}")
        state = torch.load(path / "controller.pt", map_location=self.device, weights_only=True)
        for name, module in self.modules().items():
            module.load_state_dict(state[name])
