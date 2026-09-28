"""Networks of the online implementation.

``RNNAgent`` and ``QMixer`` implement the standard QMIX architecture (Rashid
et al., 2018). They are adapted, with modifications, from PyMARL
(https://github.com/oxwhirl/pymarl, Apache License 2.0); see NOTICE and
licenses/Apache-2.0.txt. ``SymbolCodec`` and ``Receiver`` are part of RAVEN.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class RNNAgent(nn.Module):
    """Shared local network: input -> 64 -> GRU(64) -> action head."""

    def __init__(self, input_dim: int, hidden_dim: int, n_actions: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.rnn = nn.GRUCell(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_actions)

    def initial_state(self, batch: int, n_agents: int) -> torch.Tensor:
        return self.fc1.weight.new_zeros(batch, n_agents, self.hidden_dim)

    def forward(self, inputs: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        """Return the next hidden state; shapes (..., input_dim) and (..., hidden_dim)."""
        shape = hidden.shape
        x = F.relu(self.fc1(inputs.reshape(-1, inputs.shape[-1])))
        return self.rnn(x, hidden.reshape(-1, self.hidden_dim)).view(shape)

    def local_q(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.fc2(hidden)


class QMixer(nn.Module):
    """Monotonic mixing network conditioned on the global state (training only)."""

    def __init__(self, n_agents: int, state_dim: int, embed_dim: int = 32, hypernet_dim: int = 64):
        super().__init__()
        self.n_agents, self.state_dim, self.embed_dim = n_agents, state_dim, embed_dim
        self.hyper_w_1 = nn.Sequential(nn.Linear(state_dim, hypernet_dim), nn.ReLU(),
                                       nn.Linear(hypernet_dim, embed_dim * n_agents))
        self.hyper_w_final = nn.Sequential(nn.Linear(state_dim, hypernet_dim), nn.ReLU(),
                                           nn.Linear(hypernet_dim, embed_dim))
        self.hyper_b_1 = nn.Linear(state_dim, embed_dim)
        self.V = nn.Sequential(nn.Linear(state_dim, embed_dim), nn.ReLU(), nn.Linear(embed_dim, 1))

    def forward(self, agent_qs: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        """agent_qs: (B, T, n_agents), states: (B, T, state_dim) -> (B, T, 1)."""
        batch = agent_qs.shape[0]
        states = states.reshape(-1, self.state_dim)
        qs = agent_qs.reshape(-1, 1, self.n_agents)
        w1 = torch.abs(self.hyper_w_1(states)).view(-1, self.n_agents, self.embed_dim)
        b1 = self.hyper_b_1(states).view(-1, 1, self.embed_dim)
        hidden = F.elu(torch.bmm(qs, w1) + b1)
        w_final = torch.abs(self.hyper_w_final(states)).view(-1, self.embed_dim, 1)
        v = self.V(states).view(-1, 1, 1)
        return (torch.bmm(hidden, w_final) + v).view(batch, -1, 1)


class SymbolCodec(nn.Module):
    """K-ary hard symbols with a straight-through surrogate gradient.

    The forward value is exactly one of K embeddings; the softmax only shapes
    the gradient during training.
    """

    def __init__(self, hidden_dim: int, n_symbols: int = 4):
        super().__init__()
        self.n_symbols = n_symbols
        self.encoder = nn.Linear(hidden_dim, n_symbols)
        self.embedding = nn.Embedding(n_symbols, hidden_dim)

    def forward(self, hidden: torch.Tensor):
        logits = self.encoder(hidden)
        symbols = logits.argmax(-1)
        probs = logits.softmax(-1)
        one_hot = F.one_hot(symbols, self.n_symbols).to(probs.dtype) + probs - probs.detach()
        return one_hot @ self.embedding.weight, symbols

    def decode(self, symbols: torch.Tensor) -> torch.Tensor:
        return self.embedding(symbols)


class Receiver(nn.Module):
    """Sparse attention over arrived messages plus a nonlinear interaction residual.

    Each agent queries with its *private* recurrent state, keys are computed
    from the public symbol embeddings, and only the ``top_k`` highest-scoring
    permitted senders are kept. The residual ``Delta`` is added to the local
    action values; its last layer starts at zero, so training starts from
    plain local Q-values.
    """

    def __init__(self, hidden_dim: int, n_actions: int, key_dim: int = 32, top_k: int = 2):
        super().__init__()
        self.key_dim, self.top_k = key_dim, top_k
        self.norm = nn.LayerNorm(hidden_dim)
        self.query = nn.Linear(hidden_dim, key_dim, bias=False)
        self.key = nn.Linear(hidden_dim, key_dim, bias=False)
        self.value = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.interaction = nn.Sequential(nn.Linear(3 * hidden_dim, hidden_dim), nn.ReLU(),
                                         nn.Linear(hidden_dim, n_actions))
        nn.init.zeros_(self.interaction[-1].weight)
        nn.init.zeros_(self.interaction[-1].bias)

    def forward(self, private: torch.Tensor, public: torch.Tensor, payload: torch.Tensor,
                allowed: torch.Tensor):
        """Compute the residual for every receiver.

        private: (..., N, H) receivers' own recurrent states
        public:  (..., N, H) embeddings of the symbols that arrived (routing keys)
        payload: (..., N, H) values to aggregate (symbol embeddings when deployed)
        allowed: (N, N) boolean, allowed[i, j] = receiver i may read sender j
        """
        n = private.shape[-2]
        scores = torch.einsum("...id,...jd->...ij", self.query(private),
                              self.key(self.norm(public))) / self.key_dim ** 0.5
        allowed = allowed.expand_as(scores)
        k = min(self.top_k, max(n - 1, 1))
        top = scores.masked_fill(~allowed, -torch.inf).topk(k, dim=-1).indices
        selected = torch.zeros_like(allowed).scatter_(-1, top, True) & allowed
        weights = scores.masked_fill(~selected, -1e9).softmax(-1) * selected
        weights = weights / weights.sum(-1, keepdim=True).clamp_min(1e-8)
        context = torch.einsum("...ij,...jd->...id", weights, self.value(self.norm(payload)))
        residual = self.interaction(torch.cat((private, context, private * context), -1))
        has_sender = allowed.any(-1, keepdim=True)
        return residual * has_sender, weights
