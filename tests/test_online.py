import copy

import pytest
import torch

from raven.online.buffer import EpisodeBatch
from raven.online.controller import communication_graph
from raven.online.runner import build


class RandomTask:
    """Synthetic task with the adapter interface, for mechanics tests."""

    def __init__(self, n_agents=4, n_actions=6, obs_dim=9, episode_limit=12, topology="all"):
        self.n_agents, self.n_actions, self.obs_dim = n_agents, n_actions, obs_dim
        self.state_dim = 2 * obs_dim * n_agents
        self.episode_limit = episode_limit
        self.topology = topology


def random_batch(task, batch=5, seed=0):
    g = torch.Generator().manual_seed(seed)
    length = task.episode_limit + 1
    b = EpisodeBatch.empty(batch, length, task.n_agents, task.obs_dim, task.state_dim, task.n_actions)
    b.obs.normal_(generator=g)
    b.state.normal_(generator=g)
    b.avail_actions.fill_(1)
    b.avail_actions[:, :, 0, -1] = 0
    actions = torch.randint(task.n_actions - 1, (batch, length, task.n_agents), generator=g)
    for t in range(length):
        b.record_actions(t, actions[:, t])
    b.reward.normal_(generator=g)
    b.filled.fill_(1)
    b.filled[0, 7:] = 0
    b.terminated[0, 6] = 1
    return b


@pytest.fixture
def model():
    task = RandomTask()
    mac, learner = build(task, {}, seed=5)
    # give the zero-initialised interaction head some weight so messages matter
    torch.nn.init.normal_(mac.receiver.interaction[-1].weight, std=0.5)
    return task, mac, learner


def step_through(mac, batch):
    mac.init_hidden(batch.batch_size)
    return torch.stack([mac.forward(batch, t) for t in range(batch.max_seq_length)], 1)


def test_unroll_equals_decentralised_steps(model):
    task, mac, _ = model
    batch = random_batch(task)
    with torch.no_grad():
        torch.testing.assert_close(step_through(mac, batch), mac.unroll(batch)["q"], atol=2e-6, rtol=2e-5)


def test_messages_are_delayed_by_one_step(model):
    task, mac, _ = model
    batch = random_batch(task)
    altered = batch.clone()
    altered.obs[:, 4, 2] += 5.0         # change agent 2's input at t = 4
    with torch.no_grad():
        q, q2 = mac.unroll(batch)["q"], mac.unroll(altered)["q"]
    others = [i for i in range(task.n_agents) if i != 2]
    assert torch.equal(q[:, :5][:, :, others], q2[:, :5][:, :, others])
    assert not torch.equal(q[:, 5][:, others], q2[:, 5][:, others])


def test_global_state_never_reaches_the_policy(model):
    task, mac, _ = model
    batch = random_batch(task)
    altered = batch.clone()
    altered.state.add_(100.0)
    with torch.no_grad():
        assert torch.equal(mac.unroll(batch, True)["q"], mac.unroll(altered, True)["q"])


def test_cv_gradient_reaches_only_the_codec(model):
    task, mac, learner = model
    batch = random_batch(task)
    mask = batch.filled[:, :-1].clone()
    loss = learner.conditional_value_loss(mac.unroll(batch, True), mask, batch)
    loss.backward()
    codec = list(mac.codec.parameters())
    frozen = list(mac.agent.parameters()) + list(mac.receiver.parameters()) + list(learner.mixer.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in codec)
    assert all(p.grad is None or p.grad.abs().sum() == 0 for p in frozen)


def test_ring_graph_restricts_senders():
    allowed = communication_graph(4, "ring")
    assert allowed.sum() == 4 and bool(allowed[0, 3]) and bool(allowed[2, 1])
    task = RandomTask(topology="ring")
    mac, _ = build(task, {}, seed=1)
    torch.nn.init.normal_(mac.receiver.interaction[-1].weight, std=0.5)
    batch = random_batch(task)
    altered = batch.clone()
    altered.obs[:, 3, 2] += 5.0         # agent 2 only talks to agent 3
    with torch.no_grad():
        q, q2 = mac.unroll(batch)["q"], mac.unroll(altered)["q"]
    assert torch.equal(q[:, 4, [0, 1]], q2[:, 4, [0, 1]])
    assert not torch.equal(q[:, 4, 3], q2[:, 4, 3])


@pytest.mark.parametrize("variant", ["full", "no_cv", "no_reference", "no_comm"])
def test_training_step_and_checkpoint(variant, tmp_path):
    task = RandomTask()
    mac, learner = build(task, {"variant": variant, "target_update_interval": 2}, seed=2)
    batch = random_batch(task)
    before = copy.deepcopy(mac.agent.state_dict())
    for _ in range(3):
        stats = learner.train(batch)
    assert all(torch.isfinite(torch.tensor(v)) for v in stats.values())
    assert any(not torch.equal(before[k], v) for k, v in mac.agent.state_dict().items())
    learner.save(tmp_path)
    fresh, _ = build(task, {"variant": variant}, seed=99)
    fresh.load(tmp_path)
    with torch.no_grad():
        assert torch.equal(mac.unroll(batch)["q"], fresh.unroll(batch)["q"])


def test_same_backbone_initialisation():
    task = RandomTask()
    full, full_learner = build(task, {}, seed=11)
    plain, plain_learner = build(task, {"variant": "no_comm"}, seed=11)
    for a, b in zip(full.agent.parameters(), plain.agent.parameters(), strict=False):
        assert torch.equal(a, b)
    for a, b in zip(full_learner.mixer.parameters(), plain_learner.mixer.parameters(), strict=False):
        assert torch.equal(a, b)
