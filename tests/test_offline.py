import torch

from raven.offline.pipeline import run_offline
from raven.offline.policy import OfflinePolicy, joint_expectation


def test_joint_expectation_matches_explicit_sum():
    torch.manual_seed(0)
    n = 3
    q = torch.randn(4, 5 ** n)
    probs = torch.randn(4, n, 5).softmax(-1)
    explicit = probs[:, 0]
    for i in range(1, n):
        explicit = (explicit[:, :, None] * probs[:, i, None, :]).flatten(1)
    torch.testing.assert_close(joint_expectation(q, probs), (q * explicit).sum(-1))


def test_sender_reads_only_its_own_observation():
    policy = OfflinePolicy(3, 11, 4, seed=0)
    x = torch.randn(16, 3, 11)
    altered = x.clone()
    altered[:, 1] += 3.0
    assert torch.equal(policy.symbols(x)[:, [0, 2]], policy.symbols(altered)[:, [0, 2]])


def test_small_end_to_end_run(tmp_path):
    config = {
        "seed": 3, "n_agents": 3, "n_symbols": 4,
        "teacher": {"head": "additive", "steps": 8000, "warmup": 1000, "construction_rows": 2000},
        "policy": {"sender_updates": 20, "receiver_updates": 40, "batch_size": 64},
        "evaluation": {"episodes": 8},
    }
    summary = run_offline(config, tmp_path)
    assert len(summary["codebooks"]) == 3
    assert all(len(set(c["code"])) == 4 for c in summary["codebooks"])
    assert (tmp_path / "policy.pt").exists() and (tmp_path / "summary.json").exists()
