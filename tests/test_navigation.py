import numpy as np
import pytest

from raven.envs.navigation import NavigationWorld, actions_to_joint, joint_to_actions


def test_joint_action_round_trip():
    actions = np.random.default_rng(0).integers(5, size=(50, 4))
    assert np.array_equal(joint_to_actions(actions_to_joint(actions), 4), actions)


def test_ring_goal_is_held_by_predecessor():
    world = NavigationWorld("ring", 4, [11])
    colours = world.observation()[0, :, 8:11].argmax(1)
    world.positions[:] = world.landmarks[0, np.roll(colours, 1)]
    world.velocity[:] = 0
    _, reward, _ = world.step(np.zeros((1, 4), dtype=int))
    assert np.allclose(reward, 0.0)


@pytest.mark.parametrize("family", ["ring", "broadcast"])
def test_matches_mpe2(family):
    pytest.importorskip("mpe2")
    from mpe2 import simple_reference_v3, simple_speaker_listener_v4

    for seed in (0, 7, 123):
        if family == "ring":
            env = simple_reference_v3.parallel_env(max_cycles=25, local_ratio=0)
            world = NavigationWorld("ring", 2, [seed])
        else:
            env = simple_speaker_listener_v4.parallel_env(max_cycles=25)
            world = NavigationWorld("broadcast", 1, [seed])
        obs, _ = env.reset(seed=seed)
        rng = np.random.default_rng(seed + 1)
        try:
            for t in range(25):
                if family == "ring":
                    native = np.stack([obs[f"agent_{i}"][:11] for i in range(2)])[None]
                else:
                    native = obs["listener_0"][None, None, :8]
                assert np.array_equal(native, world.observation())
                actions = rng.integers(5, size=(1, world.n_agents))
                if family == "ring":
                    commands = {f"agent_{i}": int(actions[0, i]) for i in range(2)}
                else:
                    commands = {"speaker_0": 0, "listener_0": int(actions[0, 0])}
                obs, reward, _, _, _ = env.step(commands)
                _, ours, done = world.step(actions)
                assert abs(float(np.mean(list(reward.values()))) - float(ours.mean())) < 1e-12
                assert done == (t == 24)
        finally:
            env.close()
