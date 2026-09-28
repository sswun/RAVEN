import itertools

import numpy as np
import pytest

from raven.offline.codebook import codebook_distortion, set_partitions, shrunk_table, solve_codebook


def stirling2(n, k):
    table = [[0] * (k + 1) for _ in range(n + 1)]
    table[0][0] = 1
    for i in range(1, n + 1):
        for j in range(1, k + 1):
            table[i][j] = j * table[i - 1][j] + table[i - 1][j - 1]
    return table[n][k]


@pytest.mark.parametrize("n,k", [(1, 1), (4, 2), (6, 3), (8, 4), (12, 2), (12, 4)])
def test_partition_count_and_order(n, k):
    parts = set_partitions(n, k)
    assert len(parts) == stirling2(n, k)
    assert np.all(parts.max(1) == k - 1)
    # restricted growth strings: each value at most one above the running maximum
    running = np.maximum.accumulate(parts, axis=1)
    assert np.all(parts[:, 1:] <= running[:, :-1] + 1)
    keys = [tuple(row) for row in parts]
    assert keys == sorted(keys) and len(set(keys)) == len(keys)


def test_solver_matches_brute_force():
    rng = np.random.default_rng(3)
    n_types, n_cond, n_actions, k = 7, 3, 5, 3
    table = rng.normal(size=(n_types, n_cond, n_actions))
    mass = rng.random((n_types, n_cond))
    mass /= mass.sum()
    result = solve_codebook(table, mass, k)
    best = min(codebook_distortion(table, mass, np.array(code))
               for code in itertools.product(range(k), repeat=n_types) if len(set(code)) == k)
    assert result.distortion == pytest.approx(best, abs=1e-10)
    assert codebook_distortion(table, mass, result.code) == pytest.approx(best, abs=1e-10)


def test_shrinkage_pulls_sparse_cells_to_type_mean():
    target = np.array([[1.0, -1.0], [1.0, -1.0], [3.0, -3.0]])
    types = np.array([0, 0, 0])
    conditions = np.array([0, 0, 1])
    table, mass = shrunk_table(target, types, conditions, 1, 2, shrinkage=32)
    prior = target.mean(0)
    assert np.allclose(table[0, 1], (target[2] + 32 * prior) / 33)
    assert mass.sum() == pytest.approx(1.0)
