"""Exact codebook search for the offline implementation.

A codebook maps T sender types onto K symbols, i.e. it is a partition of the
types into K non-empty groups. For T = 12 and K = 4 there are S(12, 4) =
611,501 such partitions, few enough to score all of them.

Given the shrunk conditional value table ``mu[t, c, :]`` (type x receiver
condition x receiver action) and the empirical mass ``p[t, c]``, the distortion
of a codebook e is

    D(e) = sum_{t,c} p[t,c] * || mu[t,c] - mean_{t' : e(t') = e(t)} mu[t',c] ||^2,

where the group mean is mass-weighted within each receiver condition. Up to a
constant, minimising D is the same as maximising

    sum_groups sum_c || sum_{t in group} p[t,c] mu[t,c] ||^2 / sum_{t in group} p[t,c],

which only depends on subset sums; we tabulate it for all 2^T subsets once and
then score every partition by table look-ups.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np


@lru_cache(maxsize=8)
def set_partitions(n_items: int, n_groups: int) -> np.ndarray:
    """All partitions of ``n_items`` into exactly ``n_groups`` non-empty groups.

    Partitions are returned as restricted growth strings in lexicographic
    order, shape ``(S(n_items, n_groups), n_items)``, dtype int8.
    """
    if not 1 <= n_groups <= n_items <= 16:
        raise ValueError("need 1 <= n_groups <= n_items <= 16")
    strings = np.zeros((1, 1), dtype=np.int8)
    for position in range(1, n_items):
        remaining = n_items - position - 1
        peak = strings.max(1)
        rows, values = [], []
        for value in range(min(position, n_groups - 1) + 1):
            new_peak = np.maximum(peak, value)
            ok = (value <= peak + 1) & (new_peak + 1 + remaining >= n_groups)
            rows.append(np.flatnonzero(ok))
            values.append(np.full(ok.sum(), value, dtype=np.int8))
        order = np.lexsort((np.concatenate(values), np.concatenate(rows)))
        index = np.concatenate(rows)[order]
        strings = np.concatenate((strings[index], np.concatenate(values)[order, None]), 1)
    return strings[strings.max(1) == n_groups - 1]


def partition_masks(partitions: np.ndarray, n_groups: int) -> np.ndarray:
    """Bit mask of every group of every partition, shape ``(P, n_groups)``."""
    powers = 1 << np.arange(partitions.shape[1], dtype=np.int64)
    return np.stack([(partitions == k) @ powers for k in range(n_groups)], 1)


@dataclass
class CodebookResult:
    code: np.ndarray       # symbol of each sender type, shape (T,)
    distortion: float      # D(code) on the shrunk table
    energy: float          # sum_{t,c} p[t,c] ||mu[t,c]||^2, the distortion of sending nothing useful
    partitions_scored: int
    ties: int


def shrunk_table(target: np.ndarray, types: np.ndarray, conditions: np.ndarray,
                 n_types: int, n_conditions: int, shrinkage: float = 32.0):
    """Conditional mean of ``target`` per (type, condition), shrunk towards the type mean.

    Returns ``(table, mass)`` with shapes ``(T, C, A)`` and ``(T, C)``.
    """
    counts = np.bincount(types * n_conditions + conditions,
                         minlength=n_types * n_conditions).reshape(n_types, n_conditions)
    if np.any(counts.sum(1) == 0):
        raise ValueError("every sender type needs at least one sample")
    sums = np.zeros((n_types, n_conditions, target.shape[1]))
    np.add.at(sums, (types, conditions), target)
    prior = sums.sum(1) / counts.sum(1, keepdims=True)
    table = (sums + shrinkage * prior[:, None]) / (counts[:, :, None] + shrinkage)
    return table, counts / counts.sum()


def solve_codebook(table: np.ndarray, mass: np.ndarray, n_symbols: int) -> CodebookResult:
    """Exhaustively find the codebook with the smallest distortion.

    Ties are broken by the lexicographic order of the partitions.
    """
    n_types = table.shape[0]
    stats = table * mass[..., None]
    bits = ((np.arange(1 << n_types)[:, None] >> np.arange(n_types)) & 1).astype(np.float64)
    totals = (bits @ stats.reshape(n_types, -1)).reshape(-1, *stats.shape[1:])
    weights = bits @ mass
    subset_score = np.divide(np.square(totals).sum(-1), weights,
                             out=np.zeros_like(weights), where=weights > 0).sum(-1)

    partitions = set_partitions(n_types, n_symbols)
    scores = subset_score[partition_masks(partitions, n_symbols)].sum(1)
    best = int(scores.argmax())
    code = partitions[best].astype(np.int64)

    energy = float((mass[..., None] * np.square(table)).sum())
    distortion = energy - float(scores[best])
    return CodebookResult(
        code=code,
        distortion=distortion,
        energy=energy,
        partitions_scored=len(partitions),
        ties=int(np.sum(np.abs(scores - scores[best]) <= 1e-12)),
    )


def codebook_distortion(table: np.ndarray, mass: np.ndarray, code: np.ndarray) -> float:
    """Direct evaluation of D(code); used for checks and diagnostics."""
    distortion = 0.0
    for symbol in np.unique(code):
        group = code == symbol
        weight = mass[group].sum(0)
        centre = np.divide((mass[group, :, None] * table[group]).sum(0), weight[:, None],
                           out=np.zeros(table.shape[1:]), where=weight[:, None] > 0)
        distortion += float((mass[group, :, None] * np.square(table[group] - centre)).sum())
    return distortion
