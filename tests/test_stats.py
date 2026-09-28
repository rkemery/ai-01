from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
import pytest
from scipy import stats as sps

from llm_eval_harness import stats


@pytest.mark.parametrize(
    ("k", "n"), [(0, 10), (1, 10), (5, 10), (10, 10), (7, 40), (39, 40), (3, 3)]
)
def test_wilson_matches_scipy(k: int, n: int) -> None:
    ours = stats.wilson_interval(k, n)
    ref = sps.binomtest(k, n).proportion_ci(confidence_level=0.95, method="wilson")
    assert ours.low == pytest.approx(ref.low, abs=1e-12)
    assert ours.high == pytest.approx(ref.high, abs=1e-12)
    assert ours.estimate == k / n


def test_wilson_known_value_zero_successes() -> None:
    # 0 of 10: upper bound z^2 / (n + z^2) = 3.8415 / 13.8415
    interval = stats.wilson_interval(0, 10)
    assert interval.low == 0.0
    assert interval.high == pytest.approx(0.27753, abs=1e-5)


@pytest.mark.parametrize(("k", "n"), [(-1, 10), (11, 10), (0, 0)])
def test_wilson_rejects_bad_counts(k: int, n: int) -> None:
    with pytest.raises(ValueError, match="must be"):
        stats.wilson_interval(k, n)


def test_clustered_se_without_clusters_is_ordinary_se() -> None:
    x = np.random.default_rng(1).normal(size=37)
    assert stats.clustered_se(x) == pytest.approx(sps.sem(x))


def test_clustered_se_with_singleton_clusters_is_ordinary_se() -> None:
    x = np.random.default_rng(2).integers(0, 2, size=25).astype(float)
    singletons = [f"c{i}" for i in range(x.size)]
    assert stats.clustered_se(x, singletons) == pytest.approx(sps.sem(x), rel=1e-12)
    none_labels = [None] * x.size  # None means "its own cluster"
    assert stats.clustered_se(x, none_labels) == pytest.approx(sps.sem(x), rel=1e-12)


def test_clustered_se_matches_the_pairwise_formula() -> None:
    rng = np.random.default_rng(3)
    x = rng.normal(size=12)
    clusters = ["a"] * 4 + ["b"] * 3 + ["c"] * 5
    n = x.size
    d = x - x.mean()
    cross = sum(
        d[i] * d[j] for i, j in itertools.permutations(range(n), 2) if clusters[i] == clusters[j]
    )
    expected = math.sqrt(np.var(x, ddof=1) / n + cross / n**2)
    assert stats.clustered_se(x, clusters) == pytest.approx(expected, rel=1e-12)


def test_clustering_widens_se_for_correlated_items() -> None:
    # Whole clusters pass or fail together, so there are really only 6 observations.
    x = np.repeat([1, 0, 1, 1, 0, 1], 5).astype(float)
    clusters = np.repeat(list("abcdef"), 5).tolist()
    assert stats.clustered_se(x, clusters) > 2 * stats.clustered_se(x)


def test_clustered_se_needs_two_clusters() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        stats.clustered_se([1.0, 0.0, 1.0], ["a", "a", "a"])


def test_mean_ci_is_symmetric_normal_interval() -> None:
    x = np.array([0.2, 0.4, 0.9, 0.5, 0.7])
    interval = stats.mean_ci(x)
    half = sps.norm.ppf(0.975) * sps.sem(x)
    assert interval.low == pytest.approx(x.mean() - half)
    assert interval.high == pytest.approx(x.mean() + half)


def test_design_effect_is_one_without_variance_and_at_least_one() -> None:
    assert stats.design_effect([0.0] * 10, list("aabbccddee")) == 1.0
    rng = np.random.default_rng(4)
    x = rng.integers(0, 2, size=40).astype(float)
    assert stats.design_effect(x, [f"c{i % 8}" for i in range(40)]) >= 1.0


def test_clustered_wilson_reduces_to_wilson_for_singletons() -> None:
    x = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    ours = stats.wilson_interval_clustered(x, [str(i) for i in range(x.size)])
    plain = stats.wilson_interval(int(x.sum()), x.size)
    assert ours.low == pytest.approx(plain.low, rel=1e-9)
    assert ours.high == pytest.approx(plain.high, rel=1e-9)


def test_clustered_wilson_stays_open_at_zero() -> None:
    interval = stats.wilson_interval_clustered([0.0] * 40, [f"c{i % 8}" for i in range(40)])
    assert interval.low == 0.0
    assert interval.high == pytest.approx(stats.wilson_interval(0, 40).high)


def test_clustered_wilson_is_wider_when_clusters_matter() -> None:
    x = np.repeat([1, 0, 1, 1, 0, 1, 1, 0], 5).astype(float)
    clusters = np.repeat(list("abcdefgh"), 5).tolist()
    clustered = stats.wilson_interval_clustered(x, clusters)
    plain = stats.wilson_interval(int(x.sum()), x.size)
    assert clustered.high - clustered.low > plain.high - plain.low


def test_mcnemar_known_value_and_scipy() -> None:
    # 1 pair where only A passes, 9 where only B passes: p = 2 * (1 + 10) / 2**10
    a = [True] * 1 + [False] * 9 + [True] * 5
    b = [False] * 1 + [True] * 9 + [True] * 5
    result = stats.mcnemar_exact(a, b)
    assert (result.a_only, result.b_only) == (1, 9)
    assert result.pvalue == pytest.approx(22 / 1024)
    assert result.pvalue == pytest.approx(sps.binomtest(1, 10, 0.5).pvalue)


def test_mcnemar_no_discordant_pairs() -> None:
    assert stats.mcnemar_exact([1, 0, 1], [1, 0, 1]).pvalue == 1.0


def test_mcnemar_rejects_non_binary() -> None:
    with pytest.raises(ValueError, match="binary"):
        stats.mcnemar_exact([0.5, 1.0], [1.0, 1.0])


def test_paired_bootstrap_is_seeded_and_centered() -> None:
    rng = np.random.default_rng(5)
    a = rng.integers(0, 2, size=60)
    b = np.where(rng.uniform(size=60) < 0.8, a, 1 - a)
    first = stats.paired_bootstrap(a, b, n_boot=2000, seed=7)
    again = stats.paired_bootstrap(a, b, n_boot=2000, seed=7)
    other = stats.paired_bootstrap(a, b, n_boot=2000, seed=8)
    assert first == again
    assert other.diff == first.diff
    d = b - a
    assert not np.array_equal(stats.bootstrap_means(d, seed=7), stats.bootstrap_means(d, seed=8))
    assert first.diff == pytest.approx(b.mean() - a.mean())
    assert first.low <= first.diff <= first.high
    assert first.mcnemar is not None


def test_paired_bootstrap_cluster_resampling_widens_ci() -> None:
    # Differences are identical within each cluster, so resampling items overstates precision.
    d = np.repeat([1, 0, 0, 1, -1, 1, 0, 1], 6).astype(float)
    clusters = np.repeat(list("abcdefgh"), 6).tolist()
    base = np.zeros_like(d)
    items = stats.paired_bootstrap(base, d, n_boot=4000, seed=0)
    grouped = stats.paired_bootstrap(base, d, clusters, n_boot=4000, seed=0)
    assert grouped.high - grouped.low > 1.5 * (items.high - items.low)
    assert grouped.n_clusters == 8
    assert grouped.mcnemar is None  # the differences include -1, so not binary pairs of 0/1


def test_bootstrap_means_handles_partial_chunks() -> None:
    reps = stats.bootstrap_means(np.arange(10.0), n_boot=2_345, seed=1)
    assert reps.shape == (2_345,)
    assert np.all((reps >= 0) & (reps <= 9))


def test_mde_paired_binary_known_value() -> None:
    expected = (sps.norm.ppf(0.975) + sps.norm.ppf(0.8)) * math.sqrt(0.2 / 150)
    assert stats.mde_paired_binary(150, 0.2) == pytest.approx(expected)
    assert stats.mde_paired_binary(150, 0.2) == pytest.approx(0.1023, abs=1e-4)


def test_mde_paired_binary_is_conservative_vs_connor() -> None:
    # Solve Connor (1987) exactly: delta sqrt(n) = z_a sqrt(p_d) + z_b sqrt(p_d - delta^2).
    n, p_d = 100, 0.3
    z_a, z_b = sps.norm.ppf(0.975), sps.norm.ppf(0.8)
    exact = next(
        delta
        for delta in np.linspace(0.001, p_d, 100_000)
        if delta * math.sqrt(n) >= z_a * math.sqrt(p_d) + z_b * math.sqrt(p_d - delta**2)
    )
    assert exact <= stats.mde_paired_binary(n, p_d) <= exact * 1.05


def test_mde_two_proportions_known_value() -> None:
    expected = 2.8016 * math.sqrt(0.7 * 0.3 * (2 / 40))
    assert stats.mde_two_proportions(40, 40, 0.7) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (stats.mde_paired_binary, (0, 0.2)),
        (stats.mde_paired_binary, (10, 0.0)),
        (stats.mde_two_proportions, (0, 10, 0.5)),
        (stats.mde_two_proportions, (10, 10, 1.5)),
    ],
)
def test_mde_rejects_bad_inputs(fn: Callable[..., float], args: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match="must be"):
        fn(*args)


def test_pass_hat_k_and_pass_at_k_exact_values() -> None:
    assert stats.pass_hat_k(4, 2, 2) == pytest.approx(1 / 6)
    assert stats.pass_at_k(4, 2, 2) == pytest.approx(5 / 6)
    assert stats.pass_hat_k(5, 3, 1) == stats.pass_at_k(5, 3, 1) == pytest.approx(0.6)
    assert stats.pass_hat_k(4, 4, 3) == 1.0
    assert stats.pass_at_k(4, 0, 3) == 0.0


def test_pass_k_matches_brute_force_over_subsets() -> None:
    outcomes = [True, False, True, True, False, True]
    n, c = len(outcomes), sum(outcomes)
    for k in range(1, n + 1):
        subsets = list(itertools.combinations(outcomes, k))
        assert stats.pass_hat_k(n, c, k) == pytest.approx(np.mean([all(s) for s in subsets]))
        assert stats.pass_at_k(n, c, k) == pytest.approx(np.mean([any(s) for s in subsets]))


def test_pass_k_averages_over_tasks() -> None:
    result = stats.pass_k([(4, 4), (4, 2), (4, 0)], k=2, n_boot=500)
    assert result.pass_hat_k.estimate == pytest.approx((1 + 1 / 6 + 0) / 3)
    assert result.pass_at_k.estimate == pytest.approx((1 + 5 / 6 + 0) / 3)
    assert result.n_tasks == 3


@pytest.mark.parametrize(("n", "c", "k"), [(3, 4, 1), (3, 2, 4), (3, 2, 0), (-1, 0, 1)])
def test_pass_k_rejects_bad_counts(n: int, c: int, k: int) -> None:
    with pytest.raises(ValueError, match=r"must be|exceed"):
        stats.pass_hat_k(n, c, k)


def test_percentile_interval_rejects_all_nan() -> None:
    with pytest.raises(ValueError, match="NaN"):
        stats.percentile_interval([np.nan, np.nan], 0.5, 2)
