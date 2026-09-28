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


def test_clustered_se_is_cr1() -> None:
    """CR1: G / (G - 1) times the sandwich sum of every within-cluster pair, over n^2."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=12)
    clusters = ["a"] * 4 + ["b"] * 3 + ["c"] * 5
    n, g = x.size, 3
    d = x - x.mean()
    same = sum(
        d[i] * d[j] for i, j in itertools.product(range(n), repeat=2) if clusters[i] == clusters[j]
    )
    expected = math.sqrt(g / (g - 1) * same) / n
    assert stats.clustered_se(x, clusters) == pytest.approx(expected, rel=1e-12)


def test_clustering_widens_se_for_correlated_items() -> None:
    # Whole clusters pass or fail together, so there are really only 6 observations.
    x = np.repeat([1, 0, 1, 1, 0, 1], 5).astype(float)
    clusters = np.repeat(list("abcdef"), 5).tolist()
    assert stats.clustered_se(x, clusters) > 2 * stats.clustered_se(x)


def test_clustered_se_needs_two_clusters() -> None:
    with pytest.raises(ValueError, match="at least 2 clusters"):
        stats.clustered_se([1.0, 0.0, 1.0], ["a", "a", "a"])


def test_mean_ci_is_a_t_interval() -> None:
    x = np.array([0.2, 0.4, 0.9, 0.5, 0.7])
    interval = stats.mean_ci(x)
    half = sps.t.ppf(0.975, 4) * sps.sem(x)
    assert interval.low == pytest.approx(x.mean() - half)
    assert interval.high == pytest.approx(x.mean() + half)


def test_clustered_mean_ci_uses_cr1_and_t_with_g_minus_1_df() -> None:
    rng = np.random.default_rng(6)
    labels = np.repeat(list("abcdefgh"), 5).tolist()
    x = rng.normal(size=8)[np.repeat(np.arange(8), 5)] + rng.normal(size=40)
    se = stats.clustered_se_floored(x, labels)
    assert se == stats.clustered_se(x, labels)  # strongly clustered, so the floor is inactive
    interval = stats.mean_ci(x, labels)
    assert interval.high == pytest.approx(x.mean() + sps.t.ppf(0.975, 7) * se)
    assert "8 clusters, 7 df" in interval.method


def test_design_effect_is_one_without_variance_and_at_least_one() -> None:
    assert stats.design_effect([0.0] * 10, list("aabbccddee")) == 1.0
    rng = np.random.default_rng(4)
    x = rng.integers(0, 2, size=40).astype(float)
    assert stats.design_effect(x, [f"c{i % 8}" for i in range(40)]) >= 1.0


def _wilson(p: float, n: float, z: float) -> tuple[float, float]:
    center = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z / (1 + z * z / n) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return center - half, center + half


def test_clustered_wilson_uses_korn_graubard_effective_n() -> None:
    # Singleton clusters: design effect 1, so only the df adjustment (z / t_{G-1})^2 remains.
    x = np.array([1, 1, 0, 1, 0, 1, 1, 1, 0, 1], dtype=float)
    ours = stats.wilson_interval_clustered(x, [str(i) for i in range(x.size)])
    z, t = sps.norm.ppf(0.975), sps.t.ppf(0.975, 9)
    low, high = _wilson(0.7, 10 * (z / t) ** 2, z)
    assert (ours.low, ours.high) == (pytest.approx(low), pytest.approx(high))
    assert "Korn-Graubard" in ours.method


def test_clustered_wilson_stays_open_at_zero() -> None:
    interval = stats.wilson_interval_clustered([0.0] * 40, [f"c{i % 8}" for i in range(40)])
    assert interval.low == 0.0
    # No failures means no evidence about clustering (deff = 1), but only 7 df.
    assert stats.wilson_interval(0, 40).high < interval.high < 0.15


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


def test_clustered_paired_comparison_widens_ci_and_matches_its_p_value() -> None:
    # Differences are identical within each cluster, so treating items as independent
    # overstates precision.
    d = np.repeat([1, 0, 0, 1, -1, 1, 0, 1], 6).astype(float)
    clusters = np.repeat(list("abcdefgh"), 6).tolist()
    base = np.zeros_like(d)
    items = stats.paired_comparison(base, d, n_boot=4000, seed=0)
    grouped = stats.paired_comparison(base, d, clusters)
    assert grouped.high - grouped.low > 1.5 * (items.high - items.low)
    assert grouped.n_clusters == 8
    assert (grouped.test, grouped.df, grouped.mcnemar) == ("clustered t-test", 7, None)
    se = stats.clustered_se(d, clusters)
    assert grouped.pvalue == pytest.approx(2 * sps.t.sf(d.mean() / se, 7))
    assert (grouped.low > 0) == (grouped.pvalue < 0.05)


def test_bootstrap_means_handles_partial_chunks() -> None:
    reps = stats.bootstrap_means(np.arange(10.0), n_boot=2_345, seed=1)
    assert reps.shape == (2_345,)
    assert np.all((reps >= 0) & (reps <= 9))


def test_mde_two_proportions_known_value() -> None:
    expected = 2.8016 * math.sqrt(0.7 * 0.3 * (2 / 40))
    assert stats.mde_two_proportions(40, 40, 0.7) == pytest.approx(expected, rel=1e-4)


@pytest.mark.parametrize(
    ("fn", "args"),
    [
        (stats.mde_paired_binary, (0, 0.2)),
        (stats.mde_paired_binary, (10, 0.0)),
        (stats.mcnemar_exact_power, (10, 0.2, 0.3)),
        (stats.mde_two_proportions, (0, 10, 0.5)),
        (stats.mde_two_proportions, (10, 10, 1.5)),
    ],
)
def test_mde_rejects_bad_inputs(fn: Callable[..., float], args: tuple[float, ...]) -> None:
    with pytest.raises(ValueError, match=r"must be|cannot exceed"):
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


# Review findings: the paired MDE must be attainable and match the exact McNemar test.


@pytest.mark.parametrize(
    ("n", "p_d"), [(40, 0.15), (30, 0.1), (20, 0.05), (39, 0.282), (100, 0.2), (10, 0.1)]
)
def test_mde_paired_binary_never_exceeds_the_discordant_rate(n: int, p_d: float) -> None:
    mde = stats.mde_paired_binary(n, p_d)
    assert mde is None or 0 < mde <= p_d


def test_mcnemar_exact_power_matches_enumeration() -> None:
    """Brute force over every (baseline-only, candidate-only) count at n = 20."""
    n = 20
    for p_d, diff in [(0.3, 0.1), (0.5, 0.3), (0.2, -0.2), (0.4, 0.0)]:
        p_a, p_b = (p_d - diff) / 2, (p_d + diff) / 2
        expected = 0.0
        for a in range(n + 1):
            for b in range(n + 1 - a):
                prob = sps.multinomial.pmf([a, b, n - a - b], n, [p_a, p_b, 1 - p_d])
                pvalue = 1.0 if a + b == 0 else sps.binomtest(a, a + b, 0.5).pvalue
                expected += prob * (pvalue < 0.05)
        assert stats.mcnemar_exact_power(n, p_d, diff) == pytest.approx(expected, abs=1e-9)


@pytest.mark.parametrize(("n", "p_d"), [(40, 0.3), (39, 0.282), (100, 0.2)])
def test_paired_binary_mde_has_80_percent_power_by_simulation(n: int, p_d: float) -> None:
    """At the MDE, `mcnemar_exact` rejects about 80% of the time, and just below it less."""
    mde = stats.mde_paired_binary(n, p_d)
    assert mde is not None
    assert stats.mcnemar_exact_power(n, p_d, mde) >= 0.8
    assert stats.mcnemar_exact_power(n, p_d, mde - 0.005) < 0.8
    rng = np.random.default_rng(n)
    p_a = (p_d - mde) / 2
    reps, rejections = 3000, 0
    for _ in range(reps):
        u = rng.uniform(size=n)
        both = u >= p_d + (1 - p_d) / 2  # half of the concordant pairs pass in both runs
        a = (u < p_a) | both
        b = ((u >= p_a) & (u < p_d)) | both
        rejections += stats.mcnemar_exact(a, b).pvalue < 0.05
    assert rejections / reps == pytest.approx(0.8, abs=0.03)


# Review findings: few-cluster intervals and the paired clustered path.


def _labels(g: int, m: int) -> list[str]:
    return np.repeat(np.arange(g), m).astype(str).tolist()


def test_few_cluster_intervals_cover_at_eight_clusters() -> None:
    """95% intervals at 8 clusters of 5 items (the demo's shape) should cover >= 93%."""
    rng = np.random.default_rng(2026)
    g, m, reps = 8, 5, 1000
    labels = _labels(g, m)
    member = np.repeat(np.arange(g), m)
    beta_a, beta_b = 0.8 * 4, 0.2 * 4  # pass rate 0.8, intra-cluster correlation 0.2
    hits = {"mean": 0, "wilson": 0, "paired": 0}
    for _ in range(reps):
        x = rng.normal(0, math.sqrt(0.3), g)[member] + rng.normal(0, math.sqrt(0.7), g * m)
        ci = stats.mean_ci(x, labels)
        hits["mean"] += ci.low <= 0 <= ci.high

        p = rng.beta(beta_a, beta_b, g)[member]
        y = (rng.uniform(size=g * m) < p).astype(float)
        ci = stats.wilson_interval_clustered(y, labels)
        hits["wilson"] += ci.low <= 0.8 <= ci.high

        # Paired pass/fail with a cluster-level effect of the change that averages to 0.
        logit = rng.normal(0, 1, g)[member] + rng.normal(0, 0.5, g * m)
        shift = rng.normal(0, 1, g)[member]
        base = rng.uniform(size=g * m) < 1 / (1 + np.exp(-logit))
        cand = rng.uniform(size=g * m) < 1 / (1 + np.exp(-(logit + shift)))
        pc = stats.paired_comparison(base, cand, labels)
        hits["paired"] += pc.low <= 0 <= pc.high
    coverage = {k: v / reps for k, v in hits.items()}
    assert all(c >= 0.93 for c in coverage.values()), coverage


def test_clustering_never_narrows_the_paired_interval() -> None:
    # Within each cluster one item improves and one regresses, so the clustered SE is
    # smaller than the item-level SE. The design effect is floored at 1, as for Wilson.
    base = np.array([1, 0, 1, 1] * 8, dtype=float)
    cand = np.array([0, 1, 1, 1] * 8, dtype=float)
    cand[:2] = 1
    labels = _labels(8, 4)
    assert stats.clustered_se(cand - base, labels) < stats.clustered_se(cand - base)
    grouped = stats.paired_comparison(base, cand, labels)
    items = stats.paired_comparison(base, cand, n_boot=4000)
    assert grouped.high - grouped.low >= items.high - items.low
