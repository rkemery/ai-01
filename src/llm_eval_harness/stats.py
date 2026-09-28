"""Statistics for small LLM evals, using numpy and scipy only.

Everything here works on plain arrays. `analysis.py` adapts these functions to
`EvalRecord` lists.

Every function that resamples takes a `seed` and builds its own
`numpy.random.Generator` from it, so the same inputs always give the same
interval. Nothing touches global random state.
"""

from __future__ import annotations

import math
import operator
from collections.abc import Hashable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy import stats as sps

_BOOT_CHUNK = 1_000  # replicates drawn per batch, bounds memory for large n


@dataclass(frozen=True)
class Interval:
    """A point estimate with a two-sided confidence interval."""

    estimate: float
    low: float
    high: float
    n: int
    confidence: float = 0.95
    method: str = ""


@dataclass(frozen=True)
class McNemarResult:
    """Exact McNemar test on paired binary outcomes.

    `a_only` counts pairs where only the first run passed, `b_only` pairs where
    only the second run passed.
    """

    a_only: int
    b_only: int
    pvalue: float

    @property
    def discordant(self) -> int:
        return self.a_only + self.b_only


@dataclass(frozen=True)
class PairedComparison:
    """Candidate minus baseline on the same items.

    `low` and `high` are the CI from `method`. `pvalue` is the two-sided
    p-value of `test` for "no difference", or None when only a bootstrap CI
    was computed. For the clustered t-test the CI and the p-value are duals:
    the CI excludes 0 exactly when p < 1 - confidence. Without clusters, a
    pass/fail metric gets the exact McNemar p-value next to a bootstrap CI, and
    the two can disagree when only a few pairs are discordant.
    """

    n: int
    baseline_mean: float
    candidate_mean: float
    diff: float
    low: float
    high: float
    confidence: float
    method: str
    n_clusters: int
    test: str | None = None
    pvalue: float | None = None
    df: int | None = None
    mcnemar: McNemarResult | None = None
    n_boot: int | None = None
    seed: int | None = None


@dataclass(frozen=True)
class PassKResult:
    """pass^k and pass@k averaged over tasks, with bootstrap CIs over tasks."""

    k: int
    n_tasks: int
    pass_hat_k: Interval
    pass_at_k: Interval


def z_value(confidence: float) -> float:
    """Two-sided standard normal quantile, z_{1 - alpha/2} with alpha = 1 - confidence."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    return float(sps.norm.ppf(1.0 - (1.0 - confidence) / 2.0))


def t_value(confidence: float, df: int) -> float:
    """Two-sided Student t quantile, t_{df, 1 - alpha/2} with alpha = 1 - confidence."""
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if df < 1:
        raise ValueError(f"df must be >= 1, got {df}")
    return float(sps.t.ppf(1.0 - (1.0 - confidence) / 2.0, df))


def wilson_interval(successes: int, n: int, confidence: float = 0.95) -> Interval:
    """Wilson score interval for a binomial proportion.

    With p = k / n and z = z_{1 - alpha/2}:

        center = (p + z^2 / (2n)) / (1 + z^2 / n)
        half   = z / (1 + z^2 / n) * sqrt(p (1 - p) / n + z^2 / (4 n^2))

    The interval stays inside [0, 1] and keeps close to nominal coverage at small
    n and at p near 0 or 1, where the normal (Wald) interval collapses to zero
    width. Bowyer et al. (2025), "Don't Use the CLT in LLM Evals With Fewer Than
    a Few Hundred Datapoints", arXiv 2503.01747, make this case for LLM evals.
    Original: Wilson (1927), JASA 22(158).
    """
    k = operator.index(successes)
    n = operator.index(n)
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0 <= k <= n:
        raise ValueError(f"successes must be in [0, n], got {k} of {n}")
    p = k / n
    low, high = _wilson_bounds(p, n, z_value(confidence))
    return Interval(p, low, high, n, confidence, "Wilson")


def design_effect(values: ArrayLike, clusters: Sequence[Hashable | None]) -> float:
    """Kish design effect of the mean, floored at 1.

        deff = max(1, SE_clustered^2 / SE_iid^2)

    both from `clustered_se`. Returns 1 when the values have no variance (for
    example no failures at all), since the data then says nothing about
    clustering. The floor means clustering can widen an interval but never
    narrow it: with 8 clusters a clustered SE below the item-level SE is far
    more likely to be noise than real negative correlation.
    """
    x = _as_1d(values)
    se_iid = clustered_se(x)
    if se_iid == 0:
        return 1.0
    return max(1.0, (clustered_se(x, clusters) / se_iid) ** 2)


def wilson_interval_clustered(
    values: ArrayLike, clusters: Sequence[Hashable | None], confidence: float = 0.95
) -> Interval:
    """Wilson interval for a pass rate whose items come in G clusters.

    Plugs the Korn-Graubard effective sample size into the Wilson formula:

        n_eff = n / deff * (z_{1 - alpha/2} / t_{G - 1, 1 - alpha/2})^2

    with deff from `design_effect` (CR1 clustered SE, floored at 1). The second
    factor is Korn and Graubard's degrees-of-freedom adjustment (Korn and
    Graubard 1998, Survey Methodology 24(2), who pair n_eff with a
    Clopper-Pearson interval). In simulation at 8 clusters of 5 items, n / deff
    alone covered 90 to 92%, and this version 95 to 97% (`tests/test_stats.py`
    checks at least 93%). Unlike a clustered normal interval this stays
    sensible at p = 0 or 1.
    """
    x = _as_1d(values)
    if not _is_binary(x):
        raise ValueError("wilson_interval_clustered needs binary values")
    g = n_clusters(clusters, x.size)
    if g < 2:
        raise ValueError("a clustered interval needs at least 2 clusters")
    z = z_value(confidence)
    n_eff = x.size / design_effect(x, clusters) * (z / t_value(confidence, g - 1)) ** 2
    p = float(x.mean())
    low, high = _wilson_bounds(p, n_eff, z)
    method = f"Wilson with Korn-Graubard effective n ({g} clusters)"
    return Interval(p, low, high, x.size, confidence, method)


def cluster_codes(clusters: Sequence[Hashable | None]) -> NDArray[np.intp]:
    """Map cluster labels to integer codes 0..G-1. A `None` label is its own cluster."""
    codes: dict[Hashable, int] = {}
    out = np.empty(len(clusters), dtype=np.intp)
    for i, label in enumerate(clusters):
        key: Hashable = ("__singleton__", i) if label is None else label
        out[i] = codes.setdefault(key, len(codes))
    return out


def n_clusters(clusters: Sequence[Hashable | None], n: int | None = None) -> int:
    """Number of distinct clusters G. Pass `n` to check there is one label per value."""
    codes = cluster_codes(clusters) if n is None else _checked_codes(clusters, n)
    return int(codes.max()) + 1 if codes.size else 0


def clustered_se(values: ArrayLike, clusters: Sequence[Hashable | None] | None = None) -> float:
    """Standard error of the mean, optionally cluster-robust.

    Without clusters this is s / sqrt(n) with the sample variance s^2 (ddof = 1).
    With G clusters it is the CR1 cluster-robust SE of a mean. With
    d_i = x_i - xbar:

        SE^2 = G / (G - 1) * sum_c (sum_{i in c} d_i)^2 / n^2

    This is the Liang-Zeger sandwich with the usual G / (G - 1) small-sample
    factor, which is what Stata and statsmodels report as CR1. Miller (2024),
    "Adding Error Bars to Evals", arXiv 2411.00640, writes the same quantity as
    the ordinary SE plus the covariance of items that share a cluster. The two
    forms differ only in small-sample factors, and the G / (G - 1) one matters
    with the handful of clusters typical of evals. When every item is its own
    cluster (G = n) the formula reduces exactly to s / sqrt(n).
    """
    x = _as_1d(values)
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 values for a standard error")
    d = x - x.mean()
    if clusters is None:
        return math.sqrt(float(np.dot(d, d)) / (n - 1) / n)
    codes = _checked_codes(clusters, n)
    g = int(codes.max()) + 1
    if g < 2:
        raise ValueError("clustered SE needs at least 2 clusters")
    cluster_sums = np.bincount(codes, weights=d)
    return math.sqrt(g / (g - 1) * float(np.dot(cluster_sums, cluster_sums))) / n


def clustered_se_floored(values: ArrayLike, clusters: Sequence[Hashable | None]) -> float:
    """max(item-level SE, CR1 clustered SE): the SE after flooring the design effect at 1."""
    x = _as_1d(values)
    return max(clustered_se(x), clustered_se(x, clusters))


def mean_ci(
    values: ArrayLike,
    clusters: Sequence[Hashable | None] | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> Interval:
    """Mean with a Student t interval, xbar +/- t * SE.

    Without clusters: SE = s / sqrt(n) and n - 1 degrees of freedom. With G
    clusters: SE = `clustered_se_floored` and G - 1 degrees of freedom. Both
    corrections matter with few clusters. In simulation at 8 clusters, a z
    quantile without the G / (G - 1) factor covered about 88%, and this version
    about 96% (`tests/test_stats.py` checks at least 93%).

    Pass `bounds` to clip the interval to a valid range. For pass rates use
    `wilson_interval` or `wilson_interval_clustered`, which behave much better
    at small n and near 0 or 1.
    """
    x = _as_1d(values)
    mean = float(x.mean())
    if clusters is None:
        se, df = clustered_se(x), x.size - 1
        method = "t, SE = s/sqrt(n)"
    else:
        g = n_clusters(clusters, x.size)
        se, df = clustered_se_floored(x, clusters), g - 1
        method = f"t with CR1 clustered SE ({g} clusters, {df} df)"
    half = t_value(confidence, df) * se
    low, high = mean - half, mean + half
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    return Interval(mean, low, high, x.size, confidence, method)


def bootstrap_means(
    values: ArrayLike,
    clusters: Sequence[Hashable | None] | None = None,
    n_boot: int = 10_000,
    seed: int = 0,
) -> NDArray[np.float64]:
    """Bootstrap replicates of the mean, resampling items or whole clusters.

    With clusters, each replicate draws G clusters with replacement and takes the
    mean over every item in the drawn clusters (sum of values / number of items).
    Without clusters, every item is its own cluster, which is the usual
    nonparametric bootstrap (Efron and Tibshirani 1993).
    """
    x = _as_1d(values)
    if x.size == 0:
        raise ValueError("cannot bootstrap an empty sample")
    if n_boot < 1:
        raise ValueError(f"n_boot must be >= 1, got {n_boot}")
    codes = np.arange(x.size) if clusters is None else _checked_codes(clusters, x.size)
    sums = np.bincount(codes, weights=x)
    sizes = np.bincount(codes).astype(np.float64)
    g = sums.size
    rng = np.random.default_rng(seed)
    out = np.empty(n_boot, dtype=np.float64)
    for start in range(0, n_boot, _BOOT_CHUNK):
        stop = min(start + _BOOT_CHUNK, n_boot)
        idx = rng.integers(0, g, size=(stop - start, g))
        out[start:stop] = sums[idx].sum(axis=1) / sizes[idx].sum(axis=1)
    return out


def percentile_interval(
    replicates: ArrayLike, estimate: float, n: int, confidence: float = 0.95, method: str = ""
) -> Interval:
    """Percentile bootstrap interval. NaN replicates are ignored."""
    reps = _as_1d(replicates, allow_nan=True)
    alpha = 1.0 - confidence
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if np.isnan(reps).all():
        raise ValueError("every bootstrap replicate is NaN")
    low, high = np.nanquantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return Interval(estimate, float(low), float(high), n, confidence, method)


def mcnemar_exact(baseline: ArrayLike, candidate: ArrayLike) -> McNemarResult:
    """Exact (binomial) McNemar test for paired binary outcomes.

    Let b = #(baseline pass, candidate fail) and c = #(baseline fail, candidate
    pass). Under H0 (same pass rate) each discordant pair falls either way with
    probability 1/2, so b ~ Binomial(b + c, 1/2). The two-sided p-value comes
    from `scipy.stats.binomtest(b, b + c, 0.5)`. With no discordant pairs p = 1.
    McNemar (1947), Psychometrika 12(2). The exact form is the right one at the
    small discordant counts typical of evals.
    """
    a = _as_bool(baseline, "baseline")
    b = _as_bool(candidate, "candidate")
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    a_only = int(np.sum(a & ~b))
    b_only = int(np.sum(~a & b))
    if a_only + b_only == 0:
        return McNemarResult(a_only, b_only, 1.0)
    pvalue = float(sps.binomtest(a_only, a_only + b_only, 0.5).pvalue)
    return McNemarResult(a_only, b_only, pvalue)


def paired_bootstrap(
    baseline: ArrayLike,
    candidate: ArrayLike,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Mean difference (candidate - baseline) on paired, independent items with a bootstrap CI.

    d_i = candidate_i - baseline_i, diff = mean(d). The CI is the percentile
    interval of `bootstrap_means(d)`. Pairing removes the between-item
    variance that an unpaired comparison would carry (Miller 2024, arXiv
    2411.00640). When both inputs are binary the exact McNemar test supplies
    the p-value. For clustered items use `paired_clustered`: a percentile
    cluster bootstrap covered only about 88% at 8 clusters.
    """
    a, b = _paired_arrays(baseline, candidate)
    d = b - a
    reps = bootstrap_means(d, n_boot=n_boot, seed=seed)
    ci = percentile_interval(reps, float(d.mean()), d.size, confidence)
    mcnemar = mcnemar_exact(a, b) if _is_binary(a) and _is_binary(b) else None
    return PairedComparison(
        n=d.size,
        baseline_mean=float(a.mean()),
        candidate_mean=float(b.mean()),
        diff=ci.estimate,
        low=ci.low,
        high=ci.high,
        confidence=confidence,
        method="percentile bootstrap over items",
        n_clusters=d.size,
        test=None if mcnemar is None else "exact McNemar test",
        pvalue=None if mcnemar is None else mcnemar.pvalue,
        mcnemar=mcnemar,
        n_boot=n_boot,
        seed=seed,
    )


def paired_clustered(
    baseline: ArrayLike,
    candidate: ArrayLike,
    clusters: Sequence[Hashable | None],
    *,
    confidence: float = 0.95,
) -> PairedComparison:
    """Mean difference on paired items that come in G clusters, with a clustered t-test.

    With d_i = candidate_i - baseline_i and SE = `clustered_se_floored(d)`:

        CI = mean(d) +/- t_{G - 1, 1 - alpha/2} * SE
        p  = 2 * P(T_{G - 1} > |mean(d)| / SE)

    so the CI excludes 0 exactly when p < alpha. Flooring the design effect at
    1 means clustering never makes the interval narrower than the item-level
    SE would. This is the same interval `mean_ci` gives for the differences.
    For binary metrics it replaces McNemar, which assumes independent pairs.
    Durkalski et al. (2003), Statistics in Medicine 22(15), use the same
    cluster-sum idea for clustered matched pairs, with a chi-square reference.
    """
    a, b = _paired_arrays(baseline, candidate)
    d = b - a
    g = n_clusters(clusters, d.size)
    if g < 2:
        raise ValueError("a clustered comparison needs at least 2 clusters")
    df = g - 1
    diff = float(d.mean())
    se = clustered_se_floored(d, clusters)
    half = t_value(confidence, df) * se
    # With SE = 0 every difference is identical: p is 1 if they are all 0, else 0.
    pvalue = float(2.0 * sps.t.sf(abs(diff) / se, df)) if se > 0 else float(diff == 0)
    low, high = diff - half, diff + half
    if _is_binary(a) and _is_binary(b):
        low, high = max(-1.0, low), min(1.0, high)
    return PairedComparison(
        n=d.size,
        baseline_mean=float(a.mean()),
        candidate_mean=float(b.mean()),
        diff=diff,
        low=low,
        high=high,
        confidence=confidence,
        method=f"t with CR1 clustered SE ({g} clusters, {df} df)",
        n_clusters=g,
        test="clustered t-test",
        pvalue=pvalue,
        df=df,
    )


def paired_comparison(
    baseline: ArrayLike,
    candidate: ArrayLike,
    clusters: Sequence[Hashable | None] | None = None,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """`paired_bootstrap` without clusters, `paired_clustered` with them."""
    if clusters is None:
        return paired_bootstrap(
            baseline, candidate, n_boot=n_boot, confidence=confidence, seed=seed
        )
    return paired_clustered(baseline, candidate, clusters, confidence=confidence)


def mde_from_se(se: float, alpha: float = 0.05, power: float = 0.8, df: int | None = None) -> float:
    """Minimum detectable effect of a two-sided z-test (or t-test with `df`) with standard error se.

    MDE = (z_{1 - alpha/2} + z_{power}) * se. At alpha 0.05 and 80% power the
    multiplier is 1.960 + 0.842 = 2.802. With `df`, t quantiles replace the z
    quantiles, the usual approximation for a t-test (at 7 df: 2.365 + 0.896).
    """
    if se < 0:
        raise ValueError(f"se must be >= 0, got {se}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    if df is None:
        return (z_value(1.0 - alpha) + float(sps.norm.ppf(power))) * se
    return (t_value(1.0 - alpha, df) + float(sps.t.ppf(power, df))) * se


def mcnemar_exact_power(n: int, discordant_rate: float, diff: float, alpha: float = 0.05) -> float:
    """Power of the two-sided exact McNemar test (`mcnemar_exact`, reject when p < alpha).

    Pairs are independent. Each is discordant with probability p_d, and a
    discordant pair favors the candidate with probability
    q = (p_d + diff) / (2 p_d), where diff is the difference in pass rates. So
    the number of discordant pairs is D ~ Binomial(n, p_d) and, given D = d,
    the candidate-only count is Binomial(d, q). The test rejects when the
    smaller of the two discordant counts is at most k_d, the largest k with
    2 * P(Binomial(d, 1/2) <= k) < alpha. Therefore

        power = sum_d P(D = d) * P(reject | d, q)

    computed exactly, with no normal approximation.
    """
    _check_paired_power_args(n, discordant_rate, alpha)
    if abs(diff) > discordant_rate + 1e-12:
        raise ValueError(
            f"|diff| = {abs(diff)} cannot exceed the discordant rate {discordant_rate}"
        )
    return _mcnemar_power(n, discordant_rate, diff, _mcnemar_critical(n, alpha))


def mde_paired_binary(
    n: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.8
) -> float | None:
    """Smallest difference in pass rates the exact McNemar test detects with `power`.

    Holds the discordant rate p_d at its observed value and searches (by
    bisection, since power grows with |diff|) for the smallest diff in
    (0, p_d] with `mcnemar_exact_power` >= power. The pass rates of two runs
    can differ by at most p_d, so the MDE never exceeds it. Returns None when
    even diff = p_d (every discordant pair going one way) has less power than
    asked, which happens when n * p_d is small: the test needs at least 6
    discordant pairs to reach p < 0.05 at all.
    """
    _check_paired_power_args(n, discordant_rate, alpha)
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    critical = _mcnemar_critical(n, alpha)
    if _mcnemar_power(n, discordant_rate, discordant_rate, critical) < power:
        return None
    low, high = 0.0, discordant_rate
    while high - low > 1e-7:
        mid = (low + high) / 2.0
        if _mcnemar_power(n, discordant_rate, mid, critical) >= power:
            high = mid
        else:
            low = mid
    return high


def mde_two_proportions(
    n_a: int, n_b: int, p: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """MDE for an unpaired comparison of two pass rates.

        MDE = (z_{1 - alpha/2} + z_{power}) * sqrt(p (1 - p) (1/n_a + 1/n_b))

    This is the normal approximation with both arms given the variance of the
    baseline rate p (see Fleiss, Levin and Paik, Statistical Methods for Rates
    and Proportions, 3rd ed., ch. 4). It understates the MDE a little when the
    other arm sits closer to 0.5 than p does.
    """
    if n_a <= 0 or n_b <= 0:
        raise ValueError(f"sample sizes must be positive, got {n_a} and {n_b}")
    if not 0.0 <= p <= 1.0:
        raise ValueError(f"p must be in [0, 1], got {p}")
    return mde_from_se(math.sqrt(p * (1.0 - p) * (1.0 / n_a + 1.0 / n_b)), alpha, power)


def pass_hat_k(n: int, c: int, k: int) -> float:
    """pass^k for one task: chance that k draws without replacement from n trials all pass.

    pass^k = C(c, k) / C(n, k), the unbiased estimator from tau-bench (Yao et al.
    2024, arXiv 2406.12045). It measures consistency: pass^1 is the plain pass
    rate and pass^k falls as k grows unless the agent always succeeds.
    """
    _check_nck(n, c, k)
    return math.comb(c, k) / math.comb(n, k)


def pass_at_k(n: int, c: int, k: int) -> float:
    """pass@k for one task: chance that at least one of k draws from n trials passes.

    pass@k = 1 - C(n - c, k) / C(n, k) (Chen et al. 2021, arXiv 2107.03374).
    """
    _check_nck(n, c, k)
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def pass_k(
    counts: Sequence[tuple[int, int]],
    k: int,
    confidence: float = 0.95,
    n_boot: int = 10_000,
    seed: int = 0,
) -> PassKResult:
    """Average pass^k and pass@k over tasks given (n_trials, n_successes) per task.

    CIs are percentile bootstrap intervals over tasks, which treats the task set
    as the sample and the per-task estimates as fixed.
    """
    if not counts:
        raise ValueError("need at least one task")
    hat = np.array([pass_hat_k(n, c, k) for n, c in counts])
    at = np.array([pass_at_k(n, c, k) for n, c in counts])
    method = "bootstrap over tasks"
    hat_ci = percentile_interval(
        bootstrap_means(hat, n_boot=n_boot, seed=seed),
        float(hat.mean()),
        hat.size,
        confidence,
        method,
    )
    at_ci = percentile_interval(
        bootstrap_means(at, n_boot=n_boot, seed=seed),
        float(at.mean()),
        at.size,
        confidence,
        method,
    )
    return PassKResult(k=k, n_tasks=len(counts), pass_hat_k=hat_ci, pass_at_k=at_ci)


def _wilson_bounds(p: float, n: float, z: float) -> tuple[float, float]:
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return max(0.0, center - half), min(1.0, center + half)


def _check_paired_power_args(n: int, discordant_rate: float, alpha: float) -> None:
    if operator.index(n) <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError(f"discordant_rate must be in (0, 1], got {discordant_rate}")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")


def _mcnemar_critical(n: int, alpha: float) -> NDArray[np.int64]:
    """For d = 0..n discordant pairs, the largest k with 2 * P(Bin(d, 1/2) <= k) < alpha.

    -1 where no count is significant. For p = 1/2 the two-sided binomial
    p-value of `scipy.stats.binomtest` is min(1, 2 * P(X <= min(b, c))), so
    `mcnemar_exact` gives p < alpha exactly when min(b, c) <= k_d.
    """
    d = np.arange(n + 1)
    half = alpha / 2.0
    k = sps.binom.ppf(half, d, 0.5).astype(np.int64) - 1
    # ppf is the smallest k with cdf >= half. Step once either way to absorb rounding.
    k = np.where(sps.binom.cdf(k, d, 0.5) >= half, k - 1, k)
    k = np.where(sps.binom.cdf(k + 1, d, 0.5) < half, k + 1, k)
    return np.maximum(k, -1)


def _mcnemar_power(
    n: int, discordant_rate: float, diff: float, critical: NDArray[np.int64]
) -> float:
    d = np.arange(n + 1)
    weights = sps.binom.pmf(d, n, discordant_rate)
    q = min(1.0, max(0.0, (discordant_rate + diff) / (2.0 * discordant_rate)))
    some = critical >= 0
    k = np.where(some, critical, 0)
    reject = sps.binom.cdf(k, d, q) + sps.binom.sf(d - k - 1, d, q)
    return float(np.dot(weights, np.where(some, reject, 0.0)))


def _paired_arrays(
    baseline: ArrayLike, candidate: ArrayLike
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    a = _as_1d(baseline)
    b = _as_1d(candidate)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    return a, b


def _check_nck(n: int, c: int, k: int) -> None:
    for name, value in (("n", n), ("c", c), ("k", k)):
        operator.index(value)
        if value < 0:
            raise ValueError(f"{name} must be >= 0, got {value}")
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    if c > n:
        raise ValueError(f"successes c={c} exceed trials n={n}")
    if k > n:
        raise ValueError(f"k={k} exceeds the number of trials n={n}")


def _as_1d(values: ArrayLike, allow_nan: bool = False) -> NDArray[np.float64]:
    x = np.asarray(values, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"expected a 1-D array, got shape {x.shape}")
    bad = np.isinf(x) if allow_nan else ~np.isfinite(x)
    if bad.any():
        raise ValueError("values must be finite")
    return x


def _as_bool(values: ArrayLike, name: str) -> NDArray[np.bool_]:
    x = np.asarray(values)
    if x.ndim != 1:
        raise ValueError(f"{name}: expected a 1-D array, got shape {x.shape}")
    if x.dtype != np.bool_:
        if not _is_binary(x.astype(np.float64)):
            raise ValueError(f"{name}: expected binary values (bool or 0/1)")
        x = x.astype(np.bool_)
    return x


def _is_binary(x: NDArray[np.float64]) -> bool:
    return bool(np.isin(x, (0.0, 1.0)).all())


def _checked_codes(clusters: Sequence[Hashable | None], n: int) -> NDArray[np.intp]:
    if len(clusters) != n:
        raise ValueError(f"got {len(clusters)} cluster labels for {n} values")
    return cluster_codes(clusters)
