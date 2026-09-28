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
    """Candidate minus baseline on the same items."""

    n: int
    baseline_mean: float
    candidate_mean: float
    diff: float
    low: float
    high: float
    confidence: float
    n_boot: int
    seed: int
    n_clusters: int
    mcnemar: McNemarResult | None = None


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
    example no failures at all), since the data then says nothing about clustering.
    """
    x = _as_1d(values)
    se_iid = clustered_se(x)
    if se_iid == 0:
        return 1.0
    return max(1.0, (clustered_se(x, clusters) / se_iid) ** 2)


def wilson_interval_clustered(
    values: ArrayLike, clusters: Sequence[Hashable | None], confidence: float = 0.95
) -> Interval:
    """Wilson interval for a pass rate whose items come in clusters.

    Plugs the effective sample size n_eff = n / deff (`design_effect`) into the
    Wilson formula. The effective-sample-size idea is from Korn and Graubard
    (1998), Survey Methodology 24(2), who pair it with a Clopper-Pearson
    interval. Unlike a clustered normal interval this stays sensible at p = 0
    or 1, where the normal interval collapses to a single point.
    """
    x = _as_1d(values)
    if not _is_binary(x):
        raise ValueError("wilson_interval_clustered needs binary values")
    n_clusters = int(_checked_codes(clusters, x.size).max()) + 1
    n_eff = x.size / design_effect(x, clusters)
    p = float(x.mean())
    low, high = _wilson_bounds(p, n_eff, z_value(confidence))
    method = f"Wilson with design-effect n ({n_clusters} clusters)"
    return Interval(p, low, high, x.size, confidence, method)


def cluster_codes(clusters: Sequence[Hashable | None]) -> NDArray[np.intp]:
    """Map cluster labels to integer codes 0..G-1. A `None` label is its own cluster."""
    codes: dict[Hashable, int] = {}
    out = np.empty(len(clusters), dtype=np.intp)
    for i, label in enumerate(clusters):
        key: Hashable = ("__singleton__", i) if label is None else label
        out[i] = codes.setdefault(key, len(codes))
    return out


def clustered_se(values: ArrayLike, clusters: Sequence[Hashable | None] | None = None) -> float:
    """Standard error of the mean, optionally clustered.

    Miller (2024), "Adding Error Bars to Evals", arXiv 2411.00640, writes the
    clustered SE as the ordinary (CLT) standard error plus the covariance of
    items that share a cluster. With d_i = x_i - xbar, n items and s^2 the
    sample variance (ddof = 1):

        SE^2 = s^2 / n + (1 / n^2) * sum_c sum_{i != j in c} d_i d_j
             = s^2 / n + (sum_c (sum_{i in c} d_i)^2 - sum_i d_i^2) / n^2

    Compared with the textbook cluster-robust form, sqrt(sum_c (sum_{i in c}
    d_i)^2) / n, the only change is the unbiased s^2 in the first term, which
    is n / (n - 1) times larger than sum_i d_i^2 / n^2. That is what makes the
    fallback exact: when every item is its own cluster the cross terms vanish
    and this returns the ordinary SE, s / sqrt(n). With `clusters=None` that is
    what you get directly.
    """
    x = _as_1d(values)
    n = x.size
    if n < 2:
        raise ValueError("need at least 2 values for a standard error")
    d = x - x.mean()
    sum_sq = float(np.dot(d, d))
    s2 = sum_sq / (n - 1)
    if clusters is None:
        return math.sqrt(s2 / n)
    codes = _checked_codes(clusters, n)
    if codes.max() + 1 < 2:
        raise ValueError("clustered SE needs at least 2 clusters")
    cluster_sums = np.bincount(codes, weights=d)
    cross = float(np.dot(cluster_sums, cluster_sums)) - sum_sq
    # SE^2 >= sum_c(...)^2 / n^2 >= 0 algebraically. max() only absorbs rounding.
    return math.sqrt(max(s2 / n + cross / (n * n), 0.0))


def mean_ci(
    values: ArrayLike,
    clusters: Sequence[Hashable | None] | None = None,
    confidence: float = 0.95,
    bounds: tuple[float, float] | None = None,
) -> Interval:
    """Mean with a normal-approximation CI, xbar +/- z * SE, using `clustered_se`.

    Pass `bounds` to clip the interval to a valid range. For pass rates use
    `wilson_interval` or `wilson_interval_clustered`, which behave much better
    at small n and near 0 or 1.
    """
    x = _as_1d(values)
    se = clustered_se(x, clusters)
    z = z_value(confidence)
    mean = float(x.mean())
    low, high = mean - z * se, mean + z * se
    if bounds is not None:
        low, high = max(bounds[0], low), min(bounds[1], high)
    if clusters is None:
        method = "Normal, SE = s/sqrt(n)"
    else:
        method = f"Normal, clustered SE ({int(cluster_codes(clusters).max()) + 1} clusters)"
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
    clusters: Sequence[Hashable | None] | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> PairedComparison:
    """Mean difference (candidate - baseline) on paired items with a bootstrap CI.

    d_i = candidate_i - baseline_i, diff = mean(d). The CI is the percentile
    interval of `bootstrap_means(d)`, resampling clusters when given so that
    correlated items (for example questions about the same source article) move
    together. Pairing removes the between-item variance that an unpaired
    comparison would carry (Miller 2024, arXiv 2411.00640). When both
    inputs are binary the exact McNemar test is attached as well.
    """
    a = _as_1d(baseline)
    b = _as_1d(candidate)
    if a.size != b.size:
        raise ValueError(f"length mismatch: {a.size} vs {b.size}")
    d = b - a
    reps = bootstrap_means(d, clusters, n_boot=n_boot, seed=seed)
    ci = percentile_interval(reps, float(d.mean()), d.size, confidence)
    binary = _is_binary(a) and _is_binary(b)
    n_clusters = d.size if clusters is None else int(cluster_codes(clusters).max()) + 1
    return PairedComparison(
        n=d.size,
        baseline_mean=float(a.mean()),
        candidate_mean=float(b.mean()),
        diff=ci.estimate,
        low=ci.low,
        high=ci.high,
        confidence=confidence,
        n_boot=n_boot,
        seed=seed,
        n_clusters=n_clusters,
        mcnemar=mcnemar_exact(a, b) if binary else None,
    )


def mde_from_se(se: float, alpha: float = 0.05, power: float = 0.8) -> float:
    """Minimum detectable effect for a two-sided z-test with standard error `se`.

    MDE = (z_{1 - alpha/2} + z_{power}) * se. At alpha 0.05 and 80% power the
    multiplier is 1.960 + 0.842 = 2.802.
    """
    if se < 0:
        raise ValueError(f"se must be >= 0, got {se}")
    if not 0.0 < power < 1.0:
        raise ValueError(f"power must be in (0, 1), got {power}")
    return (z_value(1.0 - alpha) + float(sps.norm.ppf(power))) * se


def mde_paired_binary(
    n: int, discordant_rate: float, alpha: float = 0.05, power: float = 0.8
) -> float:
    """MDE (difference in pass rates) for a paired binary comparison of n items.

    Each pair contributes d_i in {-1, 0, 1}. With discordant rate p_d and true
    difference delta, Var(d_i) = p_d - delta^2. We use the variance under H0,
    p_d, for both terms:

        MDE = (z_{1 - alpha/2} + z_{power}) * sqrt(p_d / n)

    Dropping delta^2 makes this slightly larger than the exact solution of
    Connor (1987), Biometrics 43(1), so it errs on the conservative side.
    """
    if n <= 0:
        raise ValueError(f"n must be positive, got {n}")
    if not 0.0 < discordant_rate <= 1.0:
        raise ValueError(f"discordant_rate must be in (0, 1], got {discordant_rate}")
    return mde_from_se(math.sqrt(discordant_rate / n), alpha, power)


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
