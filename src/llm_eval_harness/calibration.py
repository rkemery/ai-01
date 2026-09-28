"""Judge vs human agreement, and pass rates corrected for judge error.

Positive means "pass". TPR is the share of human passes the judge also passes,
TNR the share of human fails the judge also fails.

The workflow the plan requires:

1. `split_dev_test` the labeled items once, with a fixed seed, and save it.
2. Tune the judge prompt on dev labels only.
3. Freeze the judge (its fingerprint goes into every result's meta).
4. Score the test labels. `pair_labels` refuses to mix judge fingerprints, and
   refuses labels that were not sampled uniformly at random.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

from llm_eval_harness.labeling import LabelRecord
from llm_eval_harness.records import EvalRecord, OnError, RecordError, metric_column
from llm_eval_harness.stats import (
    Interval,
    bootstrap_means,
    percentile_interval,
    wilson_interval,
)


class CalibrationError(ValueError):
    """The labels or judge results cannot support the requested calibration."""


@dataclass(frozen=True)
class Confusion:
    tp: int
    fn: int
    tn: int
    fp: int

    @property
    def n(self) -> int:
        return self.tp + self.fn + self.tn + self.fp


@dataclass(frozen=True)
class JudgeAgreement:
    n: int
    confusion: Confusion
    tpr: Interval
    tnr: Interval
    kappa: Interval


@dataclass(frozen=True)
class CorrectedPassRate:
    """Judge pass rate on unlabeled data, before and after Rogan-Gladen correction.

    `invalid_replicates` counts the `n_boot` draws where TPR* + TNR* <= 1, in
    which the correction is undefined. They are left out of the interval, which
    is then conditional on the judge being informative. Report the count: if it
    is more than a few percent of `n_boot`, the judge is too weak for the
    interval to mean much.
    """

    observed: Interval
    corrected: Interval
    tpr: float
    tnr: float
    n_calibration: int
    invalid_replicates: int
    n_boot: int


@dataclass(frozen=True)
class Split:
    seed: int
    dev: tuple[str, ...]
    test: tuple[str, ...]

    def __post_init__(self) -> None:
        overlap = set(self.dev) & set(self.test)
        if overlap:
            raise CalibrationError(f"dev and test overlap on {sorted(overlap)[:5]}")


@dataclass(frozen=True)
class LabelPairs:
    """Judge and human labels for one check, aligned by item."""

    check: str
    item_ids: tuple[str, ...]
    judge: tuple[bool, ...]
    human: tuple[bool, ...]
    excluded: tuple[str, ...]


def confusion(judge: ArrayLike, human: ArrayLike) -> Confusion:
    j, h = _paired_bools(judge, human)
    return Confusion(
        tp=int(np.sum(j & h)),
        fn=int(np.sum(~j & h)),
        tn=int(np.sum(~j & ~h)),
        fp=int(np.sum(j & ~h)),
    )


def cohens_kappa(judge: ArrayLike, human: ArrayLike) -> float:
    """Cohen's kappa for two raters and binary labels.

        p_o = (TP + TN) / n
        p_e = p_j * p_h + (1 - p_j)(1 - p_h), with p_j, p_h each rater's pass rate
        kappa = (p_o - p_e) / (1 - p_e)

    Cohen (1960), Educational and Psychological Measurement 20(1). Returns NaN
    when p_e = 1 (both raters gave one identical label to everything), where
    kappa is undefined.
    """
    c = confusion(judge, human)
    return float(_kappa(c.tp, c.fn, c.tn, c.fp))


def kappa_interval(
    judge: ArrayLike,
    human: ArrayLike,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> Interval:
    """Cohen's kappa with a percentile bootstrap CI over items. Undefined replicates are skipped."""
    j, h = _paired_bools(judge, human)
    n = j.size
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    jb, hb = j[idx], h[idx]
    tp = np.sum(jb & hb, axis=1)
    fn = np.sum(~jb & hb, axis=1)
    tn = np.sum(~jb & ~hb, axis=1)
    fp = np.sum(jb & ~hb, axis=1)
    reps = _kappa(tp, fn, tn, fp)
    estimate = cohens_kappa(j, h)
    return percentile_interval(reps, estimate, n, confidence, "bootstrap over items")


def judge_agreement(
    judge: ArrayLike,
    human: ArrayLike,
    *,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> JudgeAgreement:
    """TPR and TNR with Wilson intervals, and kappa with a bootstrap interval."""
    c = confusion(judge, human)
    if c.tp + c.fn == 0 or c.tn + c.fp == 0:
        raise CalibrationError(
            "the human labels need both passes and fails to estimate TPR and TNR "
            f"(got {c.tp + c.fn} passes, {c.tn + c.fp} fails)"
        )
    return JudgeAgreement(
        n=c.n,
        confusion=c,
        tpr=wilson_interval(c.tp, c.tp + c.fn, confidence),
        tnr=wilson_interval(c.tn, c.tn + c.fp, confidence),
        kappa=kappa_interval(judge, human, n_boot=n_boot, confidence=confidence, seed=seed),
    )


def rogan_gladen(observed: float, tpr: float, tnr: float) -> float:
    """Bias-corrected pass rate from a judge's observed pass rate.

        theta = (p + TNR - 1) / (TPR + TNR - 1), clipped to [0, 1]

    Rogan and Gladen (1978), American Journal of Epidemiology 107(1). Lee et
    al. (2025, arXiv 2511.21140) apply it to LLM-as-a-judge results. Raises
    `CalibrationError` when TPR + TNR <= 1, where the judge carries no
    information about the true rate and the correction is undefined.
    """
    for name, value in (("observed", observed), ("tpr", tpr), ("tnr", tnr)):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be in [0, 1], got {value}")
    youden = tpr + tnr - 1.0
    if youden <= 0:
        raise CalibrationError(f"TPR + TNR = {tpr + tnr:.3f} <= 1, so the correction is undefined")
    return min(1.0, max(0.0, (observed + tnr - 1.0) / youden))


def corrected_pass_rate(
    test_judge: ArrayLike,
    calibration_judge: ArrayLike,
    calibration_human: ArrayLike,
    *,
    test_clusters: Sequence[str | None] | None = None,
    n_boot: int = 10_000,
    confidence: float = 0.95,
    seed: int = 0,
) -> CorrectedPassRate:
    """Rogan-Gladen pass rate with a CI that includes calibration uncertainty.

    Lee et al. (2025, arXiv 2511.21140) point out that the error in the judge's
    TPR and TNR, estimated from a finite labeled set, has to show up in the
    interval next to the sampling error of the test set. Each replicate
    therefore redraws both:

    - p*: the judged test set, resampled by item or by cluster
    - TPR* ~ Beta(TP + 1/2, FN + 1/2) and TNR* ~ Beta(TN + 1/2, FP + 1/2),
      the Jeffreys posteriors given the labeled confusion counts

    and computes theta* = rogan_gladen(p*, TPR*, TNR*). The CI is the percentile
    interval of theta*. Replicates with TPR* + TNR* <= 1 are counted and dropped.

    Posterior draws instead of resampling the labeled pairs matter when the
    observed TPR or TNR is 0 or 1. Resampling then gives TPR* = 1 in every
    replicate, as if the judge were known to be perfect. With 10 human passes
    all passed by a judge whose true TPR was 0.9, that interval covered 73 to
    77% in 1000- to 2000-run simulations, and the Beta version about 97%
    (`tests/test_calibration.py` checks at least 93%).
    """
    test = _as_bools(test_judge, "test_judge")
    agreement = confusion(calibration_judge, calibration_human)
    n_pos = agreement.tp + agreement.fn
    n_neg = agreement.tn + agreement.fp
    if n_pos == 0 or n_neg == 0:
        raise CalibrationError("calibration labels need both human passes and human fails")
    tpr = agreement.tp / n_pos
    tnr = agreement.tn / n_neg
    observed = float(test.mean())
    estimate = rogan_gladen(observed, tpr, tnr)

    test_seed, calib_seed = (int(s) for s in np.random.SeedSequence(seed).generate_state(2))
    p_star = bootstrap_means(test.astype(np.float64), test_clusters, n_boot, test_seed)
    rng = np.random.default_rng(calib_seed)
    tpr_star = rng.beta(agreement.tp + 0.5, agreement.fn + 0.5, size=n_boot)
    tnr_star = rng.beta(agreement.tn + 0.5, agreement.fp + 0.5, size=n_boot)
    youden = tpr_star + tnr_star - 1.0
    valid = youden > 0
    theta = np.full(n_boot, np.nan)
    theta[valid] = np.clip((p_star[valid] + tnr_star[valid] - 1.0) / youden[valid], 0.0, 1.0)

    corrected = percentile_interval(
        theta, estimate, test.size, confidence, "Rogan-Gladen, bootstrap over test and calibration"
    )
    return CorrectedPassRate(
        observed=wilson_interval(int(test.sum()), test.size, confidence),
        corrected=corrected,
        tpr=tpr,
        tnr=tnr,
        n_calibration=agreement.n,
        invalid_replicates=int(n_boot - valid.sum()),
        n_boot=n_boot,
    )


def check_same_judge(
    calibration_records: Sequence[EvalRecord], applied_records: Sequence[EvalRecord]
) -> str:
    """Return the judge fingerprint both sets share, or raise `CalibrationError`.

    A judge's TPR and TNR say how to correct that judge's verdicts only. Both
    sets must carry one and the same `meta["judge_fingerprint"]`.
    """
    calibrated = _fingerprints(calibration_records, "calibration")
    applied = _fingerprints(applied_records, "--apply")
    if len(calibrated) != 1 or calibrated != applied:
        raise CalibrationError(
            f"the --apply results were judged by {sorted(applied)}, but the calibration is "
            f"for {sorted(calibrated)}. A judge's TPR and TNR only correct that judge's verdicts."
        )
    return calibrated.pop()


def _fingerprints(records: Sequence[EvalRecord], name: str) -> set[str]:
    found = {r.meta.get("judge_fingerprint") for r in records}
    if not found or None in found:
        raise CalibrationError(
            f"some {name} results have no judge_fingerprint in meta, so the judge cannot be "
            "verified. Record the judge's fingerprint with every result."
        )
    return {str(f) for f in found}


def labeled_checks(labels: Sequence[LabelRecord]) -> list[str]:
    """Checks that every label record answers."""
    if not labels:
        raise CalibrationError("no labels")
    common = set(labels[0].labels)
    for label in labels[1:]:
        common &= set(label.labels)
    return sorted(common)


def split_dev_test(item_ids: Iterable[str], n_dev: int, seed: int = 0) -> Split:
    """Seeded dev/test split. Ids are sorted first, so input order does not matter."""
    ids = list(item_ids)
    unique = sorted(set(ids))
    if len(unique) != len(ids):
        raise CalibrationError("item ids must be unique")
    if not 0 < n_dev < len(unique):
        raise CalibrationError(f"n_dev must be between 1 and {len(unique) - 1}, got {n_dev}")
    perm = np.random.default_rng(seed).permutation(len(unique))
    dev = sorted(unique[i] for i in perm[:n_dev])
    test = sorted(unique[i] for i in perm[n_dev:])
    return Split(seed=seed, dev=tuple(dev), test=tuple(test))


def save_split(split: Split, path: str | Path) -> None:
    payload = {"seed": split.seed, "dev": list(split.dev), "test": list(split.test)}
    Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def load_split(path: str | Path) -> Split:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or set(data) != {"seed", "dev", "test"}:
        raise CalibrationError(f"{path}: expected keys seed, dev and test")
    return Split(seed=int(data["seed"]), dev=tuple(data["dev"]), test=tuple(data["test"]))


def pair_labels(
    judge_records: Sequence[EvalRecord],
    labels: Sequence[LabelRecord],
    check: str,
    *,
    only_items: Iterable[str] | None = None,
    on_error: OnError = "raise",
) -> LabelPairs:
    """Align one check's judge verdicts with human labels.

    Refuses labels from disagreement sampling (they over-represent hard cases
    and would bias TPR, TNR and kappa) and judge results produced by more than
    one judge fingerprint.
    """
    biased = sorted({label.sampling for label in labels} - {"uniform"})
    if biased:
        raise CalibrationError(
            f"labels sampled with {biased} are not a uniform sample and cannot be used for "
            "TPR, TNR or kappa. Use them only to look for judge bugs."
        )
    fingerprints = {r.meta.get("judge_fingerprint") for r in judge_records}
    if len(fingerprints) > 1:
        raise CalibrationError(
            f"judge results come from {len(fingerprints)} different judge versions "
            f"{sorted(map(str, fingerprints))}. Calibrate one frozen judge at a time."
        )
    column = metric_column(judge_records, check, on_error=on_error)
    if not column.binary:
        raise CalibrationError(f"judge metric {check!r} is not boolean")
    wanted = None if only_items is None else set(only_items)
    item_ids: list[str] = []
    judge: list[bool] = []
    human: list[bool] = []
    excluded: list[str] = []
    for label in sorted(labels, key=lambda r: r.item_id):
        if wanted is not None and label.item_id not in wanted:
            continue
        if check not in label.labels:
            raise CalibrationError(f"label for {label.item_id!r} has no {check!r}")
        if label.item_id in column.excluded:
            excluded.append(label.item_id)
            continue
        if label.item_id not in column.values:
            raise RecordError(f"no judge result for labeled item {label.item_id!r}")
        item_ids.append(label.item_id)
        judge.append(column.values[label.item_id] == 1.0)
        human.append(label.labels[check])
    if not item_ids:
        raise CalibrationError(f"no labeled items with a judge verdict for {check!r}")
    return LabelPairs(check, tuple(item_ids), tuple(judge), tuple(human), tuple(excluded))


def _kappa(tp: ArrayLike, fn: ArrayLike, tn: ArrayLike, fp: ArrayLike) -> np.ndarray:
    """Kappa from confusion counts, vectorized over bootstrap replicates. NaN where p_e = 1."""
    tp, fn, tn, fp = (np.asarray(v, dtype=np.float64) for v in (tp, fn, tn, fp))
    n = tp + fn + tn + fp
    p_o = (tp + tn) / n
    p_j = (tp + fp) / n
    p_h = (tp + fn) / n
    p_e = p_j * p_h + (1 - p_j) * (1 - p_h)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(p_e < 1, (p_o - p_e) / (1 - p_e), np.nan)


def _as_bools(values: ArrayLike, name: str) -> np.ndarray:
    x = np.asarray(values)
    if x.ndim != 1 or x.size == 0:
        raise ValueError(f"{name} must be a non-empty 1-D sequence")
    if x.dtype != np.bool_:
        if not np.isin(x, (0, 1)).all():
            raise ValueError(f"{name} must be binary")
        x = x.astype(np.bool_)
    return x


def _paired_bools(judge: ArrayLike, human: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    j = _as_bools(judge, "judge")
    h = _as_bools(human, "human")
    if j.size != h.size:
        raise ValueError(f"judge and human lengths differ: {j.size} vs {h.size}")
    return j, h
