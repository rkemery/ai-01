from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from helpers import record

from llm_eval_harness import calibration as cal
from llm_eval_harness.labeling import LabelRecord
from llm_eval_harness.records import EvalRecord, RecordError
from llm_eval_harness.stats import wilson_interval


def table(tp: int, fp: int, fn: int, tn: int) -> tuple[list[bool], list[bool]]:
    """Judge and human label lists for a 2x2 table (positive = pass)."""
    judge = [True] * tp + [True] * fp + [False] * fn + [False] * tn
    human = [True] * tp + [False] * fp + [True] * fn + [False] * tn
    return judge, human


def test_confusion_counts() -> None:
    judge, human = table(tp=20, fp=5, fn=10, tn=15)
    assert cal.confusion(judge, human) == cal.Confusion(tp=20, fn=10, tn=15, fp=5)


def test_kappa_known_value() -> None:
    # p_o = 0.7, judge passes 0.5, human passes 0.6, p_e = 0.5 * 0.6 + 0.5 * 0.4 = 0.5.
    judge, human = table(tp=20, fp=5, fn=10, tn=15)
    assert cal.cohens_kappa(judge, human) == pytest.approx(0.4)


def test_kappa_edge_cases() -> None:
    assert cal.cohens_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == pytest.approx(1.0)
    assert cal.cohens_kappa([1, 0, 1, 0], [0, 1, 0, 1]) == pytest.approx(-1.0)
    assert math.isnan(cal.cohens_kappa([1, 1, 1], [1, 1, 1]))  # p_e = 1, undefined


def test_kappa_interval_is_seeded_and_contains_estimate() -> None:
    judge, human = table(tp=20, fp=5, fn=10, tn=15)
    first = cal.kappa_interval(judge, human, n_boot=2000, seed=3)
    assert first == cal.kappa_interval(judge, human, n_boot=2000, seed=3)
    assert first.low < 0.4 < first.high


def test_judge_agreement_uses_wilson_for_rates() -> None:
    judge, human = table(tp=27, fp=2, fn=3, tn=8)
    agreement = cal.judge_agreement(judge, human, n_boot=500)
    assert agreement.tpr == wilson_interval(27, 30)
    assert agreement.tnr == wilson_interval(8, 10)
    assert agreement.n == 40


def test_judge_agreement_needs_both_classes() -> None:
    with pytest.raises(cal.CalibrationError, match="both passes and fails"):
        cal.judge_agreement([True, False], [True, True])


def test_rogan_gladen_values_and_clipping() -> None:
    assert cal.rogan_gladen(0.37, 1.0, 1.0) == pytest.approx(0.37)  # perfect judge
    assert cal.rogan_gladen(0.5, 0.9, 0.8) == pytest.approx(0.3 / 0.7)
    assert cal.rogan_gladen(0.1, 0.9, 0.8) == 0.0  # would be negative
    assert cal.rogan_gladen(0.95, 0.9, 0.9) == 1.0  # would be above 1


def test_rogan_gladen_undefined_for_uninformative_judge() -> None:
    with pytest.raises(cal.CalibrationError, match="undefined"):
        cal.rogan_gladen(0.5, 0.5, 0.5)
    with pytest.raises(ValueError, match="must be in"):
        cal.rogan_gladen(1.2, 0.9, 0.9)


def test_corrected_pass_rate_point_estimate_and_ci() -> None:
    test = [True] * 60 + [False] * 40
    judge, human = table(tp=36, fp=4, fn=4, tn=16)  # TPR 0.9, TNR 0.8
    result = cal.corrected_pass_rate(test, judge, human, n_boot=4000, seed=1)
    assert result.tpr == pytest.approx(0.9)
    assert result.tnr == pytest.approx(0.8)
    assert result.corrected.estimate == pytest.approx(cal.rogan_gladen(0.6, 0.9, 0.8))
    assert result.corrected.low < result.corrected.estimate < result.corrected.high
    assert result.observed == wilson_interval(60, 100)
    assert result == cal.corrected_pass_rate(test, judge, human, n_boot=4000, seed=1)


def test_corrected_ci_includes_calibration_uncertainty() -> None:
    test = [True] * 60 + [False] * 40
    small = table(tp=18, fp=2, fn=2, tn=8)
    large = table(tp=1800, fp=200, fn=200, tn=800)  # same rates, 100x the labels
    width_small = _width(cal.corrected_pass_rate(test, *small, n_boot=4000))
    width_large = _width(cal.corrected_pass_rate(test, *large, n_boot=4000))
    assert width_small > 1.3 * width_large


def test_corrected_counts_invalid_replicates_for_weak_judge() -> None:
    test = [True] * 10 + [False] * 10
    judge, human = table(tp=4, fp=2, fn=2, tn=3)  # TPR 0.67, TNR 0.6, barely informative
    result = cal.corrected_pass_rate(test, judge, human, n_boot=2000)
    assert result.invalid_replicates > 0


def _width(result: cal.CorrectedPassRate) -> float:
    return result.corrected.high - result.corrected.low


def test_split_is_seeded_disjoint_and_order_independent(tmp_path: Path) -> None:
    ids = [f"q{i:03d}" for i in range(50)]
    split = cal.split_dev_test(ids, n_dev=10, seed=4)
    assert len(split.dev) == 10
    assert len(split.test) == 40
    assert set(split.dev).isdisjoint(split.test)
    assert set(split.dev) | set(split.test) == set(ids)
    assert cal.split_dev_test(list(reversed(ids)), n_dev=10, seed=4) == split
    assert cal.split_dev_test(ids, n_dev=10, seed=5) != split
    cal.save_split(split, tmp_path / "split.json")
    assert cal.load_split(tmp_path / "split.json") == split


def test_split_validation() -> None:
    with pytest.raises(cal.CalibrationError, match="unique"):
        cal.split_dev_test(["a", "a", "b"], 1)
    with pytest.raises(cal.CalibrationError, match="n_dev"):
        cal.split_dev_test(["a", "b"], 2)
    with pytest.raises(cal.CalibrationError, match="overlap"):
        cal.Split(seed=0, dev=("a",), test=("a", "b"))


def label(item_id: str, correct: bool, sampling: str = "uniform") -> LabelRecord:
    return LabelRecord(
        item_id=item_id,
        labels={"correct": correct},
        labeler="t",
        sampling=sampling,  # type: ignore[arg-type]
        seed=0,
        created_at="2026-09-28T00:00:00+00:00",
    )


def judged(item_id: str, correct: bool | None, fingerprint: str = "fp1") -> EvalRecord:
    if correct is None:
        return record(
            item_id, scores={}, error="JudgeParseError: x", meta={"judge_fingerprint": fingerprint}
        )
    return record(item_id, scores={"correct": correct}, meta={"judge_fingerprint": fingerprint})


def test_pair_labels_aligns_and_excludes_errors() -> None:
    judge = [judged("q1", True), judged("q2", False), judged("q3", None), judged("q4", True)]
    labels = [label("q1", True), label("q2", True), label("q3", False), label("q4", False)]
    pairs = cal.pair_labels(judge, labels, "correct", on_error="exclude")
    assert pairs.item_ids == ("q1", "q2", "q4")
    assert pairs.judge == (True, False, True)
    assert pairs.human == (True, True, False)
    assert pairs.excluded == ("q3",)
    only = cal.pair_labels(judge, labels, "correct", only_items={"q1", "q2"}, on_error="exclude")
    assert only.item_ids == ("q1", "q2")


def test_pair_labels_refuses_disagreement_sampled_labels() -> None:
    judge = [judged("q1", True), judged("q2", False)]
    labels = [label("q1", True), label("q2", False, sampling="disagreement")]
    with pytest.raises(cal.CalibrationError, match="not a uniform sample"):
        cal.pair_labels(judge, labels, "correct")


def test_pair_labels_refuses_mixed_judge_versions() -> None:
    judge = [judged("q1", True, "fp1"), judged("q2", False, "fp2")]
    with pytest.raises(cal.CalibrationError, match="2 different judge versions"):
        cal.pair_labels(judge, [label("q1", True)], "correct")


def test_pair_labels_needs_a_judge_result_for_every_label() -> None:
    with pytest.raises(RecordError, match="no judge result"):
        cal.pair_labels([judged("q1", True)], [label("q9", True)], "correct")


def test_corrected_pass_rate_coverage_is_reasonable() -> None:
    """Coarse simulation: the 95% CI should cover the true rate most of the time."""
    rng = np.random.default_rng(11)
    true_rate, tpr, tnr, covered, trials = 0.7, 0.9, 0.8, 0, 150
    for t in range(trials):
        truth = rng.uniform(size=150) < true_rate
        flip = np.where(truth, rng.uniform(size=150) > tpr, rng.uniform(size=150) > tnr)
        test = truth ^ flip
        cal_truth = rng.uniform(size=80) < true_rate
        cal_flip = np.where(cal_truth, rng.uniform(size=80) > tpr, rng.uniform(size=80) > tnr)
        result = cal.corrected_pass_rate(test, cal_truth ^ cal_flip, cal_truth, n_boot=1000, seed=t)
        covered += result.corrected.low <= true_rate <= result.corrected.high
    assert covered / trials > 0.88
