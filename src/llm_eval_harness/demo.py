"""Offline demo on the synthetic data in examples/synthetic/.

Runs calibration, stats, a paired comparison and the gate with no keys and no
network, and renders the markdown that goes into the README's demo section.
"""

from __future__ import annotations

from pathlib import Path

from llm_eval_harness.analysis import compare_runs, summarize_metric
from llm_eval_harness.calibration import (
    corrected_pass_rate,
    judge_agreement,
    load_split,
    pair_labels,
)
from llm_eval_harness.gate import Floor, GateMetric, format_gate, run_gate
from llm_eval_harness.judge import load_checklist
from llm_eval_harness.labeling import read_labels
from llm_eval_harness.records import metric_column, read_records
from llm_eval_harness.report import (
    agreement_table,
    comparison_table,
    corrected_table,
    mde_line,
    methods_line,
    results_table,
)

DEFAULT_DATA_DIR = Path("examples/synthetic")
SECTION = "demo"
SEED = 0
N_BOOT = 10_000


def run_demo(data_dir: str | Path = DEFAULT_DATA_DIR) -> str:
    """Return the demo section as markdown."""
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"demo data not found at {data_dir}. Run from the repo root or pass --data."
        )
    checklist = load_checklist(data_dir / "checklist.json")
    baseline = read_records(data_dir / "baseline.jsonl")
    candidate = read_records(data_dir / "candidate.jsonl")
    labels = read_labels(data_dir / "human_labels.jsonl")
    split = load_split(data_dir / "split.json")

    agreements = []
    corrected = []
    excluded: set[str] = set()
    for check in checklist.ids:
        pairs = pair_labels(candidate, labels, check, only_items=split.test, on_error="exclude")
        excluded |= set(pairs.excluded)
        agreements.append(
            (check, judge_agreement(pairs.judge, pairs.human, n_boot=N_BOOT, seed=SEED))
        )
        column = metric_column(baseline, check)
        ids = sorted(column.values)
        corrected.append(
            (
                check,
                corrected_pass_rate(
                    [column.values[i] == 1.0 for i in ids],
                    pairs.judge,
                    pairs.human,
                    test_clusters=[column.clusters[i] for i in ids],
                    n_boot=N_BOOT,
                    seed=SEED,
                ),
            )
        )

    metrics = [*checklist.ids, "pii_leak", "latency_ms"]
    summaries = [
        summarize_metric(candidate, m, use_clusters=True, on_error="exclude") for m in metrics
    ]
    comparisons = [
        compare_runs(
            baseline, candidate, m, use_clusters=True, on_error="exclude", n_boot=N_BOOT, seed=SEED
        )
        for m in checklist.ids
    ]
    gate = run_gate(
        baseline,
        candidate,
        floors=[Floor.parse("pii_leak:max=0")],
        metrics=[GateMetric.parse(m) for m in checklist.ids],
        use_clusters=True,
        on_error="exclude",
        n_boot=N_BOOT,
        seed=SEED,
    )

    n_test = len(split.test)
    lines = [
        "> **Synthetic data.** Every number below comes from simulated answers, a simulated",
        "> judge and simulated human labels in `examples/synthetic/`. They show what the",
        "> tools print. They are not results about any model.",
        "",
        f"**Judge vs human labels** on the {n_test}-item test split "
        f"(dev split of {len(split.dev)} held out for prompt tuning). "
        "TPR and TNR with Wilson 95% CIs, kappa with a bootstrap 95% CI.",
        "",
        agreement_table(agreements),
        "",
        f"Items dropped because the judge reply did not parse: {len(excluded)} "
        f"({', '.join(sorted(excluded)) or 'none'}).",
        "",
        f"**Candidate run** ({len(candidate)} items, CIs clustered by source article).",
        "",
        results_table(summaries),
        "",
        mde_line(summaries),
        "",
        methods_line(summaries),
        "",
        f"Simulated cost of the candidate run at {candidate[0].model} list prices: "
        f"${sum(r.cost_usd for r in candidate):.4f} for {len(candidate)} answers.",
        "",
        "**Candidate vs baseline**, paired by item, bootstrap resampling articles.",
        "",
        comparison_table(comparisons),
        "",
        "**Baseline pass rate corrected for judge error** (Rogan-Gladen, CI includes "
        "calibration uncertainty).",
        "",
        corrected_table(corrected),
        "",
        f"**CI gate** (`llm-eval gate`), exit code {gate.exit_code}:",
        "",
        "```text",
        format_gate(gate),
        "```",
    ]
    return "\n".join(lines)
