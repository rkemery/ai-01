"""`llm-eval` command line. Standard library argparse only."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from llm_eval_harness import __version__
from llm_eval_harness.analysis import compare_runs, pass_k_from_records, summarize_metric
from llm_eval_harness.calibration import (
    check_same_judge,
    corrected_pass_rate,
    judge_agreement,
    labeled_checks,
    load_split,
    pair_labels,
    save_split,
    split_dev_test,
)
from llm_eval_harness.demo import DEFAULT_DATA_DIR, SECTION, run_demo
from llm_eval_harness.gate import (
    DEFAULT_MAX_EXCLUDED,
    Floor,
    GateMetric,
    format_gate,
    run_gate,
)
from llm_eval_harness.judge import load_checklist
from llm_eval_harness.labeling import (
    LabelRecord,
    disagreeing_items,
    labeling_order,
    read_items,
    read_labels,
    run_session,
)
from llm_eval_harness.records import (
    EvalRecord,
    MissingScoreError,
    metric_column,
    read_records,
    single_run_id,
)
from llm_eval_harness.report import (
    agreement_table,
    comparison_methods_line,
    comparison_table,
    corrected_note,
    corrected_table,
    mde_line,
    methods_line,
    pass_k_table,
    results_table,
    write_section,
)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return args.func(args)
    except MissingScoreError as exc:
        hint = ""
        if getattr(args, "on_error", None) is not None:
            hint = " Rerun with --on-error exclude to leave errored items out and report them."
        print(f"llm-eval {args.command}: error: {exc.detail}.{hint}", file=sys.stderr)
        return 2
    except (ValueError, OSError) as exc:
        # Contract, label, calibration and file errors end the command with a message.
        print(f"llm-eval {args.command}: error: {exc}", file=sys.stderr)
        return 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-eval",
        description="Offline stats, labeling, calibration and CI gating for LLM evals.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("stats", help="metrics with CIs and MDE for a results file")
    p.add_argument("results", type=Path, help="results JSONL (one run, or several trials with --k)")
    _metric_args(p)
    mode = p.add_mutually_exclusive_group()
    mode.add_argument(
        "--compare", type=Path, metavar="BASELINE", help="paired comparison vs this run"
    )
    mode.add_argument("--k", type=int, help="pass^k and pass@k over runs (each run_id is a trial)")
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("report", help="markdown results table, optionally written into a file")
    p.add_argument("results", type=Path)
    _metric_args(p)
    p.add_argument("--write", type=Path, metavar="FILE", help="file with section markers to update")
    p.add_argument("--section", default="results", help="marker name (default: results)")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("label", help="blind, randomized, resumable labeling in the terminal")
    p.add_argument("--items", type=Path, required=True, help="JSONL with item_id, question, answer")
    p.add_argument("--checklist", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True, help="labels JSONL (appended, resumable)")
    p.add_argument("--labeler", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--n", type=_positive_int, help="label only the first N items of the shuffled order"
    )
    p.add_argument("--mode", choices=["uniform", "disagreement"], default="uniform")
    p.add_argument("--judge-a", type=Path, help="first judge results (disagreement mode)")
    p.add_argument("--judge-b", type=Path, help="second judge results (disagreement mode)")
    p.add_argument("--split", type=Path, help="split JSON from `llm-eval split`")
    p.add_argument("--split-part", choices=["dev", "test"], default="dev")
    p.set_defaults(func=cmd_label)

    p = sub.add_parser("split", help="seeded dev/test split of item ids")
    p.add_argument("--items", type=Path, required=True, help="any JSONL with an item_id field")
    p.add_argument("--n-dev", type=int, required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=cmd_split)

    p = sub.add_parser("calibrate", help="judge vs human agreement and corrected pass rates")
    p.add_argument("--judge", type=Path, required=True, help="judge results on the labeled items")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument(
        "--check",
        action="append",
        help="check to calibrate (repeatable, default: every labeled one)",
    )
    p.add_argument(
        "--split",
        type=Path,
        required=True,
        help="split JSON from `llm-eval split`. Only --split-part labels are used, so labels "
        "used to tune the judge never count",
    )
    p.add_argument("--split-part", choices=["dev", "test"], default="test")
    p.add_argument(
        "--apply",
        type=Path,
        metavar="RESULTS",
        help="results to correct, judged by the same judge (fingerprint is checked)",
    )
    p.add_argument("--cluster", action="store_true", help="cluster the --apply bootstrap")
    p.add_argument(
        "--on-error",
        choices=["raise", "exclude"],
        default="raise",
        help="errored records in the --apply results",
    )
    _boot_args(p)
    p.set_defaults(func=cmd_calibrate)

    p = sub.add_parser("gate", help="CI gate: hard floors and paired regression checks")
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--floor", action="append", default=[], help="e.g. pii_leak:max=0")
    p.add_argument(
        "--metric", action="append", default=[], help="regression metric, NAME or NAME:lower"
    )
    p.add_argument("--cluster", action="store_true", help="clustered t-test using record.cluster")
    p.add_argument("--on-error", choices=["raise", "exclude"], default="raise")
    p.add_argument(
        "--max-excluded",
        type=float,
        default=DEFAULT_MAX_EXCLUDED,
        metavar="FRACTION",
        help="with --on-error exclude, block if more than this share of items errored "
        f"(default {DEFAULT_MAX_EXCLUDED})",
    )
    _boot_args(p)
    p.set_defaults(func=cmd_gate)

    p = sub.add_parser("demo", help="run the offline synthetic demo and refresh the README")
    p.add_argument("--data", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--readme", type=Path, default=Path("README.md"))
    p.add_argument("--no-write", action="store_true", help="print only, leave the README alone")
    p.set_defaults(func=cmd_demo)
    return parser


def cmd_stats(args: argparse.Namespace) -> int:
    records = read_records(args.results)
    metrics = args.metric or _score_names(records)
    if args.k is not None:
        if args.cluster:
            raise ValueError(
                "--cluster is not supported with --k: the pass^k interval resamples tasks"
            )
        if args.on_error != "raise":
            raise ValueError(
                "--on-error exclude does not apply to --k: pass^k counts every trial, and "
                "leaving errored trials out would inflate pass^k"
            )
        for metric in metrics:
            result = pass_k_from_records(
                records, metric, args.k, n_boot=args.n_boot, seed=args.seed
            )
            print(pass_k_table(result, metric))
        return 0
    print(_summary_markdown(records, metrics, args))
    if args.compare is not None:
        baseline = read_records(args.compare)
        comparisons = [
            compare_runs(
                baseline,
                records,
                metric,
                use_clusters=args.cluster,
                on_error=args.on_error,
                n_boot=args.n_boot,
                seed=args.seed,
            )
            for metric in metrics
        ]
        print()
        print(f"Paired: {args.results} minus {args.compare}")
        print()
        print(comparison_table(comparisons))
        print()
        print(comparison_methods_line(comparisons))
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    records = read_records(args.results)
    body = _summary_markdown(records, args.metric or _score_names(records), args)
    print(body)
    if args.write is not None:
        changed = write_section(args.write, args.section, body)
        print(f"\n{args.write}: section {args.section!r} {'updated' if changed else 'unchanged'}")
    return 0


def cmd_label(args: argparse.Namespace) -> int:
    items = read_items(args.items)
    if args.split is not None:
        keep = set(getattr(load_split(args.split), args.split_part))
        items = [item for item in items if item.item_id in keep]
    checklist = load_checklist(args.checklist)
    disagreements = None
    if args.mode == "disagreement":
        if args.judge_a is None or args.judge_b is None:
            raise ValueError("--mode disagreement needs --judge-a and --judge-b")
        disagreements = disagreeing_items(
            read_records(args.judge_a), read_records(args.judge_b), checklist.ids
        )
    ordered = labeling_order(items, args.seed, args.mode, disagreements)
    run_session(
        ordered,
        checklist,
        args.out,
        labeler=args.labeler,
        sampling=args.mode,
        seed=args.seed,
        limit=args.n,
        input_fn=input,
        print_fn=print,
    )
    return 0


def cmd_split(args: argparse.Namespace) -> int:
    ids = []
    with args.items.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if line.strip():
                row = json.loads(line)
                if not isinstance(row, dict) or not isinstance(row.get("item_id"), str):
                    raise ValueError(f"{args.items}:{lineno}: no string item_id")
                ids.append(row["item_id"])
    split = split_dev_test(ids, args.n_dev, seed=args.seed)
    save_split(split, args.out)
    print(f"{len(split.dev)} dev and {len(split.test)} test items written to {args.out}")
    return 0


def cmd_calibrate(args: argparse.Namespace) -> int:
    judge_records = read_records(args.judge)
    labels = read_labels(args.labels)
    checks = args.check or _calibration_checks(judge_records, labels)
    only = getattr(load_split(args.split), args.split_part)
    pairs = {
        c: pair_labels(judge_records, labels, c, only_items=only, on_error="exclude")
        for c in checks
    }
    agreements = [
        (c, judge_agreement(p.judge, p.human, n_boot=args.n_boot, seed=args.seed))
        for c, p in pairs.items()
    ]
    print(f"Judge vs human labels on the {args.split_part} split ({len(only)} items)")
    print()
    print(agreement_table(agreements))
    for c, p in pairs.items():
        if p.excluded:
            print(f"{c}: {len(p.excluded)} labeled items skipped because the judge errored")
    if args.apply is not None:
        applied = read_records(args.apply)
        fingerprint = check_same_judge(judge_records, applied)
        results = []
        excluded: dict[str, int] = {}
        for c, p in pairs.items():
            column = metric_column(applied, c, on_error=args.on_error)
            if column.excluded:
                excluded[c] = len(column.excluded)
            ids = sorted(column.values)
            clusters = [column.clusters[i] for i in ids] if args.cluster else None
            results.append(
                (
                    c,
                    corrected_pass_rate(
                        [column.values[i] == 1.0 for i in ids],
                        p.judge,
                        p.human,
                        test_clusters=clusters,
                        n_boot=args.n_boot,
                        seed=args.seed,
                    ),
                )
            )
        print()
        print(f"Corrected pass rates for {single_run_id(applied)} (judge {fingerprint})")
        print()
        print(corrected_table(results))
        print()
        print(corrected_note(results))
        for c, n in excluded.items():
            print(f"{c}: {n} errored items left out of the corrected pass rate")
    return 0


def cmd_gate(args: argparse.Namespace) -> int:
    result = run_gate(
        read_records(args.baseline),
        read_records(args.candidate),
        floors=[Floor.parse(spec) for spec in args.floor],
        metrics=[GateMetric.parse(spec) for spec in args.metric],
        use_clusters=args.cluster,
        on_error=args.on_error,
        max_excluded=args.max_excluded,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    print(format_gate(result))
    return result.exit_code


def cmd_demo(args: argparse.Namespace) -> int:
    body = run_demo(args.data)
    print(body)
    if not args.no_write:
        changed = write_section(args.readme, SECTION, body)
        print(f"\n{args.readme}: demo section {'updated' if changed else 'unchanged'}")
    return 0


def _summary_markdown(
    records: list[EvalRecord], metrics: list[str], args: argparse.Namespace
) -> str:
    summaries = [
        summarize_metric(records, m, use_clusters=args.cluster, on_error=args.on_error)
        for m in metrics
    ]
    return "\n\n".join([results_table(summaries), mde_line(summaries), methods_line(summaries)])


def _calibration_checks(judge_records: list[EvalRecord], labels: list[LabelRecord]) -> list[str]:
    scored = set(_score_names(judge_records))
    checks = [c for c in labeled_checks(labels) if c in scored]
    if not checks:
        raise ValueError("no check is both labeled and in the judge results, pass --check")
    return checks


def _score_names(records: list[EvalRecord]) -> list[str]:
    names = sorted({name for record in records for name in record.scores})
    if not names:
        raise ValueError("no scores found, pass --metric")
    return names


def _metric_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--metric", action="append", help="metric to report (default: every score)")
    p.add_argument("--cluster", action="store_true", help="clustered CIs using record.cluster")
    p.add_argument("--on-error", choices=["raise", "exclude"], default="raise")
    _boot_args(p)


def _positive_int(text: str) -> int:
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {value}")
    return value


def _boot_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--n-boot", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=0)


if __name__ == "__main__":
    raise SystemExit(main())
