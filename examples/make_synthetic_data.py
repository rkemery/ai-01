"""Generate the SYNTHETIC demo data in examples/synthetic/.

Nothing here comes from a real model or a real person. A fixed seed simulates:

- 40 support questions about 8 made-up help-center articles (5 per article)
- answers from a "baseline" and a "candidate" system, with a latent truth for
  each check and a per-article difficulty so items in one article correlate
- a judge that sees the truth and flips it at fixed error rates, run through
  the real `ChecklistJudge` with a `FakeClient`, including one garbled reply
- "human" labels that equal the latent truth

Run from the repo root:  uv run python examples/make_synthetic_data.py
"""

from __future__ import annotations

import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np

from llm_eval_harness.calibration import save_split, split_dev_test
from llm_eval_harness.client import DEFAULT_PRICES, FakeClient, ModelResponse, cost_usd
from llm_eval_harness.judge import Checklist, ChecklistItem, ChecklistJudge
from llm_eval_harness.labeling import LabelRecord
from llm_eval_harness.records import EvalRecord, write_records

SEED = 20260928
ANSWER_MODEL = "gpt-6-luna"
JUDGE_MODEL = "Llama-3.3-70B-Instruct"
N_DEV = 10

CHECKLIST = Checklist(
    name="synthetic-support-answer-v1",
    items=(
        ChecklistItem(
            "correct",
            "Does the answer state the same key facts as the reference, with no contradiction?",
        ),
        ChecklistItem("grounded", "Is every claim in the answer supported by the reference?"),
    ),
)

# (article id, reference, wrong answer, unsupported extra claim, five questions)
ARTICLES = [
    (
        "card-freeze",
        "Freeze your card in the app under Cards, then Freeze card. It takes effect right away "
        "and you can unfreeze it the same way.",
        "Call support to freeze your card. It can take up to 24 hours.",
        " We will also mail you a replacement card for free.",
        [
            "How do I freeze my card?",
            "Can I lock my debit card from the app?",
            "How fast does a card freeze take effect?",
            "How do I unfreeze my card?",
            "I misplaced my card. How do I stop payments on it?",
        ],
    ),
    (
        "overdraft",
        "There is no overdraft fee. A payment that would overdraw your account is declined.",
        "Overdrafts cost $15 each, up to three per day.",
        " You can also ask for an overdraft limit of up to $500.",
        [
            "Do you charge overdraft fees?",
            "What happens if I spend more than my balance?",
            "Will a payment go through if I do not have enough money?",
            "How much is the overdraft fee?",
            "Can my account go negative?",
        ],
    ),
    (
        "wire-limits",
        "Outgoing domestic wires are limited to $10,000 per day. International wires are not "
        "supported.",
        "You can wire up to $50,000 per day, including international wires.",
        " Wires sent before 2 pm arrive the same day.",
        [
            "What is the daily wire limit?",
            "Can I send an international wire?",
            "How much can I wire in one day?",
            "Is there a cap on domestic wires?",
            "Can I wire $20,000 today?",
        ],
    ),
    (
        "disputes",
        "Dispute a card charge in the app within 60 days of the statement date. We reply within "
        "10 business days.",
        "Disputes must be filed within 7 days and take 30 days to resolve.",
        " You get a provisional credit as soon as you file.",
        [
            "How do I dispute a charge?",
            "How long do I have to dispute a transaction?",
            "When will I hear back about my dispute?",
            "Can I dispute a charge from last month?",
            "Where do I report a charge I do not recognize?",
        ],
    ),
    (
        "savings-apy",
        "The savings account pays 3.10% APY on the full balance. The rate is variable.",
        "Savings pays a fixed 5% APY on balances over $1,000.",
        " Interest is paid out daily.",
        [
            "What interest rate does savings pay?",
            "Is the savings APY fixed?",
            "Do I need a minimum balance to earn interest?",
            "What is the current APY?",
            "Can the savings rate change?",
        ],
    ),
    (
        "closing",
        "Close your account in the app under Settings, then Close account, after moving your "
        "balance out. Closing is free.",
        "Closing an account costs $25 and needs a signed letter.",
        " A closed account can be reopened within 90 days.",
        [
            "How do I close my account?",
            "Is there a fee to close my account?",
            "What do I need to do before closing my account?",
            "Can I close my account in the app?",
            "Do I need to send a letter to close my account?",
        ],
    ),
    (
        "atm-fees",
        "In-network ATM withdrawals are free. Out-of-network withdrawals cost $2.50 from us, "
        "plus any operator fee.",
        "All ATM withdrawals are free anywhere.",
        " We refund operator fees up to $10 a month.",
        [
            "Are ATM withdrawals free?",
            "What does an out-of-network ATM cost?",
            "Do you refund ATM operator fees?",
            "Which ATMs can I use for free?",
            "Why was I charged at an ATM?",
        ],
    ),
    (
        "direct-deposit",
        "Direct deposits can arrive up to 2 days early, as soon as your employer sends the "
        "payment file.",
        "Direct deposits always arrive on payday at 5 pm.",
        " Early deposit also works for tax refunds.",
        [
            "When does my paycheck arrive?",
            "Can I get paid early?",
            "Why did my direct deposit come early?",
            "How early can direct deposit arrive?",
            "Does early direct deposit depend on my employer?",
        ],
    ),
]

# Simulated judge error rates per check, (TPR, TNR).
JUDGE_ERROR = {"correct": (0.92, 0.80), "grounded": (0.90, 0.70)}
GARBLED_ITEM = "q-disputes-3"  # the candidate judge reply for this item is not JSON


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def simulate(rng: np.random.Generator) -> tuple[list[dict], dict[str, dict], dict[str, dict]]:
    """Items plus latent truth for each system, keyed by item id."""
    items, truth = [], {"baseline": {}, "candidate": {}}
    for article, reference, wrong, extra, questions in ARTICLES:
        difficulty = rng.normal(0.0, 0.8)
        for k, question in enumerate(questions, start=1):
            item_id = f"q-{article}-{k}"
            u = rng.uniform()
            for system, shift, p_extra in (("baseline", 0.6, 0.10), ("candidate", 1.3, 0.25)):
                p_correct = _sigmoid(shift + difficulty)
                # Mostly shared noise across systems, so the runs are paired like real ones.
                draw = u if rng.uniform() < 0.8 else rng.uniform()
                correct = bool(draw < p_correct)
                has_extra = bool(rng.uniform() < p_extra)
                answer = (reference if correct else wrong) + (extra if has_extra else "")
                truth[system][item_id] = {
                    "answer": answer,
                    "correct": correct,
                    "grounded": correct and not has_extra,
                }
            items.append(
                {
                    "item_id": item_id,
                    "cluster": article,
                    "question": question,
                    "reference": reference,
                }
            )
    return items, truth["baseline"], truth["candidate"]


def judge_replies(
    rng: np.random.Generator, items: list[dict], truth: dict, garbled: str | None
) -> list[str]:
    replies = []
    for item in items:
        if item["item_id"] == garbled:
            replies.append("Both checks pass. The answer matches the reference.")
            continue
        verdict = {}
        for check, (tpr, tnr) in JUDGE_ERROR.items():
            actual = truth[item["item_id"]][check]
            keep = rng.uniform() < (tpr if actual else tnr)
            passed = actual if keep else not actual
            verdict[check] = {"pass": passed, "reason": "synthetic verdict"}
        replies.append(json.dumps(verdict))
    return replies


def build_run(
    rng: np.random.Generator, system: str, items: list[dict], truth: dict, garbled: str | None
) -> list[EvalRecord]:
    judge = ChecklistJudge(
        FakeClient(judge_replies(rng, items, truth, garbled)),
        JUDGE_MODEL,
        CHECKLIST,
        temperature=0.0,
    )
    records = []
    for item in items:
        answer = truth[item["item_id"]]["answer"]
        outcome = judge.score(item["question"], answer, item["reference"])
        usage = ModelResponse(
            text=answer,
            model=ANSWER_MODEL,
            input_tokens=int(rng.normal(1400, 150)),
            output_tokens=max(20, int(rng.normal(120 if system == "baseline" else 150, 30))),
            latency_ms=float(np.round(rng.lognormal(np.log(900), 0.3), 1)),
        )
        records.append(
            EvalRecord(
                run_id=f"demo-{system}",
                item_id=item["item_id"],
                config=system,
                model=ANSWER_MODEL,
                # pii_leak stands in for a code-based check (regex or Presidio) in real use.
                scores={**outcome.scores, "pii_leak": False},
                cluster=item["cluster"],
                tokens_in=usage.input_tokens,
                tokens_out=usage.output_tokens,
                cost_usd=round(cost_usd(DEFAULT_PRICES[ANSWER_MODEL], usage), 8),
                latency_ms=usage.latency_ms,
                error=outcome.error,
                meta={"synthetic": True, "judge_fingerprint": judge.fingerprint},
            )
        )
    return records


def main(out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    items, baseline_truth, candidate_truth = simulate(rng)
    out_dir.mkdir(parents=True, exist_ok=True)

    checklist = {
        "name": CHECKLIST.name,
        "items": [{"id": i.id, "question": i.question} for i in CHECKLIST.items],
    }
    (out_dir / "checklist.json").write_text(
        json.dumps(checklist, indent=2) + "\n", encoding="utf-8"
    )

    write_records(
        out_dir / "baseline.jsonl", build_run(rng, "baseline", items, baseline_truth, None)
    )
    write_records(
        out_dir / "candidate.jsonl",
        build_run(rng, "candidate", items, candidate_truth, GARBLED_ITEM),
    )

    with (out_dir / "items.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            row = {
                "item_id": item["item_id"],
                "question": item["question"],
                "reference": item["reference"],
                "answer": candidate_truth[item["item_id"]]["answer"],
                "meta": {"synthetic": True, "config": "candidate", "model": ANSWER_MODEL},
            }
            fh.write(json.dumps(row) + "\n")

    with (out_dir / "human_labels.jsonl").open("w", encoding="utf-8") as fh:
        for item in items:
            truth = candidate_truth[item["item_id"]]
            label = LabelRecord(
                item_id=item["item_id"],
                labels={check: truth[check] for check in CHECKLIST.ids},
                labeler="synthetic-human",
                sampling="uniform",
                seed=SEED,
                created_at="2026-09-28T00:00:00+00:00",
            )
            fh.write(json.dumps(asdict(label)) + "\n")

    save_split(
        split_dev_test([i["item_id"] for i in items], N_DEV, seed=SEED), out_dir / "split.json"
    )


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "synthetic")
