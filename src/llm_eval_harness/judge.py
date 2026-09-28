"""LLM judges: a binary, reference-guided checklist judge and a pairwise judge.

Checklist questions are yes/no, not 1 to 5 ratings. CheckEval (Lee et al.,
EMNLP 2025, arXiv 2403.18771) found that breaking a rubric into binary
checklist questions gives more consistent and more explainable judgments than
Likert scales. The gold reference answer goes into the prompt, which is the
reference-guided setup from Zheng et al. (2023, arXiv 2306.05685).

Replies must be strict JSON, with no duplicate keys. Anything else raises
`JudgeParseError`.
`ChecklistJudge.score` turns that into an error on the result, so a garbled
reply never counts as a pass.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from string import Template
from typing import Any, Literal

from llm_eval_harness.client import ModelClient, ModelRequest, ModelResponse
from llm_eval_harness.stats import Interval, wilson_interval

_ID_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
_FENCE = re.compile(r"^```(?:json)?[ \t]*\n(.*)\n```$", re.DOTALL)

Winner = Literal["a", "b", "tie"]
# Map the judge's "1" / "2" back to answers A and B for each presentation order.
_A_FIRST: dict[str, Winner] = {"1": "a", "2": "b", "tie": "tie"}
_B_FIRST: dict[str, Winner] = {"1": "b", "2": "a", "tie": "tie"}


class JudgeParseError(ValueError):
    """The judge's reply is not the strict JSON the prompt asked for."""

    def __init__(self, message: str, raw: str, response: ModelResponse | None = None) -> None:
        super().__init__(message)
        self.raw = raw
        self.response = response


@dataclass(frozen=True)
class ChecklistItem:
    id: str
    question: str


@dataclass(frozen=True)
class Checklist:
    name: str
    items: tuple[ChecklistItem, ...]

    def __post_init__(self) -> None:
        if not self.items:
            raise ValueError("a checklist needs at least one item")
        ids = [item.id for item in self.items]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate checklist ids in {ids}")
        for item in self.items:
            if not _ID_PATTERN.match(item.id):
                raise ValueError(f"checklist id {item.id!r} must match {_ID_PATTERN.pattern}")
            if not item.question.strip():
                raise ValueError(f"checklist item {item.id!r} has an empty question")

    @property
    def ids(self) -> tuple[str, ...]:
        return tuple(item.id for item in self.items)


def load_checklist(path: str | Path) -> Checklist:
    """Load `{"name": ..., "items": [{"id": ..., "question": ...}, ...]}`."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not isinstance(data.get("items"), list):
        raise ValueError(f"{path}: expected an object with an 'items' list")
    items = []
    for entry in data["items"]:
        if (
            not isinstance(entry, dict)
            or set(entry) != {"id", "question"}
            or not all(isinstance(v, str) for v in entry.values())
        ):
            raise ValueError(f"{path}: each item needs string 'id' and 'question', got {entry!r}")
        items.append(ChecklistItem(id=entry["id"], question=entry["question"]))
    return Checklist(name=str(data.get("name", Path(path).stem)), items=tuple(items))


def load_template(name: str) -> str:
    """Read a prompt template shipped in `llm_eval_harness/prompts/`."""
    return (resources.files("llm_eval_harness") / "prompts" / name).read_text(encoding="utf-8")


@dataclass(frozen=True)
class CheckResult:
    passed: bool
    reason: str


@dataclass(frozen=True)
class ChecklistVerdict:
    results: dict[str, CheckResult]
    response: ModelResponse

    @property
    def scores(self) -> dict[str, bool]:
        return {item_id: result.passed for item_id, result in self.results.items()}


@dataclass(frozen=True)
class JudgeOutcome:
    """Result of `ChecklistJudge.score`: scores on success, an error message otherwise.

    Store `error` in the record's `score_error`, not its `error`: the answer
    that was judged exists, only its scoring failed.
    """

    scores: dict[str, bool]
    reasons: dict[str, str]
    error: str | None
    response: ModelResponse | None


def parse_checklist_reply(text: str, item_ids: Sequence[str]) -> dict[str, CheckResult]:
    """Parse `{item_id: {"pass": bool, "reason": str}}` with exactly the expected keys.

    A single Markdown code fence around the object is tolerated because it
    changes nothing about the content. Everything else is strict: no extra or
    missing keys, and "pass" must be a JSON boolean, not "yes" or 1.
    """
    data = _load_json_object(text)
    missing = [i for i in item_ids if i not in data]
    unexpected = sorted(set(data) - set(item_ids))
    if missing or unexpected:
        raise JudgeParseError(
            f"reply keys do not match the checklist (missing {missing}, unexpected {unexpected})",
            raw=text,
        )
    results: dict[str, CheckResult] = {}
    for item_id in item_ids:
        entry = data[item_id]
        if not isinstance(entry, dict) or set(entry) != {"pass", "reason"}:
            raise JudgeParseError(
                f"{item_id!r} must be an object with exactly 'pass' and 'reason'", raw=text
            )
        if not isinstance(entry["pass"], bool):
            raise JudgeParseError(f"{item_id!r}: 'pass' must be true or false", raw=text)
        if not isinstance(entry["reason"], str):
            raise JudgeParseError(f"{item_id!r}: 'reason' must be a string", raw=text)
        results[item_id] = CheckResult(passed=entry["pass"], reason=entry["reason"])
    return results


class ChecklistJudge:
    """Grades one answer against a reference with a binary checklist in one call."""

    def __init__(
        self,
        client: ModelClient,
        model: str,
        checklist: Checklist,
        *,
        template: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.checklist = checklist
        self.template = load_template("checklist_judge.txt") if template is None else template
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    @property
    def fingerprint(self) -> str:
        """Short hash of everything that defines this judge.

        Store it with results. Calibration refuses to mix verdicts from judges
        with different fingerprints, which is how "freeze the judge after tuning
        on dev labels" gets enforced.
        """
        spec = {
            "kind": "checklist",
            "model": self.model,
            "template": self.template,
            "checklist": [[item.id, item.question] for item in self.checklist.items],
            "max_output_tokens": self.max_output_tokens,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }
        blob = json.dumps(spec, sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def build_request(self, question: str, answer: str, reference: str) -> ModelRequest:
        checklist = "\n".join(f"- {item.id}: {item.question}" for item in self.checklist.items)
        example = json.dumps(
            {item_id: {"pass": True, "reason": "..."} for item_id in self.checklist.ids}
        )
        prompt = Template(self.template).substitute(
            question=question,
            reference=reference,
            answer=answer,
            checklist=checklist,
            item_ids=", ".join(self.checklist.ids),
            example=example,
        )
        return ModelRequest(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    def judge(self, question: str, answer: str, reference: str) -> ChecklistVerdict:
        """Call the model and parse its reply. Raises `JudgeParseError` on a bad reply."""
        response = self.client.complete(self.build_request(question, answer, reference))
        try:
            results = parse_checklist_reply(response.text, self.checklist.ids)
        except JudgeParseError as exc:
            raise JudgeParseError(str(exc), raw=exc.raw, response=response) from exc
        return ChecklistVerdict(results=results, response=response)

    def score(self, question: str, answer: str, reference: str) -> JudgeOutcome:
        """Like `judge`, but a parse failure comes back as an error instead of raising.

        Only `JudgeParseError` is caught. Budget, cache and network errors still
        propagate, since those mean the run itself is broken.
        """
        try:
            verdict = self.judge(question, answer, reference)
        except JudgeParseError as exc:
            return JudgeOutcome(
                scores={}, reasons={}, error=f"JudgeParseError: {exc}", response=exc.response
            )
        reasons = {item_id: r.reason for item_id, r in verdict.results.items()}
        return JudgeOutcome(
            scores=verdict.scores, reasons=reasons, error=None, response=verdict.response
        )


@dataclass(frozen=True)
class PairwiseVerdict:
    """Reconciled verdict from both presentation orders.

    `first_order` is the verdict with answer A shown first, `second_order` with
    B shown first, both mapped back to "a" / "b" / "tie". If they differ, the
    verdict flipped with position and the final `winner` is "tie".
    """

    winner: Winner
    first_order: Winner
    second_order: Winner
    reasons: tuple[str, str]
    responses: tuple[ModelResponse, ModelResponse]

    @property
    def flipped(self) -> bool:
        return self.first_order != self.second_order


@dataclass(frozen=True)
class PairwiseSummary:
    n: int
    a_wins: int
    b_wins: int
    ties: int
    flips: int
    flip_rate: Interval


def parse_pairwise_reply(text: str) -> tuple[str, str]:
    """Parse `{"winner": "1" | "2" | "tie", "reason": str}` strictly."""
    data = _load_json_object(text)
    if set(data) != {"winner", "reason"}:
        raise JudgeParseError(f"expected keys 'winner' and 'reason', got {sorted(data)}", raw=text)
    winner, reason = data["winner"], data["reason"]
    if winner not in ("1", "2", "tie"):
        raise JudgeParseError(f'\'winner\' must be "1", "2" or "tie", got {winner!r}', raw=text)
    if not isinstance(reason, str):
        raise JudgeParseError("'reason' must be a string", raw=text)
    return winner, reason


class PairwiseJudge:
    """Compares two answers, asking twice with the order swapped to expose position bias.

    Zheng et al. (2023, arXiv 2306.05685) found LLM judges often prefer whichever
    answer comes first, and recommend swapping positions and calling an
    inconsistent pair a tie. That is what `compare` does.
    """

    def __init__(
        self,
        client: ModelClient,
        model: str,
        *,
        template: str | None = None,
        max_output_tokens: int | None = None,
        temperature: float | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.template = load_template("pairwise_judge.txt") if template is None else template
        self.max_output_tokens = max_output_tokens
        self.temperature = temperature
        self.reasoning_effort = reasoning_effort

    def build_request(
        self, question: str, answer_1: str, answer_2: str, reference: str
    ) -> ModelRequest:
        prompt = Template(self.template).substitute(
            question=question, reference=reference, answer_1=answer_1, answer_2=answer_2
        )
        return ModelRequest(
            model=self.model,
            input=prompt,
            max_output_tokens=self.max_output_tokens,
            temperature=self.temperature,
            reasoning_effort=self.reasoning_effort,
        )

    def compare(
        self, question: str, answer_a: str, answer_b: str, reference: str
    ) -> PairwiseVerdict:
        """Run A-first then B-first. Raises `JudgeParseError` if either reply is malformed."""
        first = self._ask(question, answer_a, answer_b, reference)
        second = self._ask(question, answer_b, answer_a, reference)
        first_order = _A_FIRST[first[0]]
        second_order = _B_FIRST[second[0]]
        winner: Winner = first_order if first_order == second_order else "tie"
        return PairwiseVerdict(
            winner=winner,
            first_order=first_order,
            second_order=second_order,
            reasons=(first[1], second[1]),
            responses=(first[2], second[2]),
        )

    def _ask(
        self, question: str, answer_1: str, answer_2: str, reference: str
    ) -> tuple[str, str, ModelResponse]:
        response = self.client.complete(self.build_request(question, answer_1, answer_2, reference))
        try:
            winner, reason = parse_pairwise_reply(response.text)
        except JudgeParseError as exc:
            raise JudgeParseError(str(exc), raw=exc.raw, response=response) from exc
        return winner, reason, response


def summarize_pairwise(verdicts: Sequence[PairwiseVerdict]) -> PairwiseSummary:
    """Win counts after reconciliation, plus the flip rate with a Wilson interval."""
    if not verdicts:
        raise ValueError("no verdicts to summarize")
    flips = sum(v.flipped for v in verdicts)
    return PairwiseSummary(
        n=len(verdicts),
        a_wins=sum(v.winner == "a" for v in verdicts),
        b_wins=sum(v.winner == "b" for v in verdicts),
        ties=sum(v.winner == "tie" for v in verdicts),
        flips=flips,
        flip_rate=wilson_interval(flips, len(verdicts)),
    )


class _DuplicateKey(ValueError):
    pass


def _unique_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    """`object_pairs_hook` that rejects duplicate keys instead of keeping the last one."""
    data: dict[str, Any] = {}
    for key, value in pairs:
        if key in data:
            raise _DuplicateKey(key)
        data[key] = value
    return data


def _load_json_object(text: str) -> dict:
    """Parse the reply as one JSON object. Duplicate keys at any depth are an error.

    Python's json keeps the last of two equal keys, so a reply with "correct"
    twice would be scored by whichever came second, and a fail could become a
    pass. The judge was asked for one verdict per check, so two is malformed.
    """
    body = text.strip()
    fenced = _FENCE.match(body)
    if fenced:
        body = fenced.group(1)
    try:
        data = json.loads(body, object_pairs_hook=_unique_keys)
    except json.JSONDecodeError as exc:
        raise JudgeParseError(f"reply is not valid JSON ({exc.msg})", raw=text) from exc
    except _DuplicateKey as exc:
        raise JudgeParseError(f"reply has a duplicate key {exc.args[0]!r}", raw=text) from exc
    if not isinstance(data, dict):
        raise JudgeParseError(f"reply must be a JSON object, got {type(data).__name__}", raw=text)
    return data
