"""llm-eval-harness: a small, dependency-light toolkit for honest LLM evals.

The core (records, stats, analysis, judge, calibration, labeling, gate,
report) needs only numpy and scipy. The Azure client and the Inspect AI
adapter are optional extras and are not imported here.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("llm-eval-harness")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

from llm_eval_harness.client import (
    BudgetExceeded,
    CachedClient,
    CacheMiss,
    DollarCap,
    FakeClient,
    ModelClient,
    ModelRequest,
    ModelResponse,
    RetryingClient,
)
from llm_eval_harness.judge import ChecklistJudge, JudgeParseError, PairwiseJudge
from llm_eval_harness.records import EvalRecord, RecordError, read_records, write_records

__all__ = [
    "BudgetExceeded",
    "CacheMiss",
    "CachedClient",
    "ChecklistJudge",
    "DollarCap",
    "EvalRecord",
    "FakeClient",
    "JudgeParseError",
    "ModelClient",
    "ModelRequest",
    "ModelResponse",
    "PairwiseJudge",
    "RecordError",
    "RetryingClient",
    "__version__",
    "read_records",
    "write_records",
]
