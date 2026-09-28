# AI Engineering Portfolio: Build Plan (v2, 2026-09-28)

Five public repos built around one fictional neobank's customer support team. Every repo leads with a results table, runs with one command and no keys, and states its limits.

Budget: $0 cash. LLM calls are paid from Visual Studio monthly Azure credits (spending limit left on, no card on file). Hugging Face and Colab are free tiers.

## Models

### Azure (Foundry, sold directly by Azure, Global Standard, paid by credits)

| Role | Model | $/1M in / cached / out | Notes |
|---|---|---|---|
| RAG answers, agents, zero-shot baseline | `gpt-6-luna` | 0.10 / 0.01 / 0.50 | Reasoning effort none or low. Keep prompts in the short-context band. |
| Frontier comparison (Banking77, run once) | `gpt-6-sol` | 2.00 / 0.20 / 10.00 | Full 3,080 test set, retrieved few-shot, cached prefix. |
| Judge (cross-family) | `DeepSeek-V4-Flash` (GA, pinned, not `-0731`) | 0.19 / 0.028 / 0.51 | Thinking stays on, so budget for hidden tokens. Billed under "Azure Deepseek Models". Confirm credit drawdown on day 1. |
| Fallback judge, gateway fallback | `gpt-oss-120b` | 0.15 / n/a / 0.60 | Azure OpenAI OSS line, so credits clearly apply. Same vendor as luna, so disclose it. |
| Contingency if GPT-6 quota is refused | `gpt-5-mini` or `gpt-5.4-mini` | 0.25 / 2.00, 0.75 / 4.50 | Only if day-1 smoke test fails. |

Not used: gpt-4.1-mini/nano, gpt-4o-mini (deprecated for new deployments), `gpt-6-astra` (too expensive for the signal), Marketplace models such as Claude (not credit-eligible), Batch API (no batch quota on credit subscriptions for these models, so prompt caching replaces it).

Client: `openai` SDK against the Foundry `/openai/v1/` endpoint, Entra ID via `azure-identity` (`get_bearer_token_provider`), Responses API. SDK `max_retries=0`, our own retry policy.

### Open models (Hugging Face, all Apache-2.0 or MIT, CPU)

| Role | Model | Where it runs |
|---|---|---|
| Embedding baseline | `BAAI/bge-small-en-v1.5` (33M) | Runtime |
| Embedding arm | `ibm-granite/granite-embedding-small-english-r2` (47M) | Runtime |
| Embedding strong arm | `Qwen/Qwen3-Embedding-0.6B` | Offline benchmark only |
| Reranker | `ibm-granite/granite-embedding-reranker-english-r2` (149M), top-10 | Runtime |
| Reranker strong arm | `Qwen/Qwen3-Reranker-0.6B` | Offline benchmark only |
| Hallucination second opinion | `vectara/hallucination_evaluation_model` (HHEM-2.1-Open) | Offline eval |
| Fine-tune, CPU | `Qwen/Qwen3-0.6B-Base` + LoRA, sequence-classification head | Container |
| Fine-tune, Colab T4 | `Qwen/Qwen3-8B-Base` QLoRA (one run) | Colab |
| Encoder baseline | `answerdotai/ModernBERT-base` | Container or Colab |
| Injection detectors | `leolee99/PIGuard`, `protectai/deberta-v3-base-prompt-injection-v2` | Runtime |
| Injection detector (optional, gated) | `meta-llama/Llama-Prompt-Guard-2-22M` | Optional |
| PII | Microsoft Presidio + custom Luhn/IBAN recognizers | Runtime |

Not used: Qwen3.5-0.8B-Base on CPU (it is a vision-language checkpoint, so it stays an optional Colab arm via the text-only class), EmbeddingGemma and ShieldGemma (gated Gemma terms), jina v5 (non-commercial), Llama Guard 4 12B and gpt-oss-safeguard-20b (too big for CPU).

## Repo 1: `llm-eval-harness` (built first)

The core is dependency-light (numpy, scipy) so every other repo can import it without dragging in a framework.

MUST
- JSONL results contract: one record per item per run, with model, config, scores, tokens, cost, latency.
- Stats: Wilson intervals for small N, clustered standard errors, paired McNemar and paired bootstrap for comparing runs, an MDE line printed with every result, pass^k for repeated trials.
- Judge: binary checklist items (not 1 to 5 scales), reference-guided, pairwise mode that runs both orders and reports the flip rate.
- Calibration: judge TPR and TNR against human labels, Cohen's kappa with a bootstrap CI, bias-corrected pass rate with a CI that includes calibration uncertainty (Lee et al., arXiv 2511.21140).
- Labeling CLI: blind, randomized, stores labels as JSONL. Uniform random sample for agreement metrics, disagreement sampling only for finding judge bugs.
- Cached model client with a hard dollar cap computed from `usage`.
- CI: every PR replays cached responses with fake clients. Live runs only on manual `workflow_dispatch`. A regression blocks only when the paired CI says it is real.

SHOULD
- Inspect AI adapter: the harness repo's own demo eval runs on Inspect AI (pinned `inspect-ai==0.3.271`), and an adapter converts Inspect logs to the JSONL contract. Other repos don't need Inspect installed.

CUT
- Krippendorff's alpha and Gwet's AC1 (kappa plus TPR/TNR is enough for one rater), judge juries, scheduled nightly or weekly live runs, a shared Inspect runtime across repos.

## Repo 2: `rag-support-assistant`

MUST
- Corpus of about 150 synthetic help-center articles with deliberate distractors (versioned policies, per-plan variants) so retrieval isn't at ceiling.
- 200 questions split 50 dev and 150 test, gold article IDs, 40 unanswerable (out of scope, false premise, near miss). Test is reported once.
- Retrieval grid scored with `ranx`, no LLM: about 8 configs changed one factor at a time. Fixed-token vs header-aware chunks with titles, dense (bge vs granite) vs hybrid BM25 (convex fusion tuned on dev vs RRF), reranker on vs off. Recall@k, MRR@10, nDCG@10.
- Full-context baseline: the whole corpus in the prompt with no retrieval, run once on luna.
- Generation and judging on the 3 best configs only.
- Abstention 2x2 table (answerable or not vs answered or abstained), false refusal rate, hallucination rate (answers with at least one unsupported claim).
- Bootstrap CIs clustered by source article, paired comparisons, MDE line.
- 150 final-config answers labeled blind by a human, which gives a human-measured headline and validates the judge at the same time.
- Stack: LlamaIndex (`llama-index-core` 0.14.x) and Qdrant local mode (`qdrant-client` 1.19.x), fusion method set explicitly.

SHOULD
- One contextual-retrieval cell (short LLM-written context per chunk).
- HHEM as a non-LLM second opinion on faithfulness.
- Ragas 0.4.3 (pinned, wrapped behind the harness interface) on 2 configs.
- Qwen3-Embedding-0.6B and Qwen3-Reranker-0.6B as offline "strong arm" rows.

CUT: semantic chunking, late chunking, HyDE, ColBERT, RAGChecker, MLflow/OTel tracing (per-query JSONL logs of latency, tokens and cost instead).

## Repo 3: `banking77-lora-vs-frontier`

Data: `legacy-datasets/banking77` (parquet, CC-BY-4.0). `PolyAI/banking77` no longer loads with current `datasets`.

Positioning: a public repo (ank018/lora-banking77) already shows LoRA on a small Qwen roughly ties roberta-base with full data and wins below about 500 labels. We cite it. Our addition is the frontier-API arm with cost and latency, plus proper error bars.

MUST
- Sequence-classification head with PEFT LoRA (`task_type=SEQ_CLS`), LoRA on all linear layers, rank 16 to 32, alpha 32, learning rate about 10x full fine-tuning (LoRA Without Regret recipe). Tune on a dev split from train, never on test.
- Hyperparameter sweep (3 learning rates, 3 seeds) on Colab T4. The container trains only the final config on Qwen3-0.6B-Base and measures CPU latency.
- Baselines: logistic regression on frozen bge-small embeddings, fine-tuned ModernBERT-base.
- Prompting arms, all on the full 3,080 test set with structured outputs forcing one of 77 labels and a cacheable static prefix: luna zero-shot with label descriptions, luna retrieved few-shot (10 to 20 nearest training examples), sol retrieved few-shot (run once).
- Metrics: accuracy, macro-F1, 95% bootstrap CIs, McNemar for pairwise differences, seed spread, calibration error for local models, p50/p95 latency, cost per 1k predictions from measured usage.
- Deduplicated-test row (13.8% of test items have a near twin in train).
- Honest framing: "within X points at 1/Y the cost", not "beats frontier". Note roughly 14% label noise and that contamination of big models can't be ruled out.
- LoRA adapter pushed to the user's HF account with a model card. Free CPU Space demo of the classifier.

SHOULD: learning curve at 5, 10, 20 examples per class plus full data (the regime where the LLM should win), one Qwen3-8B-Base QLoRA run on Colab.

CUT: gpt-6-astra, static few-shot on sol, generative label ablation, fresh hand-written queries, RL fine-tuning (one README line on why: classification has a verifiable label, so SFT gets the full signal).

## Repo 4: `support-triage-agents`

Framework: LangGraph (`langgraph>=1.2,<2`), explicit `StateGraph`, no prebuilt swarm/supervisor.

MUST
- Agents that differ in permissions and context, not just prompts. Intake classifies. Researcher retrieves from a frozen snapshot of the RAG index. Resolver writes a typed `ActionPlan` but can't execute it. Compliance reviewer sees only ticket facts, the draft and policy excerpts. Executor applies changes only after approval. Typed Pydantic handoffs, at most 2 bounce-backs.
- Human-in-the-loop: `interrupt()` in a `human_review` node (approve, edit, reject), `SqliteSaver` checkpointer, idempotent tools keyed on ticket and action, a kill-and-resume demo.
- Fake bank backend in SQLite, exposed as tools.
- 50 tasks, LLM-drafted and human-audited, each with a gold final DB state, a `should_escalate` label and a hidden fact sheet.
- Three arms on the same tasks: A single agent (same model, tools, docs, step budget), B full graph, C graph without reviewer. k=4 for A and B, k=2 for C.
- Metrics: final-DB-state success (checked by code), pass^1 and pass^k, escalation precision and recall, policy violations, dollars per resolved ticket, tokens, p50/p95 latency. Bootstrap CIs and McNemar.
- Pre-registered hypothesis in the README: the graph wins on policy violations, not raw resolution. Report it either way.
- 40 failures hand-tagged with codes adapted from the MAST taxonomy.

SHOULD: tools served by an in-repo stdio MCP server (both arms use the same server), an LLM customer simulator for the roughly 20% of tasks that need clarification.

CUT or DEFER: tau3-bench banking run (deferred, and if ever run, in a separate env with LiteLLM pinned below 1.82.7), a sol reviewer arm, LLM-based failure tagging, Langfuse/LangSmith, a Microsoft Agent Framework port, A2A.

## Repo 5: `guarded-llm-gateway`

MUST
- FastAPI + Pydantic in front of the RAG assistant. Layers in order: input validation, PII redaction (Presidio with Luhn/IBAN recognizers), injection detection (PIGuard, deberta baseline, Azure Prompt Shields with results cached), spotlighting of retrieved documents, the model call, output rules (canary token block, PII echo check, link and markdown allowlist, schema validation with one repair try).
- Foundry's default content filter set to annotate-only and logged as its own layer, so the ablation isn't contaminated.
- Reliability: slowapi rate limits plus a per-key token budget with 429 and Retry-After, an overall `asyncio.timeout` deadline, retries with jitter, a small async circuit breaker tested with a fake clock, fallback luna then gpt-oss-120b then a retrieval-only answer then 503. Fault injection tests report the fallback rate.
- Attack suite of about 150, frozen as JSONL with provenance and license: direct (deepset/prompt-injections, JailbreakBench, Lakera gandalf), indirect (LLMail-Inject payloads planted in KB chunks), about 30 hand-written neobank attacks (the headline number), obfuscated variants, output-handling and consumption attacks. Benign traffic: Banking77 test, the RAG questions, NotInject.
- Thresholds tuned on a dev split, results on held-out only. End-to-end attack success rate judged by code (canary leaked, PII echoed, link emitted), detector TPR at 1% FPR, FPR per benign set, Wilson CIs, added latency per layer, which layer caught each attack.
- A 1-hour manual adaptive attack session, reported in its own column, with a note that static suites overstate robustness.
- Prometheus metrics, Docker with CPU-only torch, default profile loads only models under about 200M parameters.

SHOULD: OWASP LLM Top 10 (2026) table listing only the rows that have tests, a targeted garak scan.

CUT: MITRE ATLAS mapping, PyRIT, openai/privacy-filter at runtime, Qwen3Guard at runtime, XSTest, LiteLLM anywhere.

## Cross-repo rules

- Share data, not code. The neobank corpus, questions, agent tasks and attacks live in one HF dataset, pinned by revision. The harness is pinned by git tag.
- Every repo: `uv run make demo` works with no keys (replayed cache) and regenerates the README results table.
- Every README: results table, architecture diagram, what we measured and why, design trade-offs, what didn't work, limitations, cost of a full run.

## Budget and compute

Estimates from the fit review (list prices, 30 to 90% cache hits, gpt-5-mini as judge, which is pricier per token than DeepSeek-V4-Flash):

| Repo | Final run | With about 2x reruns | Month |
|---|---|---|---|
| Harness | $3.7 | $7.3 | 1 |
| RAG | $3.8 | $7.6 | 1 |
| Banking77 (sol run once, about $7) | $8.7 | $10.1 | 2 |
| Agents | $4.9 | $9.7 | 2 |
| Gateway | $0.5 | $0.9 | 3 |
| Total | about $21 | about $36 | peak about $20 in one month |

Fits a $50/month credit with margin. If the quota tier forces everything onto gpt-5-mini, a final run is $28 to $49, so Ragas and pairwise judging get cut and agents drop to k=2.

Compute: about 4 to 10 hours of CPU in the container, 2 to 6 hours on free Colab.

Cost guards: in-client dollar cap, Azure budget alerts at $25 and $40, low TPM on every deployment, spending limit left on.

## Human labeling (the user's time)

About 21 hours total, about 9 of it pure labeling:

| Task | Count | Time |
|---|---|---|
| Blind-label final RAG answers (3 yes/no checks each) | 150 | 3.1 h |
| Relabel a sample a week later (intra-rater check) | 30 | 0.6 h |
| Pairwise preferences (optional) | 60 | 1.0 h |
| Review RAG questions and gold article IDs | 200 | 3.3 h |
| Audit LLM-drafted agent tasks | 50 | 8.3 h |
| Hand-tag agent failures | 40 | 2.7 h |
| Hand-written attacks plus adaptive session | 30 | 2.5 h |

## Day 1 checklist (Azure)

1. Check the subscription's quota tier (`quotaTiers` API).
2. Deploy and smoke-test luna, sol, gpt-oss-120b and DeepSeek-V4-Flash at low TPM.
3. After 24 hours, confirm in Cost Management that DeepSeek usage drew down credits. If not, gpt-oss-120b becomes the judge and the README discloses the same-vendor judge.
4. Test whether luna accepts `temperature` and `logprobs` (the docs conflict).
5. Set budget alerts.

## Build order

1. Shared data (HF dataset) and `llm-eval-harness` v0.1.0.
2. RAG, Banking77 and gateway in parallel.
3. Agents, once the RAG index snapshot exists.
4. One reviewer per repo: clone fresh, run the README cold, try to break the code and the claims.
