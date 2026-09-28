.PHONY: install lint format test demo data

install:
	uv sync --all-extras

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff format .
	uv run ruff check --fix .

test:
	uv run pytest -q

# Runs offline on synthetic data and rewrites the demo section of README.md.
demo:
	uv run llm-eval demo

# Regenerates examples/synthetic/ from its fixed seed. Run `make demo` after.
data:
	uv run python examples/make_synthetic_data.py
