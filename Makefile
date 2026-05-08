.PHONY: install install-mps test lint format check clean

install:
	uv sync

# Reserved for optional Metal/accelerated stacks (e.g. PyTorch MPS); currently same as install.
install-mps:
	uv sync

test:
	uv run pytest

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

check:
	uv run ruff check .
	uv run ruff format --check .
	uv run mypy src/
	uv run pytest

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true
