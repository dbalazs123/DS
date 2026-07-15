# Thin wrappers over uv so contributors have one obvious command per task.
.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test check docs clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| sort | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Create the environment and install the package + dev tools
	uv sync
	uv run pre-commit install

lint:  ## Lint (ruff) without modifying files
	uv run ruff check .

format:  ## Auto-format and auto-fix lint issues
	uv run ruff format .
	uv run ruff check --fix .

typecheck:  ## Static type checking (mypy)
	uv run mypy

test:  ## Run the test suite with coverage
	uv run pytest

check: lint typecheck test  ## Run everything CI runs

docs:  ## Build the documentation site
	uv run mkdocs build

clean:  ## Remove caches and build artifacts
	rm -rf .mypy_cache .pytest_cache .ruff_cache htmlcov site dist build .coverage
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
