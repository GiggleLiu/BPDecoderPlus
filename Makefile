.PHONY: help install setup dev-setup test test-cov generate-dataset generate-dem generate-syndromes docs docs-serve clean format lint lint-fix type-check pre-commit check

help:
	@echo "Available targets:"
	@echo "  install            - Install uv package manager"
	@echo "  setup              - Set up development environment with uv"
	@echo "  dev-setup          - Set up development environment and install pre-commit hooks"
	@echo "  generate-dataset   - Generate noisy circuit dataset"
	@echo "  generate-dem       - Generate detector error models"
	@echo "  generate-syndromes - Generate syndrome database (1000 shots)"
	@echo "  test               - Run tests"
	@echo "  test-cov           - Run tests with coverage report"
	@echo "  format             - Format code with ruff"
	@echo "  lint               - Run ruff linting"
	@echo "  lint-fix           - Run ruff linting with auto-fix"
	@echo "  type-check         - Run mypy type checking"
	@echo "  pre-commit         - Run pre-commit on all files"
	@echo "  check              - Run all checks (lint, type-check, test)"
	@echo "  docs               - Build documentation"
	@echo "  docs-serve         - Serve documentation locally"
	@echo "  clean              - Remove generated files and caches"

install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}

setup: install
	uv sync --dev

dev-setup: setup
	uv run pre-commit install
	@echo "Development environment ready!"

generate-dataset:
	uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 5 7 --task z --output datasets

generate-dem:
	uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 5 7 --task z --output datasets --generate-dem

generate-syndromes:
	uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 5 7 --task z --output datasets --generate-syndromes 1000

test:
	uv run pytest

test-cov:
	uv run pytest --cov=bpdecoderplus --cov-report=html --cov-report=term

format:
	uv run ruff format .

lint:
	uv run ruff check .

lint-fix:
	uv run ruff check --fix .

type-check:
	uv run mypy src/bpdecoderplus

pre-commit:
	uv run pre-commit run --all-files

check: lint type-check test
	@echo "All checks passed!"

docs:
	pip install mkdocs-material mkdocstrings[python] pymdown-extensions
	mkdocs build

docs-serve:
	pip install mkdocs-material mkdocstrings[python] pymdown-extensions
	mkdocs serve

clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
