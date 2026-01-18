.PHONY: help install setup test test-cov generate-dataset clean

help:
	@echo "Available targets:"
	@echo "  install          - Install uv package manager"
	@echo "  setup            - Set up development environment with uv"
	@echo "  generate-dataset - Generate noisy circuit dataset"
	@echo "  test             - Run tests"
	@echo "  test-cov         - Run tests with coverage report"
	@echo "  clean            - Remove generated files and caches"

install:
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}

setup: install
	uv sync --dev

generate-dataset:
	uv run generate-noisy-circuits --distance 3 --p 0.01 --rounds 3 5 7 --task z --output datasets/noisy_circuits

test:
	uv run pytest

test-cov:
	uv run pytest --cov=bpdecoderplus --cov-report=html --cov-report=term

clean:
	rm -rf .pytest_cache
	rm -rf __pycache__
	rm -rf htmlcov
	rm -rf .coverage
	rm -rf coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
