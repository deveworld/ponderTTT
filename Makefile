.PHONY: install test lint format clean help

help:
	@echo "PonderTTT Development Commands:"
	@echo "  make install    - Install the package and dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters (ruff, mypy)"
	@echo "  make format     - Format code with ruff"
	@echo "  make clean      - Clean generated files"

install:
	uv pip install -e .
	uv pip install -Ue ".[all]"

test:
	python scripts/test_pipeline.py
	pytest tests/ -v

lint:
	ruff check src/
	mypy src/

format:
	ruff format src/
	ruff check --fix src/

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
