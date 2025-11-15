.PHONY: install test lint format clean help

help:
	@echo "PonderTTT Development Commands:"
	@echo "  make install    - Install the package and dependencies"
	@echo "  make test       - Run tests"
	@echo "  make lint       - Run linters (ruff, mypy)"
	@echo "  make format     - Format code with ruff"
	@echo "  make clean      - Clean generated files"
	@echo "  make train-baseline - Train baseline model"
	@echo "  make train-policy   - Train policy"

install:
	uv pip install -e .
	uv pip install -e ".[dev]"

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

train-baseline:
	python -m ponderttt.experiments.train_baseline \
		--model_scale 125m \
		--action UPDATE_1 \
		--max_chunks 100

train-policy:
	python -m ponderttt.experiments.train_policy \
		--model_scale 125m \
		--num_iterations 10
