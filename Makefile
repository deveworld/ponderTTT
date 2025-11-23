.PHONY: install test lint format clean help

help:
	@echo "PonderTTT Development Commands:"
	@echo "  make check     - Check code with ruff, ty, pyright, pyrefly"

check:
	uv run ruff check . --fix
	uv run ty check
	uv run pyright .
	uv run pyrefly check . 
