.PHONY: install test lint format clean help

help:
	@echo "PonderTTT Development Commands:"
	@echo "  make check     - Check code with ruff and pyright"

check:
	uv run ruff check . --fix
	uv run pyright .
 
