.PHONY: install install-dev install-full clean test format check run-hooks update-hooks

# get path to current makefile
MKFILE_PATH := $(realpath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))

# NOTE ONLY WORKS WITH .venv
VENV_DIR   := $(MKFILE_DIR).venv
PYTHON     := $(VENV_DIR)/bin/python3
PIP        := $(VENV_DIR)/bin/pip3


CUDA_VERSION ?= cu124
TORCH_VERSION := 2.4.1


PYG_URL := https://data.pyg.org/whl/torch-$(TORCH_VERSION)+$(CUDA_VERSION).html
DGL_URL := https://data.dgl.ai/wheels/torch-2.4/$(CUDA_VERSION)/repo.html


FIND_LINKS := --find-links $(PYG_URL) --find-links $(DGL_URL)

install:
	$(PIP) install -e . $(FIND_LINKS)

install-dev: _install-dev setup-hooks test

_install-dev:
	$(PIP) install -e ".[dev]" $(FIND_LINKS)

install-full: _install-full setup-hooks test

_install-full:
	$(PIP) install -e ".[full]" $(FIND_LINKS)

test:
	$(PYTHON) -m pytest tests/ -v

format:
	ruff format src/ scripts/ tests/
	@echo "✅ Code formatted with ruff"

lint:
	ruff check src/ scripts/ tests/
	@echo "✅ Linting complete"

lint-fix:
	ruff check --fix src/ scripts/ tests/
	@echo "✅ Auto-fixed linting issues"

# check both format and lint (without modifying files)
check:
	@echo "Checking code format..."
	ruff format --check src/ scripts/ tests/
	@echo "Checking code quality..."
	ruff check src/ scripts/ tests/
	@echo "✅ All checks passed"

setup-hooks:
	pre-commit install
	pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed"

run-hooks:
	pre-commit run --all-files

update-hooks:
	pre-commit autoupdate
	@echo "✅ Hooks updated"

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "Available targets:"
	@echo "  install           - Install base dependencies"
	@echo "  install-dev       - Install with dev dependencies"
	@echo "  install-full      - Install everything (default: cu124)"
	@echo "  test              - Run all tests"
	@echo "  setup-hooks       - Setup pre-commit hooks"
	@echo "  run-hooks         - Run pre-commit hooks"
	@echo "  update-hooks      - Update pre-commit hooks"
	@echo "  format            - Run ruff format"
	@echo "  lint              - Run ruff linting"
	@echo "  lint-fix          - Run ruff linting with fixes if applicable"
	@echo "  check             - Run ruff format check + linting"
	@echo "  clean             - Clean build artifacts"
