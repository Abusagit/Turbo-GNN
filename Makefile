.PHONY: venv install install-dev install-full clean test format check run-hooks update-hooks

PYTHON_BIN   ?= $(HOME)/micromamba/envs/graph_ml/bin/python
CUDA_VERSION ?= cu124
CUDA_HOME    ?= /usr/local/cuda-12.4

MKFILE_PATH := $(realpath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))

VENV_DIR   := $(MKFILE_DIR)/.venv
PYTHON     := $(VENV_DIR)/bin/python3
PIP        := $(VENV_DIR)/bin/pip3

TORCH_VERSION := 2.4.1
PYG_URL := https://data.pyg.org/whl/torch-$(TORCH_VERSION)+$(CUDA_VERSION).html
DGL_URL := https://data.dgl.ai/wheels/torch-2.4/$(CUDA_VERSION)/repo.html


FIND_LINKS := --find-links $(PYG_URL) --find-links $(DGL_URL)
NO_ISO := --no-build-isolation

venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment with $(PYTHON_BIN)..."; \
		$(PYTHON_BIN) -m venv $(VENV_DIR); \
		$(PIP) install -U pip; \
	else \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	fi

_install-torch: venv
	$(PIP) install torch==$(TORCH_VERSION) wheel numpy ninja packaging psutil "setuptools>=77.0"

_patch-triton:
	$(PIP) install -U triton

install: _install-torch
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e . $(NO_ISO) $(FIND_LINKS)

install-dev: _install-torch _install-dev _patch-triton setup-hooks test

_install-dev:
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e ".[dev]" $(NO_ISO) $(FIND_LINKS)

install-full: _install-torch _install-full _patch-triton _install-tcgnn setup-hooks test

_install-full:
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e ".[full]" $(NO_ISO) $(FIND_LINKS)

_install-tcgnn:
	mkdir -p thirdparty
	git clone https://github.com/MachineLearningSystem/ATC23-TCGNN-Pytorch thirdparty/tcgnn || true
	cd thirdparty/tcgnn/TCGNN_conv && CUDA_HOME=/usr/local/cuda-12/ LD_LIBRARY_PATH=/usr/local/cuda-12/lib64/ $(PYTHON) setup.py install || true
	cd ../../.

test:
	$(PYTHON) -m pytest tests/ -v

format:
	ruff format src/ scripts/ tests/ turbo_gnn/
	@echo "✅ Code formatted with ruff"

lint:
	ruff check src/ scripts/ tests/ turbo_gnn/
	@echo "✅ Linting complete"

lint-fix:
	ruff check --fix src/ scripts/ tests/ turbo_gnn/
	@echo "✅ Auto-fixed linting issues"

# check both format and lint (without modifying files)
check:
	@echo "Checking code format..."
	ruff format --check src/ scripts/ tests/ turbo_gnn/
	@echo "Checking code quality..."
	ruff check src/ scripts/ tests/ turbo_gnn/
	@echo "✅ All checks passed"

setup-hooks:
	$(VENV_DIR)/bin/pre-commit install
	$(VENV_DIR)/bin/pre-commit install --hook-type commit-msg
	@echo "✅ Pre-commit hooks installed"

run-hooks:
	$(VENV_DIR)/bin/pre-commit run --all-files

update-hooks:
	$(VENV_DIR)/bin/pre-commit autoupdate
	@echo "✅ Hooks updated"

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

help:
	@echo "Available targets:"
	@echo "  install           - Install turbo-gnn (base: torch only)"
	@echo "  install-dev       - Install with dev/test dependencies"
	@echo "  install-full      - Install everything + TCGNN (default: cu124)"
	@echo "  test              - Run all tests"
	@echo "  setup-hooks       - Setup pre-commit hooks"
	@echo "  run-hooks         - Run pre-commit hooks"
	@echo "  update-hooks      - Update pre-commit hooks"
	@echo "  format            - Run ruff format"
	@echo "  lint              - Run ruff linting"
	@echo "  lint-fix          - Run ruff linting with fixes if applicable"
	@echo "  check             - Run ruff format check + linting"
	@echo "  clean             - Clean build artifacts"
	@echo ""
	@echo "Override defaults: make install CUDA_VERSION=cu128 PYTHON_BIN=/path/to/python"
