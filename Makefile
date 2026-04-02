.PHONY: venv install install-dev install-bench install-full clean test format check run-hooks update-hooks

PYTHON_BIN   ?= $(HOME)/micromamba/envs/graph_ml/bin/python
CUDA_HOME    ?= /usr/local/cuda

# Derive CUDA version tag (e.g. "cu129") from CUDA_HOME/bin/nvcc --version
CUDA_VERSION := $(shell $(CUDA_HOME)/bin/nvcc --version 2>/dev/null \
    | grep -oP 'release \K[0-9]+\.[0-9]+' \
    | tr -d '.' \
    | sed 's/^/cu/')

MKFILE_PATH := $(realpath $(lastword $(MAKEFILE_LIST)))
MKFILE_DIR  := $(dir $(MKFILE_PATH))

VENV_DIR   := $(MKFILE_DIR)/.venv
PYTHON     := $(VENV_DIR)/bin/python3
PIP        := $(VENV_DIR)/bin/pip3

TORCH_INDEX := https://download.pytorch.org/whl/$(CUDA_VERSION)
PYG_URL     := https://data.pyg.org/whl/torch-2.4.1+$(CUDA_VERSION).html
DGL_URL     := https://data.dgl.ai/wheels/torch-2.4/$(CUDA_VERSION)/repo.html

NO_ISO := --no-build-isolation

## Auxillary targets

test:
	$(PYTHON) -m pytest tests/ -v

format:
	ruff format src/ scripts/ tests/ turbo_gnn/
	@echo "Code formatted with ruff"

lint:
	ruff check src/ scripts/ tests/ turbo_gnn/
	@echo "Linting complete"

lint-fix:
	ruff check --fix src/ scripts/ tests/ turbo_gnn/
	@echo "Auto-fixed linting issues"

# check both format and lint (without modifying files)
check:
	@echo "Checking code format..."
	ruff format --check src/ scripts/ tests/ turbo_gnn/
	@echo "Checking code quality..."
	ruff check src/ scripts/ tests/ turbo_gnn/
	@echo "All checks passed"

setup-hooks:
	$(VENV_DIR)/bin/pre-commit install
	$(VENV_DIR)/bin/pre-commit install --hook-type commit-msg
	@echo "Pre-commit hooks installed"

run-hooks:
	$(VENV_DIR)/bin/pre-commit run --all-files

update-hooks:
	$(VENV_DIR)/bin/pre-commit autoupdate
	@echo "Hooks updated"

clean:
	rm -rf build/ dist/ *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .pytest_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name .mypy_cache -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true





## build targets
venv:
	@if [ ! -d "$(VENV_DIR)" ]; then \
		echo "Creating virtual environment with $(PYTHON_BIN)..."; \
		$(PYTHON_BIN) -m venv $(VENV_DIR); \
		$(PIP) install -U pip; \
	else \
		echo "Virtual environment already exists at $(VENV_DIR)"; \
	fi





# Base: just turbo-gnn CUDA kernels with existing torch
install: venv
	$(PIP) install wheel numpy ninja packaging psutil "setuptools>=77.0"
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e . $(NO_ISO)





# Dev: research + PyG + test + lint (no DGL)
# Torch is installed separately via --index-url to force the correct CUDA build.
# Build tools come from PyPI in a second step.
# Query the latest torch version from the CUDA-specific index, then pin it.
# Bare `pip install torch --index-url` can silently pick the wrong CUDA build.
_install-torch-dev: venv
	TORCH_VER=$$($(PIP) index versions torch --index-url $(TORCH_INDEX) 2>/dev/null \
		| head -1 | sed 's/.*(\(.*\))/\1/') && \
	$(PIP) install "torch==$$TORCH_VER" --index-url $(TORCH_INDEX)
	$(PIP) install wheel numpy ninja packaging psutil "setuptools>=77.0"

# Freeze the already-installed torch so pip doesn't replace it with a PyPI build.
_install-dev:
	$(PIP) freeze | grep "^torch==" > /tmp/_torch_constraint.txt
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e ".[dev]" $(NO_ISO) -c /tmp/_torch_constraint.txt

install-dev: venv _install-torch-dev _install-dev setup-hooks test





# Bench: dev + DGL + viz + cugraph (torch==2.4.1 for DGL compat)
_install-torch-bench: venv
	$(PIP) install "torch==2.4.1+$(CUDA_VERSION)" --index-url $(TORCH_INDEX)
	$(PIP) install wheel numpy ninja packaging psutil "setuptools>=77.0"

_install-bench:
	$(PIP) freeze | grep "^torch==" > /tmp/_torch_constraint.txt
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e ".[bench]" $(NO_ISO) -c /tmp/_torch_constraint.txt --find-links $(PYG_URL) --find-links $(DGL_URL)

## patch triton for the version using DGL and older torch
_patch-triton:
	$(PIP) install -U triton

install-bench: venv _install-torch-bench _install-bench _install-tcgnn _patch-triton setup-hooks test





# Full: bench + notebook + tracking
_install-tcgnn:
	mkdir -p thirdparty
	git clone https://github.com/MachineLearningSystem/ATC23-TCGNN-Pytorch thirdparty/tcgnn || true
	cd thirdparty/tcgnn/TCGNN_conv && CUDA_HOME=/usr/local/cuda-12/ LD_LIBRARY_PATH=/usr/local/cuda-12/lib64/ $(PYTHON) setup.py install || true
	cd ../../.

_install-full:
	$(PIP) freeze | grep "^torch==" > /tmp/_torch_constraint.txt
	CUDA_HOME=$(CUDA_HOME) $(PIP) install -e ".[full]" $(NO_ISO) -c /tmp/_torch_constraint.txt --find-links $(PYG_URL) --find-links $(DGL_URL)

install-full: venv _install-torch-bench _install-full _install-tcgnn setup-hooks test


help:
	@echo "Available targets:"
	@echo "  install           - Install turbo-gnn only (base: CUDA kernels + torch)"
	@echo "  install-dev       - Dev environment: research + PyG + tests (no DGL, torch>=2.9)"
	@echo "  install-bench     - Benchmarking: dev + DGL + viz + cugraph (torch==2.4.1)"
	@echo "  install-full      - Everything: bench + notebooks + tracking"
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
	@echo "Override defaults: make install-dev CUDA_VERSION=cu128 PYTHON_BIN=/path/to/python"
