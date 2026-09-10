# g-SpMM benchmarks on this T4, one entry point.
#
#   make help                      what is here
#   make snapshot                  copy the live tree to turbo-bench/snap-<stamp>, point snap-latest at it
#   make archive-results           move gspmm_vs_dgl/results aside so a sweep re-measures
#   make dgl-16graphs BG=1         turbo vs DGL, 16 graphs, d=64, stages=0, fp32+fp16, fwd/bwd/fb
#   make dgl-matrix BG=1           turbo vs DGL, 4 graphs x {fp32,fp16} x STAGES_LIST x DIMS
#   make dgl-products BG=1         ogbn-products alone (copy_u, d=64)
#   make dgl-one GRAPH=web-fraud   one graph, one dtype, in the foreground
#   make before-after BEFORE=<ref> AFTER=<ref> BG=1   turbo vs turbo across two commits
#   make summary                   markdown tables + per-graph diff against the archived results
#   make plots                     per-cell charts from results/, published as fp32_forward.png etc.
#   make plots-pergraph            one bar per graph (geomean), published as pergraph_fp32_backward.png etc.
#   make plot-custom ARGS="--kind bwd --ops copy_u --reducers min,max"   one chart, your selection
#   make tests                     the g-SpMM correctness suite through the partial-extension shim
#   make prepare-graph NAMES="cora pubmed"   add graphs to the cache (runs under PYTHON_DGL)
#
# Knobs, all overridable on the command line: DTYPES STAGES_LIST DIMS ITERS
# REPEATS (2; 7 costs 3.5x wall clock for <0.5% stability) EDGE_LIMIT (above it a
# graph measures copy_u only) CHECK (0 skips the DGL cross-check) REUSE and
# KEEP_OPERANDS (share one export across stage counts) GRAPHS16 GRAPHS4.
#
# run_matrix.sh ends by publishing charts into THIS directory under fixed,
# committed names (fp32_forward.png, pergraph_*.png, ...), overwriting what is
# there.  A dirty tree after a sweep is expected; review before committing.
#
# There is deliberately no `clean`: gspmm_vs_dgl/results and
# gspmm_before_after/results are gitignored and hold the JSON behind RESULTS.md,
# and another session may be writing into them.  archive-results moves the set
# aside instead, and refuses while a sweep is running.
#
# Every long target measures a SNAPSHOT of the source tree, never the live one:
# other sessions edit csrc/ while a sweep runs, and a run of an hour must not
# straddle their edits.  `make snapshot` takes the copy; the sweeps read
# SNAP (default: turbo-bench/snap-latest, a symlink the snapshot target moves).
# The partial extension is built per snapshot under its own EXT_NAME so the
# shim never picks up another snapshot's build cache.
#
# BG=1 detaches a target with setsid nohup and appends to $(LOGS)/<target>.log;
# without it the target runs in the foreground and tees to the same log.
#
# Two interpreters, on purpose: DGL only has wheels up to python 3.12
# (PYTHON_DGL), turbo runs under the 3.14 venv through python-turbo, which
# installs the partial extension for the checkout named by TURBO_REPO.  The full
# extension does not compile for sm_75 (gatv2's bf16 paths).
#
# The GPU is shared with other sessions: check nvidia-smi and tell them before
# starting anything here.

SHELL := /bin/bash
.ONESHELL:
.SHELLFLAGS := -eu -o pipefail -c

HERE   := $(abspath $(dir $(lastword $(MAKEFILE_LIST))))
REPO   := $(abspath $(HERE)/../..)
BENCH  := /home/ubuntu/work/turbo-bench
LOGS   := $(BENCH)/logs
STAMP  := $(shell date +%m%d_%H%M)

# --- what to measure -------------------------------------------------------
SNAP        ?= $(BENCH)/snap-latest
SNAP_REAL   := $(shell readlink -f $(SNAP) 2>/dev/null)
EXT_NAME    ?= turbo_$(subst -,_,$(notdir $(SNAP_REAL)))_C

GRAPHS16    ?= cora citeseer pubmed city-roads-M city-roads-L artnet-exp tolokers-2 city-reviews twitch-views hm-categories avazu-ctr pokec-regions web-fraud ogbn-arxiv ogbn-proteins ogbn-products
GRAPHS4     ?= random skewed ogbn-arxiv ogbn-products
DTYPES      ?= float32 float16
STAGES_LIST ?= 0
DIMS        ?= 64
ITERS       ?= 30
REPEATS     ?= 2
EDGE_LIMIT  ?= 10000000
CHECK       ?= 1
REUSE       ?= 0
KEEP_OPERANDS ?= 0

# before-after: two commits, measured by the same bench.py
BEFORE      ?= f1bcd8c
AFTER       ?= HEAD
BA_GRAPHS   ?= random skewed ogbn-arxiv web-fraud
BA_DIMS     ?= 32,64,128

# --- this machine ----------------------------------------------------------
PYTHON_DGL   ?= /home/ubuntu/work/dgl-env/bin/python
PYTHON_TURBO ?= $(BENCH)/python-turbo
PYTHON_VENV  ?= $(REPO)/.venv/bin/python3
PYTHON_PLOT  ?= $(PYTHON_VENV)
GRAPH_CACHE  ?= /home/ubuntu/work/graph_cache
OGB_ROOT     ?= /home/ubuntu/work/ogb
GRAPHLAND_ROOT ?= /home/ubuntu/work/graphland
WORK         ?= /home/ubuntu/work/gspmm_scratch

CUDA_ENV := TORCH_CUDA_ARCH_LIST=7.5 CUDA_HOME=/usr/local/cuda NVCC_PREPEND_FLAGS=-DCCCL_DISABLE_CTK_COMPATIBILITY_CHECK PYTHONWARNINGS=ignore

# env every gspmm_vs_dgl run gets: the snapshot is both what the shim builds
# (TURBO_REPO) and what turbo_side.py imports (REPO -> PYTHONPATH)
DGL_ENV := $(CUDA_ENV) \
    OGB_ROOT=$(OGB_ROOT) GRAPHLAND_ROOT=$(GRAPHLAND_ROOT) GRAPH_CACHE=$(GRAPH_CACHE) WORK=$(WORK) \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    TURBO_REPO=$(SNAP_REAL) REPO=$(SNAP_REAL) TURBO_EXT_NAME=$(EXT_NAME) \
    PYTHON_DGL=$(PYTHON_DGL) PYTHON_TURBO=$(PYTHON_TURBO) PYTHON_PLOT=$(PYTHON_PLOT) \
    ITERS=$(ITERS) REPEATS=$(REPEATS) EDGE_LIMIT=$(EDGE_LIMIT) CHECK=$(CHECK) REUSE=$(REUSE) KEEP_OPERANDS=$(KEEP_OPERANDS)

VS_DGL := $(HERE)/gspmm_vs_dgl
BA     := $(HERE)/gspmm_before_after

# run($1 = command, $2 = log stem): detached with BG=1, else foreground + tee
ifeq ($(BG),1)
define run
	mkdir -p $(LOGS)
	setsid nohup bash -c '$(1)' >> $(LOGS)/$(2).log 2>&1 < /dev/null &
	echo "started in background; log: $(LOGS)/$(2).log"
	echo "follow with: tail -f $(LOGS)/$(2).log"
endef
else
define run
	mkdir -p $(LOGS)
	$(1) 2>&1 | tee -a $(LOGS)/$(2).log
endef
endif

.PHONY: help snapshot need-snap no-sweep-running archive-results dgl-16graphs dgl-matrix dgl-products dgl-one \
        before-after summary plots plots-pergraph plot-custom tests prepare-graph gpu

help:
	@sed -n '2,23p' $(lastword $(MAKEFILE_LIST)) | sed 's/^# \{0,1\}//'
	@echo
	@echo "current SNAP:      $(SNAP) -> $(SNAP_REAL)"
	@echo "extension name:    $(EXT_NAME)"
	@echo "results:           $(VS_DGL)/results  ($$(ls $(VS_DGL)/results/results_*.json 2>/dev/null | wc -l) json files)"

gpu:
	@nvidia-smi --query-gpu=name,memory.used,utilization.gpu --format=csv

# Copy of csrc + turbo_gnn + build files, with provenance, and snap-latest
# repointed.  The stale turbo_gnn/_C.so travels along; the shim replaces it.
snapshot:
	dir=$(BENCH)/snap-$(STAMP)
	mkdir -p "$$dir"
	cp -r $(REPO)/csrc $(REPO)/turbo_gnn $(REPO)/setup.py $(REPO)/pyproject.toml "$$dir"/
	rm -rf "$$dir"/turbo_gnn/__pycache__
	git -C $(REPO) rev-parse HEAD > "$$dir/PROVENANCE_head.txt"
	git -C $(REPO) status --short > "$$dir/PROVENANCE_status.txt"
	git -C $(REPO) diff > "$$dir/PROVENANCE_worktree.diff"
	ln -sfn "$$dir" $(BENCH)/snap-latest
	echo "snapshot: $$dir  (HEAD $$(cut -c1-7 "$$dir/PROVENANCE_head.txt"), $$(wc -l < "$$dir/PROVENANCE_status.txt") uncommitted paths)"
	echo "snap-latest -> $$dir"

need-snap:
	@test -n "$(SNAP_REAL)" -a -d "$(SNAP_REAL)/csrc" || { echo "no snapshot at $(SNAP); run 'make snapshot' or pass SNAP=<dir>"; exit 1; }

# The sweeps skip every graph whose results JSON exists, so re-measuring new
# code means moving the old set aside first.  compare_runs.py then diffs the
# two sets per graph (see 'summary').
# Another session's sweep writes into results/ as it goes; moving the directory
# under it would lose that run.
no-sweep-running:
	@pat='gspmm_vs_dgl/(run|run_matrix|sweep_graphs)\.sh|(^|/| )(run|run_matrix|sweep_graphs)\.sh( |$$)|dgl_side\.py|turbo_side\.py'; \
	if pgrep -f "$$pat" >/dev/null; then \
		echo "a gspmm_vs_dgl sweep is running:"; pgrep -af "$$pat" | cut -c1-120; \
		echo "refusing to move results/ under it"; exit 1; fi

archive-results: no-sweep-running
	cd $(VS_DGL)
	if [ -d results ] && ls results/results_*.json >/dev/null 2>&1; then
		mv results results_prev_$(STAMP); mkdir results
		echo "moved gspmm_vs_dgl/results -> results_prev_$(STAMP)"
	else
		mkdir -p results; echo "nothing to archive"
	fi

# --- turbo vs DGL ------------------------------------------------------------
dgl-16graphs: need-snap
	$(call run,cd $(VS_DGL) && env $(DGL_ENV) DTYPES="$(DTYPES)" STAGES_LIST="$(STAGES_LIST)" DIMS=$(DIMS) GRAPHS="$(GRAPHS16)" ./run_matrix.sh,dgl_16graphs_$(STAMP))

dgl-matrix: need-snap
	$(call run,cd $(VS_DGL) && env $(DGL_ENV) DTYPES="$(DTYPES)" STAGES_LIST="$(STAGES_LIST)" DIMS=$(DIMS) GRAPHS="$(GRAPHS4)" ./run_matrix.sh,dgl_matrix_$(STAMP))

dgl-products: need-snap
	$(call run,cd $(VS_DGL) && env $(DGL_ENV) DTYPES="$(DTYPES)" STAGES_LIST="$(STAGES_LIST)" DIMS=64 GRAPHS=ogbn-products ./run_matrix.sh,dgl_products_$(STAMP))

# One (graph, dtype) cell table in the foreground: GRAPH=web-fraud DTYPE=float32 STAGES=0
GRAPH  ?= web-fraud
DTYPE  ?= float32
STAGES ?= 0
dgl-one: need-snap
	cd $(VS_DGL) && env $(DGL_ENV) GRAPHS="$(GRAPH)" DTYPE=$(DTYPE) STAGES=$(STAGES) BACKWARD=1 DIMS=$(DIMS) ./run.sh 2>&1 | tee -a $(LOGS)/dgl_one_$(GRAPH)_$(DTYPE).log

# --- turbo vs turbo ---------------------------------------------------------
# Worktrees of BEFORE and AFTER, measured by this copy of bench.py.  Note that a
# worktree holds the commit's Python: a backward whose new arguments only exist
# in uncommitted _functions.py runs with the defaults there.
before-after:
	$(call run,cd $(BA) && env $(CUDA_ENV) BEFORE_REF=$(BEFORE) AFTER_REF=$(AFTER) GRAPHS="$(BA_GRAPHS)" DTYPES="$(DTYPES)" STAGES="$(STAGES_LIST)" DIMS=$(BA_DIMS) SKIP_BUILD=1 WORKDIR=$(BENCH)/wt GRAPH_CACHE=$(GRAPH_CACHE) OGB_ROOT=$(OGB_ROOT) PYTHON=$(PYTHON_VENV) BENCH_CMD="$(PYTHON_VENV) $(BENCH)/harness/bench_wrapper.py" ./run.sh,before_after_$(STAMP))

# --- reports ---------------------------------------------------------------
# PREV: the archived set to diff against (default: the newest results_prev_*)
PREV ?= $(shell ls -d $(VS_DGL)/results_prev_* 2>/dev/null | sort | tail -1)
SUM_DIM ?= 64
summary:
	cd $(VS_DGL)
	for dtype in $(DTYPES); do
		for st in $(STAGES_LIST); do
			tag="$${dtype}_bwd"; [ "$$st" != 0 ] && tag="$${tag}_st$$st"
			ls results/results_*_$$tag.json >/dev/null 2>&1 || continue
			echo; echo "### $$tag  (d=$(SUM_DIM))"
			$(PYTHON_DGL) summarize.py --tag $$tag --dim $(SUM_DIM) --markdown
			if [ -n "$(PREV)" ]; then
				echo; echo "#### vs $(notdir $(PREV))"
				$(PYTHON_DGL) compare_runs.py --old $(PREV) --new results --tag $$tag --dim $(SUM_DIM)
			fi
		done
	done

# Per-cell charts: one bar per (graph, op, reducer) at d=64, drawn from every
# results_*_<tag>.json, then copied up here as fp32_forward.png etc.
plots:
	cd $(VS_DGL) && env PLOTS_ONLY=1 PYTHON_PLOT=$(PYTHON_PLOT) DTYPES="$(DTYPES)" STAGES_LIST="$(STAGES_LIST)" REPEATS=$(REPEATS) ITERS=$(ITERS) ./run_matrix.sh

# Per-graph charts: one bar per graph, geomean over its cells -- the readable
# view at sixteen graphs.  Published as pergraph_fp32_backward.png etc.
plots-pergraph:
	cd $(VS_DGL)
	n=0
	for dtype in $(DTYPES); do
		case $$dtype in float32) short=fp32;; *) short=fp16;; esac
		for st in $(STAGES_LIST); do
			tag="$${dtype}_bwd"; stem="$$short"
			if [ "$$st" != 0 ]; then tag="$${tag}_st$$st"; stem="$${short}_stages$$st"; fi
			ls results/results_*_$$tag.json >/dev/null 2>&1 || continue
			for kind in fwd bwd fb; do
				case $$kind in fwd) word=forward; label="forward";; bwd) word=backward; label="backward";; *) word=fwd_bwd; label="forward+backward";; esac
				$(PYTHON_PLOT) plot.py results/results_*_$$tag.json --dim $(SUM_DIM) --kind $$kind --per-graph \
					--title "g-SpMM: turbo_gnn против DGL по графам (d=$(SUM_DIM), $$dtype, $$label, stages=$$st)" \
					--xlabel "Ускорение: dgl / turbo_gnn, geomean по клеткам графа ($$label)" \
					-o results/pergraph_d$(SUM_DIM)_$${tag}_$$kind.png
				cp -p results/pergraph_d$(SUM_DIM)_$${tag}_$$kind.png $(HERE)/pergraph_$${stem}_$$word.png; n=$$((n + 1))
			done
		done
	done
	echo "published $$n per-graph charts to $(HERE)"

# One chart with your own selection: make plot-custom ARGS="--kind bwd --ops copy_u --reducers min,max"
ARGS ?=
plot-custom:
	cd $(VS_DGL) && $(PYTHON_PLOT) plot.py results/results_*_$(firstword $(DTYPES))_bwd.json --dim $(SUM_DIM) $(ARGS) -o results/custom.png && echo "-> $(VS_DGL)/results/custom.png"

# --- correctness -----------------------------------------------------------
# The live tree, not the snapshot: this is the gate before taking one.
tests:
	cd $(REPO) && env $(CUDA_ENV) TURBO_REPO=$(REPO) TURBO_EXT_NAME=turbo_live_tests_C \
	$(PYTHON_VENV) -c 'import sys; sys.path.insert(0, "$(BENCH)/harness"); import build_ext; build_ext.install(); import pytest; sys.exit(pytest.main(["tests/correctness/test_gspmm_vs_torch.py", "-q", "-p", "no:cacheprovider"] + sys.argv[1:]))' -- $(PYTEST_ARGS)

# cora/citeseer/pubmed come from dgl.data and the big edge lists need pandas:
# both live only in the DGL interpreter.
NAMES ?=
prepare-graph:
	@test -n "$(NAMES)" || { echo "usage: make prepare-graph NAMES=\"cora pubmed web-fraud\""; exit 1; }
	cd $(VS_DGL) && env OGB_ROOT=$(OGB_ROOT) GRAPHLAND_ROOT=$(GRAPHLAND_ROOT) \
	$(PYTHON_DGL) prepare_graph.py $(NAMES) --out-dir $(GRAPH_CACHE)
