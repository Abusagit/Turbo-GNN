# Rebuttal Table 7 image: PR #61 (Optuna TPE autotuner, origin/optuna_autotune e4ac6c2) code and
# kernels on top of the rebuttal image's two Python environments (turbo-ours, turbo-dgl).
FROM abusagit/turbo-gnn-rebuttal:1.2
SHELL ["/bin/bash", "-o", "pipefail", "-c"]
WORKDIR /opt/Turbo-GNN
RUN uv pip install --python /opt/envs/turbo-ours/bin/python "optuna==5.0.0" && rm -rf /tmp/uv-cache \
    && rm -rf csrc turbo_gnn src scripts configs benchmarks tests build dev *.egg-info
COPY pyproject.toml setup.py MANIFEST.in README.md ./
COPY csrc/ csrc/
COPY turbo_gnn/ turbo_gnn/
RUN TORCH_CUDA_ARCH_LIST="8.0;9.0" TURBO_GNN_FORCE_BUILD=TRUE MAX_JOBS=${MAX_JOBS:-32} \
       uv pip install --python /opt/envs/turbo-ours/bin/python -e . --no-build-isolation \
    && rm -rf build /tmp/uv-cache \
    && /opt/envs/turbo-ours/bin/python -c "import turbo_gnn._C, optuna"
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
COPY benchmarks/ benchmarks/
COPY tests/ tests/
COPY docker/rebuttal_entry.sh docker/serve_results.py docker/
RUN chmod +x docker/rebuttal_entry.sh benchmarks/rebuttal/autotune/*.sh
ARG GIT_COMMIT=unknown
RUN echo "$GIT_COMMIT" > GIT_COMMIT
CMD ["/opt/Turbo-GNN/docker/rebuttal_entry.sh"]
