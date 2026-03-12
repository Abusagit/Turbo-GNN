"""Integration tests for autotuning of real CUDA kernel APIs.

Exercises reduction_aggr, gatv2_aggr, and graph_transformer_aggr
with autotune=True on actual CUDA kernels, verifying correctness,
caching, and singleton isolation.
"""

import pytest
import torch

from src.backends.base import AutotuneConfig, TunableKernel, _InlineAutotuneCache
from src.backends.cuda_backend.gatv2_aggr.utils import GATv2AggrKernel, gatv2_aggr
from src.backends.cuda_backend.gt_aggr.utils import GraphTransformerAggrKernel, graph_transformer_aggr
from src.backends.cuda_backend.reduction_aggr.utils import ReductionAggrKernel, reduction_aggr
from src.data.converters import AdjacencyForwardBackwardWithNodeBuckets

# Minimal timing config: 1 warmup + 1 timed iteration per candidate
FAST_CONFIG = AutotuneConfig(warmup=1, iters=1)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _create_csr_graph(device, N=100, E=500, index_dtype=torch.int32):
    """Random CSR graph as (indptr, indices) tensors with given index dtype."""
    src = torch.randint(0, N, (E,), device=device, dtype=torch.int64)
    dst = torch.randint(0, N, (E,), device=device, dtype=torch.int64)

    indptr = torch.zeros(N + 1, device=device, dtype=torch.int64)
    for i in range(E):
        indptr[dst[i].item() + 1] += 1
    indptr = torch.cumsum(indptr, dim=0)

    sorted_idx = torch.argsort(dst)
    indices = src[sorted_idx]
    return indptr.to(index_dtype), indices.to(index_dtype)


def _partition_nodes(indptr, threshold=100):
    deg = indptr[1:] - indptr[:-1]
    index_dtype = indptr.dtype
    light = torch.nonzero(deg <= threshold, as_tuple=True)[0].to(index_dtype).to(indptr.device)
    heavy = torch.nonzero(deg > threshold, as_tuple=True)[0].to(index_dtype).to(indptr.device)
    return light, heavy


def _make_graph(device, N=100, E=500, threshold=100, index_dtype=torch.int32):
    """Build AdjacencyForwardBackwardWithNodeBuckets from a random graph."""
    indptr, indices = _create_csr_graph(device, N, E, index_dtype=index_dtype)
    light, heavy = _partition_nodes(indptr, threshold)
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=indptr,
        forward_indices=indices,
        backward_indptr=indptr,
        backward_indices=indices,
        forward_light_nodes=light,
        forward_heavy_nodes=heavy,
        backward_light_nodes=light,
        backward_heavy_nodes=heavy,
    )


def zero_inf(x):
    return torch.where(torch.isinf(x), torch.zeros_like(x), x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def clear_kernel_singletons():
    """Clear TunableKernel singleton registry before and after each test."""
    TunableKernel._shared_instances.clear()
    yield
    TunableKernel._shared_instances.clear()


def _skip_if_no_cuda():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")


# ---------------------------------------------------------------------------
# TestReductionAggrAutotune
# ---------------------------------------------------------------------------


class TestReductionAggrAutotune:
    @pytest.mark.parametrize("reduce", ["min", "max"])
    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
    def test_autotune_matches_default(self, reduce, index_dtype):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph = _make_graph(device, N=100, E=500, index_dtype=index_dtype)
        x = torch.randn(100, 64, device=device, dtype=torch.float32)

        out_default = zero_inf(reduction_aggr(graph, x, reduce=reduce))
        out_tuned = zero_inf(reduction_aggr(graph, x, reduce=reduce, autotune=True, autotune_config=FAST_CONFIG))

        torch.testing.assert_close(
            out_tuned,
            out_default,
            atol=1e-5,
            rtol=1e-4,
            msg=f"Autotuned output differs from default for reduce={reduce}",
        )

    def test_caching_avoids_retuning(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph = _make_graph(device, N=100, E=500)
        x = torch.randn(100, 64, device=device, dtype=torch.float32)

        # First call: autotunes
        reduction_aggr(graph, x, reduce="min", autotune=True, autotune_config=FAST_CONFIG)

        # Retrieve the kernel singleton to inspect its cache
        kernel = ReductionAggrKernel._get_or_create(reduce="min")
        feat_dim = x.shape[-1]
        cached = kernel._inline_cache.lookup(graph, feat_dim)
        assert cached is not None, "Cache should be populated after first autotuned call"

        # Monkeypatch _inline_autotune to detect re-tuning
        call_count = [0]
        orig = kernel._inline_autotune

        def counting_autotune(*args, **kwargs):
            call_count[0] += 1
            return orig(*args, **kwargs)

        kernel._inline_autotune = counting_autotune
        try:
            # Second call should hit cache
            reduction_aggr(graph, x, reduce="min", autotune=True, autotune_config=FAST_CONFIG)
            assert call_count[0] == 0, "Second call should use cache, not re-autotune"
        finally:
            kernel._inline_autotune = orig

    def test_different_feat_dims_separate_cache(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph = _make_graph(device, N=100, E=500)
        x32 = torch.randn(100, 32, device=device, dtype=torch.float32)
        x64 = torch.randn(100, 64, device=device, dtype=torch.float32)

        reduction_aggr(graph, x32, reduce="min", autotune=True, autotune_config=FAST_CONFIG)
        reduction_aggr(graph, x64, reduce="min", autotune=True, autotune_config=FAST_CONFIG)

        kernel = ReductionAggrKernel._get_or_create(reduce="min")
        assert kernel._inline_cache.lookup(graph, 32) is not None
        assert kernel._inline_cache.lookup(graph, 64) is not None

    @pytest.mark.parametrize("index_dtype", [torch.int32, torch.int64])
    def test_backward_with_autotune(self, index_dtype):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph = _make_graph(device, N=100, E=500, index_dtype=index_dtype)
        x = torch.randn(100, 64, device=device, dtype=torch.float32, requires_grad=True)

        out = reduction_aggr(graph, x, reduce="min", autotune=True, autotune_config=FAST_CONFIG)
        out = zero_inf(out)

        grad_out = torch.ones_like(out)
        out.backward(grad_out)

        assert x.grad is not None, "Gradient should be computed"
        assert x.grad.shape == x.shape, "Gradient shape mismatch"
        assert not torch.isnan(x.grad).any(), "Gradient contains NaN"


# ---------------------------------------------------------------------------
# TestGATv2AggrAutotune
# ---------------------------------------------------------------------------


class TestGATv2AggrAutotune:
    def test_autotune_matches_default(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph = _make_graph(device, N, E)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        x_neighbors = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        attention_weights = torch.randn(heads, head_dim, device=device, dtype=torch.float32)
        negative_slope = 0.2

        out_default = gatv2_aggr(graph, x, x_neighbors, attention_weights, negative_slope)
        out_tuned = gatv2_aggr(
            graph,
            x,
            x_neighbors,
            attention_weights,
            negative_slope,
            autotune=True,
            autotune_config=FAST_CONFIG,
        )

        torch.testing.assert_close(
            out_tuned,
            out_default,
            atol=1e-5,
            rtol=1e-4,
            msg="Autotuned GATv2 output differs from default",
        )

    def test_caching_avoids_retuning(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph = _make_graph(device, N, E)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        x_neighbors = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        attention_weights = torch.randn(heads, head_dim, device=device, dtype=torch.float32)
        negative_slope = 0.2

        gatv2_aggr(
            graph,
            x,
            x_neighbors,
            attention_weights,
            negative_slope,
            autotune=True,
            autotune_config=FAST_CONFIG,
        )

        kernel = GATv2AggrKernel._get_or_create()
        feat_dim = x.shape[-1]
        cached = kernel._inline_cache.lookup(graph, feat_dim)
        assert cached is not None, "GATv2 cache should be populated after first autotuned call"

        call_count = [0]
        orig = kernel._inline_autotune

        def counting_autotune(*args, **kwargs):
            call_count[0] += 1
            return orig(*args, **kwargs)

        kernel._inline_autotune = counting_autotune
        try:
            gatv2_aggr(
                graph,
                x,
                x_neighbors,
                attention_weights,
                negative_slope,
                autotune=True,
                autotune_config=FAST_CONFIG,
            )
            assert call_count[0] == 0, "Second GATv2 call should use cache"
        finally:
            kernel._inline_autotune = orig


# ---------------------------------------------------------------------------
# TestGraphTransformerAggrAutotune
# ---------------------------------------------------------------------------


class TestGraphTransformerAggrAutotune:
    def test_autotune_matches_default(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph = _make_graph(device, N, E)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        Q = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        scale = 1.0 / (head_dim**0.5)

        out_default = graph_transformer_aggr(graph, x, Q, K, V, scale)
        out_tuned = graph_transformer_aggr(
            graph,
            x,
            Q,
            K,
            V,
            scale,
            autotune=True,
            autotune_config=FAST_CONFIG,
        )

        torch.testing.assert_close(
            out_tuned,
            out_default,
            atol=1e-5,
            rtol=1e-4,
            msg="Autotuned GT output differs from default",
        )

    def test_caching_avoids_retuning(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph = _make_graph(device, N, E)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        Q = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        scale = 1.0 / (head_dim**0.5)

        graph_transformer_aggr(
            graph,
            x,
            Q,
            K,
            V,
            scale,
            autotune=True,
            autotune_config=FAST_CONFIG,
        )

        kernel = GraphTransformerAggrKernel._get_or_create()
        feat_dim = x.shape[-1]
        cached = kernel._inline_cache.lookup(graph, feat_dim)
        assert cached is not None, "GT cache should be populated after first autotuned call"

        call_count = [0]
        orig = kernel._inline_autotune

        def counting_autotune(*args, **kwargs):
            call_count[0] += 1
            return orig(*args, **kwargs)

        kernel._inline_autotune = counting_autotune
        try:
            graph_transformer_aggr(
                graph,
                x,
                Q,
                K,
                V,
                scale,
                autotune=True,
                autotune_config=FAST_CONFIG,
            )
            assert call_count[0] == 0, "Second GT call should use cache"
        finally:
            kernel._inline_autotune = orig


# ---------------------------------------------------------------------------
# TestKernelDirectAutotune
# ---------------------------------------------------------------------------


class TestKernelDirectAutotune:
    def test_reduction_kernel_direct_call(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph = _make_graph(device, N=100, E=500)
        x = torch.randn(100, 64, device=device, dtype=torch.float32)

        kernel = ReductionAggrKernel(reduce="min")
        out = kernel(graph, x, autotune=True, autotune_config=FAST_CONFIG)
        out = zero_inf(out)

        assert out.shape == (100, 64)
        assert not torch.isnan(out).any()

    def test_gatv2_kernel_direct_call(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph = _make_graph(device, N, E)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        x_neighbors = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        attention_weights = torch.randn(heads, head_dim, device=device, dtype=torch.float32)
        negative_slope = 0.2

        kernel = GATv2AggrKernel()
        out = kernel(
            graph,
            x,
            autotune=True,
            autotune_config=FAST_CONFIG,
            x_neighbors=x_neighbors,
            attention_weights=attention_weights,
            negative_slope=negative_slope,
        )

        assert out.shape == (N, heads, head_dim)
        assert not torch.isnan(out).any()

    def test_gt_kernel_direct_call(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph = _make_graph(device, N, E)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        Q = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        scale = 1.0 / (head_dim**0.5)

        kernel = GraphTransformerAggrKernel()
        out = kernel(
            graph,
            x,
            autotune=True,
            autotune_config=FAST_CONFIG,
            Q=Q,
            K=K,
            V=V,
            scale=scale,
        )

        assert out.shape == (N, heads, head_dim)
        assert not torch.isnan(out).any()


# ---------------------------------------------------------------------------
# TestUnsignedIndexTypes
# ---------------------------------------------------------------------------


def _reinterpret_graph_unsigned(graph, unsigned_dtype):
    """Reinterpret all index tensors in graph as unsigned (zero-copy .view())."""
    return AdjacencyForwardBackwardWithNodeBuckets(
        forward_indptr=graph.forward_indptr.view(unsigned_dtype),
        forward_indices=graph.forward_indices.view(unsigned_dtype),
        backward_indptr=graph.backward_indptr.view(unsigned_dtype),
        backward_indices=graph.backward_indices.view(unsigned_dtype),
        forward_light_nodes=graph.forward_light_nodes.view(unsigned_dtype),
        forward_heavy_nodes=graph.forward_heavy_nodes.view(unsigned_dtype),
        backward_light_nodes=graph.backward_light_nodes.view(unsigned_dtype),
        backward_heavy_nodes=graph.backward_heavy_nodes.view(unsigned_dtype),
    )


class TestUnsignedIndexTypes:
    @pytest.mark.parametrize("reduce", ["min", "max"])
    def test_reduction_uint32(self, reduce):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph_i32 = _make_graph(device, N=100, E=500, index_dtype=torch.int32)
        graph_u32 = _reinterpret_graph_unsigned(graph_i32, torch.uint32)
        x = torch.randn(100, 64, device=device, dtype=torch.float32)

        out_signed = zero_inf(reduction_aggr(graph_i32, x, reduce=reduce))
        out_unsigned = zero_inf(reduction_aggr(graph_u32, x, reduce=reduce))

        torch.testing.assert_close(
            out_unsigned,
            out_signed,
            atol=1e-5,
            rtol=1e-4,
            msg=f"uint32 reduction output differs from int32 for reduce={reduce}",
        )

    def test_gatv2_uint32(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph_i32 = _make_graph(device, N, E, index_dtype=torch.int32)
        graph_u32 = _reinterpret_graph_unsigned(graph_i32, torch.uint32)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        x_neighbors = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        attention_weights = torch.randn(heads, head_dim, device=device, dtype=torch.float32)
        negative_slope = 0.2

        out_signed = gatv2_aggr(graph_i32, x, x_neighbors, attention_weights, negative_slope)
        out_unsigned = gatv2_aggr(graph_u32, x, x_neighbors, attention_weights, negative_slope)

        torch.testing.assert_close(
            out_unsigned,
            out_signed,
            atol=1e-5,
            rtol=1e-4,
            msg="uint32 GATv2 output differs from int32",
        )

    def test_gt_uint32(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        N, E = 100, 500
        heads, head_dim = 4, 32
        graph_i32 = _make_graph(device, N, E, index_dtype=torch.int32)
        graph_u32 = _reinterpret_graph_unsigned(graph_i32, torch.uint32)

        x = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        Q = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        K = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        V = torch.randn(N, heads, head_dim, device=device, dtype=torch.float32)
        scale = 1.0 / (head_dim**0.5)

        out_signed = graph_transformer_aggr(graph_i32, x, Q, K, V, scale)
        out_unsigned = graph_transformer_aggr(graph_u32, x, Q, K, V, scale)

        torch.testing.assert_close(
            out_unsigned,
            out_signed,
            atol=1e-5,
            rtol=1e-4,
            msg="uint32 GT output differs from int32",
        )


# ---------------------------------------------------------------------------
# TestSingletonAndCacheIsolation
# ---------------------------------------------------------------------------


class TestSingletonAndCacheIsolation:
    def test_same_init_kwargs_reuses_instance(self):
        a = ReductionAggrKernel._get_or_create(reduce="min")
        b = ReductionAggrKernel._get_or_create(reduce="min")
        assert a is b, "Same init kwargs should return the same singleton"

    def test_different_init_kwargs_different_instances(self):
        a = ReductionAggrKernel._get_or_create(reduce="min")
        b = ReductionAggrKernel._get_or_create(reduce="max")
        assert a is not b, "Different init kwargs should return different singletons"

    def test_per_kernel_separate_caches(self):
        _skip_if_no_cuda()
        device = torch.device("cuda")
        torch.manual_seed(42)

        graph = _make_graph(device, N=100, E=500)
        x = torch.randn(100, 64, device=device, dtype=torch.float32)

        # Autotune only with reduce="min"
        reduction_aggr(graph, x, reduce="min", autotune=True, autotune_config=FAST_CONFIG)

        kernel_min = ReductionAggrKernel._get_or_create(reduce="min")
        kernel_max = ReductionAggrKernel._get_or_create(reduce="max")

        feat_dim = x.shape[-1]
        assert kernel_min._inline_cache.lookup(graph, feat_dim) is not None, "min kernel cache should be populated"
        assert (
            kernel_max._inline_cache.lookup(graph, feat_dim) is None
        ), "max kernel cache should be empty (not autotuned)"
