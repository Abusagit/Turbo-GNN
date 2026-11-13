#include "spmm/cusparse_spmm.h"

#include <ATen/cuda/CUDAContext.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/util/intrusive_ptr.h>
#include <cuda_runtime_api.h>
#include <cusparse.h>
#include <torch/extension.h>

#include <algorithm>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>
#include <vector>

// =============================================================================
// Normalized SpMM through cuSPARSE: out = norm(A) @ features (or norm(A)^T @ features), for any
// feature width, where A is the CSR adjacency (rows are destinations) and norm scales each edge
// u -> v by 1 ("none"), 1 / in_deg(v) ("right"), 1 / out_deg(u) ("left") or 1 / sqrt(out_deg(u) *
// in_deg(v)) ("both"), optionally times an edge weight. The degrees and normalized edge values come
// from edge_norm_kernels.cu.
//
// cuSPARSE takes the sparse values in the dtype of the dense operands -- A, B and C of one type,
// accumulating in fp32 -- so the normalized values are computed in fp32 and stored in the features'
// dtype. With A in fp32 and the features in fp16 or bf16, cusparseSpMM rejects the combination.
//
// The cache holds, per graph structure, normalization and dtype, the normalized values and the
// sparse descriptor over them. A graph is identified by its indptr and indices tensors themselves,
// held weakly with their version counters: a cached entry is used only while those very tensors are
// alive and unmodified. A key of sizes and data pointers alone handed a graph allocated at a freed
// graph's address the freed graph's normalization. Edge weights are never cached -- the values
// depend on them, not only on the structure -- so a call with weights normalizes them afresh, an
// O(E) pass beside the O(E * D) product. Nor are graphs whose tensors track no version (created in
// inference mode): an in-place change to them could not be noticed.
// =============================================================================

#define CHECK_CUSPARSE(call)                                                                                                             \
    do {                                                                                                                                 \
        const cusparseStatus_t status_ = (call);                                                                                         \
        TORCH_CHECK(                                                                                                                     \
            status_ == CUSPARSE_STATUS_SUCCESS, "cuSPARSE: ", #call, " failed: ", cusparseGetErrorString(status_), " (",                  \
            static_cast<int>(status_), ")"                                                                                               \
        );                                                                                                                               \
    } while (0)

enum class NormType : uint8_t { NONE = 0, RIGHT = 1, LEFT = 2, BOTH = 3 };

void launch_compute_degrees(
    const torch::Tensor& indptr, const torch::Tensor& indices, torch::Tensor& in_degrees, torch::Tensor& out_degrees, int block_dim
);

void launch_compute_normalized_weights(
    const torch::Tensor& indptr, const torch::Tensor& indices, const torch::Tensor& edge_weights, torch::Tensor& normalized_weights,
    const torch::Tensor& in_degrees, const torch::Tensor& out_degrees, NormType norm, int block_dim
);

namespace {

NormType parse_norm(const std::string& norm) {
    if (norm == "none") return NormType::NONE;
    if (norm == "right") return NormType::RIGHT;
    if (norm == "left") return NormType::LEFT;
    if (norm == "both") return NormType::BOTH;
    TORCH_CHECK(false, "csr_SPMM: invalid normalization '", norm, "'; must be one of 'none', 'right', 'left', 'both'");
    return NormType::NONE;
}

cudaDataType_t value_type(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Float:
            return CUDA_R_32F;
        case c10::ScalarType::Half:
            return CUDA_R_16F;
        case c10::ScalarType::BFloat16:
            return CUDA_R_16BF;
        default:
            TORCH_CHECK(false, "csr_SPMM: features must be float32/float16/bfloat16, got ", dtype);
    }
    return CUDA_R_32F;
}

// The CSR index type. Unsigned indices are read as the signed type of their width, which is exact
// below 2^31 (2^63).
cusparseIndexType_t index_type(c10::ScalarType dtype) {
    switch (dtype) {
        case c10::ScalarType::Int:
        case c10::ScalarType::UInt32:
            return CUSPARSE_INDEX_32I;
        case c10::ScalarType::Long:
        case c10::ScalarType::UInt64:
            return CUSPARSE_INDEX_64I;
        default:
            TORCH_CHECK(false, "csr_SPMM: indptr must be int32, int64, uint32 or uint64, got ", dtype);
    }
    return CUSPARSE_INDEX_32I;
}

cusparseSpMMAlg_t parse_alg(int algorithm) {
    switch (algorithm) {
        case 1:
            return CUSPARSE_SPMM_CSR_ALG1;
        case 2:
            return CUSPARSE_SPMM_CSR_ALG2;
        case 3:
            return CUSPARSE_SPMM_CSR_ALG3;
        default:
            return CUSPARSE_SPMM_ALG_DEFAULT;
    }
}

// CSR_ALG3 supports only the non-transposed product: requested for A^T (the backward of a forward
// that asked for it) it is rejected outright. CSR_ALG2 on A^T is ALG1, per the cuSPARSE docs, so ALG1
// is the transposed form of both.
cusparseSpMMAlg_t alg_for(cusparseSpMMAlg_t alg, cusparseOperation_t op) {
    return op != CUSPARSE_OPERATION_NON_TRANSPOSE && alg == CUSPARSE_SPMM_CSR_ALG3 ? CUSPARSE_SPMM_CSR_ALG1 : alg;
}

struct SpMat {
    cusparseSpMatDescr_t d = nullptr;
    SpMat()                = default;
    SpMat(const SpMat&)    = delete;
    ~SpMat() {
        if (d != nullptr) {
            cusparseDestroySpMat(d);
        }
    }
};

struct DnMat {
    cusparseDnMatDescr_t d = nullptr;
    DnMat()                = default;
    DnMat(const DnMat&)    = delete;
    ~DnMat() {
        if (d != nullptr) {
            cusparseDestroyDnMat(d);
        }
    }
};

void check_graph(const torch::Tensor& indptr, const torch::Tensor& indices, const torch::Tensor& features) {
    TORCH_CHECK(indptr.is_cuda() && indices.is_cuda() && features.is_cuda(), "csr_SPMM: indptr, indices and features must be CUDA tensors");
    TORCH_CHECK(
        indptr.device() == features.device() && indices.device() == features.device(), "csr_SPMM: the CSR must be on the device of the features"
    );
    TORCH_CHECK(indptr.is_contiguous() && indices.is_contiguous(), "csr_SPMM: indptr and indices must be contiguous");
    TORCH_CHECK(indices.scalar_type() == indptr.scalar_type(), "csr_SPMM: indices must have the dtype of indptr");
    TORCH_CHECK(indptr.dim() == 1 && indptr.numel() >= 1 && indices.dim() == 1, "csr_SPMM: indptr must be [N + 1] and indices [E]");
    index_type(indptr.scalar_type());
    TORCH_CHECK(
        indptr.scalar_type() != c10::ScalarType::UInt32 || indices.numel() <= std::numeric_limits<int32_t>::max(),
        "csr_SPMM: uint32 indices support graphs of fewer than 2^31 edges; use int64 indices"
    );
    TORCH_CHECK(features.dim() == 2, "csr_SPMM: features must be 2-D [N, F]");
    TORCH_CHECK(
        indptr.numel() - 1 <= std::numeric_limits<int32_t>::max() && features.size(1) <= std::numeric_limits<int32_t>::max(),
        "csr_SPMM: graphs of up to 2^31 - 1 nodes and features of up to 2^31 - 1 columns"
    );
    TORCH_CHECK(features.size(0) == indptr.numel() - 1, "csr_SPMM: features.size(0) must equal the number of nodes, indptr.numel() - 1");
    value_type(features.scalar_type());
}

// The normalized edge values of a graph, computed in fp32 and stored in dtype.
torch::Tensor normalized_values(
    const torch::Tensor& indptr, const torch::Tensor& indices, const torch::Tensor& edge_weights, NormType norm, c10::ScalarType dtype,
    int block_dim
) {
    const int64_t m        = indptr.numel() - 1;
    const auto f32         = torch::dtype(torch::kFloat32).device(indptr.device());
    torch::Tensor in_deg   = torch::empty({m}, f32);
    torch::Tensor out_deg  = torch::empty({m}, f32);
    torch::Tensor weighted = torch::empty({indices.numel()}, f32);
    launch_compute_degrees(indptr, indices, in_deg, out_deg, block_dim);
    const torch::Tensor w =
        edge_weights.defined() && edge_weights.numel() > 0 ? edge_weights.to(torch::kFloat32).contiguous() : torch::Tensor();
    launch_compute_normalized_weights(indptr, indices, w, weighted, in_deg, out_deg, norm, block_dim);
    return dtype == c10::ScalarType::Float ? weighted : weighted.to(dtype);
}

void create_csr(SpMat& mat, const torch::Tensor& indptr, const torch::Tensor& indices, const torch::Tensor& values) {
    const int64_t m                = indptr.numel() - 1;
    const cusparseIndexType_t type = index_type(indptr.scalar_type());
    CHECK_CUSPARSE(cusparseCreateCsr(
        &mat.d, m, m, indices.numel(), indptr.data_ptr(), indices.data_ptr(), values.data_ptr(), type, type, CUSPARSE_INDEX_BASE_ZERO,
        value_type(values.scalar_type())
    ));
}

// A^T as a CSR of its own: edge u -> v, in row v of A, goes to row u, the rows of A^T kept in A's row
// order (a stable sort by source). For products with A^T: cuSPARSE does not take A^T of 16-bit values
// at all ("SPMM_CSR (mixed precision) is not supported when opA != OPERATION_NON_TRANSPOSE"), and its
// transposed fp32 product scatters: measured over six real graphs, the explicit A^T ran it 1.3-2.1x
// faster at D = 64 and 2.1-3.1x at D = 128-256 (reddit, D = 256: 67 ms -> 22), and 0.77-1.03x at
// D = 16, where the transposed product is already near the forward's time. So fp32 products with A^T
// keep the transposed form below 32 columns.
inline constexpr int64_t kExplicitTransposeMinColumns = 32;
struct TransposedCsr {
    torch::Tensor indptr;
    torch::Tensor indices;
    torch::Tensor values;
    SpMat mat;
};

void build_transposed(TransposedCsr& t, const torch::Tensor& indptr, const torch::Tensor& indices, const torch::Tensor& values) {
    const auto as_signed = [](const torch::Tensor& x) {
        return x.scalar_type() == c10::ScalarType::UInt32   ? x.view(torch::kInt)
               : x.scalar_type() == c10::ScalarType::UInt64 ? x.view(torch::kLong)
                                                            : x;
    };
    const torch::Tensor ptr  = as_signed(indptr).to(torch::kLong);
    const torch::Tensor src  = as_signed(indices).to(torch::kLong);
    const int64_t m          = ptr.numel() - 1;
    const torch::Tensor dst  = torch::repeat_interleave(torch::arange(m, ptr.options()), ptr.slice(0, 1) - ptr.slice(0, 0, m));
    const torch::Tensor perm = std::get<1>(src.sort(/*stable=*/std::optional<bool>(true), /*dim=*/0));
    const auto index_dtype   = index_type(indptr.scalar_type()) == CUSPARSE_INDEX_32I ? torch::kInt : torch::kLong;
    t.indices                = dst.index_select(0, perm).to(index_dtype);
    t.values                 = values.index_select(0, perm);
    t.indptr                 = torch::zeros({m + 1}, ptr.options());
    t.indptr.slice(0, 1).copy_(torch::bincount(src, {}, m).cumsum(0));
    t.indptr = t.indptr.to(index_dtype);
    create_csr(t.mat, t.indptr, t.indices, t.values);
}

// ---- The cache ----

using WeakImpl = c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl>;

struct CacheKey {
    void const *indptr;
    void const *indices;
    int64_t nodes;
    int64_t edges;
    NormType norm;
    c10::ScalarType dtype;

    bool operator<(const CacheKey& o) const {
        return std::tie(indptr, indices, nodes, edges, norm, dtype) < std::tie(o.indptr, o.indices, o.nodes, o.edges, o.norm, o.dtype);
    }
};

struct CacheEntry {
    WeakImpl indptr;
    WeakImpl indices;
    uint32_t indptr_version;
    uint32_t indices_version;
    torch::Tensor values;  // normalized edge values, in the features' dtype
    SpMat mat;
    std::unique_ptr<TransposedCsr> transposed;          // A^T, built on the first product with it
    cusparseSpMMAlg_t alg = CUSPARSE_SPMM_ALG_DEFAULT;  // the algorithm of the call that created the entry

    // The entry's tensors are written on the stream current at their creation and read by later calls
    // on whatever stream is current then. ready: recorded after each write, waited on by every read.
    // users: the streams that have read them. They are freed when the entry is replaced, swept or
    // cleared -- at a point no reading stream is synchronized with -- so each reading stream is
    // recorded on them (record_stream), which keeps the caching allocator from handing their memory
    // out while such a read may still be in flight.
    at::cuda::CUDAEvent ready;
    std::vector<c10::StreamId> users;

    void published(const at::cuda::CUDAStream& stream) { ready.record(stream); }

    void used_on(const at::cuda::CUDAStream& stream) {
        ready.block(stream);
        if (std::find(users.begin(), users.end(), stream.id()) != users.end()) {
            return;
        }
        users.push_back(stream.id());
        const torch::Tensor *tensors[] = {&values, transposed ? &transposed->indptr : nullptr, transposed ? &transposed->indices : nullptr,
            transposed ? &transposed->values : nullptr};
        for (const torch::Tensor *t : tensors) {
            if (t != nullptr && t->defined()) {
                t->record_stream(stream);
            }
        }
    }

    CacheEntry(const torch::Tensor& ip, const torch::Tensor& ix)
        : indptr(ip.getIntrusivePtr()),
          indices(ix.getIntrusivePtr()),
          indptr_version(ip.unsafeGetTensorImpl()->version_counter().current_version()),
          indices_version(ix.unsafeGetTensorImpl()->version_counter().current_version()) {}

    // Whether the entry was built from these very tensors, unmodified since.
    bool describes(const torch::Tensor& ip, const torch::Tensor& ix) const {
        return !indptr.expired() && !indices.expired() && indptr.lock().get() == ip.unsafeGetTensorImpl() &&
               indices.lock().get() == ix.unsafeGetTensorImpl() &&
               indptr_version == ip.unsafeGetTensorImpl()->version_counter().current_version() &&
               indices_version == ix.unsafeGetTensorImpl()->version_counter().current_version();
    }
    bool dead() const { return indptr.expired() || indices.expired(); }
};

// Never destroyed: its tensors and descriptors must not be freed during static teardown, after the
// CUDA runtime may already be gone.
auto& graph_cache = *new std::map<CacheKey, std::unique_ptr<CacheEntry>>();
std::mutex cache_mutex;

// Whether a graph's tensors can key the cache: they must track in-place changes.
bool cacheable(const torch::Tensor& indptr, const torch::Tensor& indices) { return !indptr.is_inference() && !indices.is_inference(); }

// The cached entry for a graph, normalization and dtype -- built, or rebuilt if the key now names a
// different graph than the one it was built from. Holds cache_mutex while it works.
CacheEntry& cached_entry(
    const torch::Tensor& indptr, const torch::Tensor& indices, NormType norm, c10::ScalarType dtype, cusparseSpMMAlg_t alg, int block_dim
) {
    // Entries whose graphs are gone can never match again: dropped on every call, so that they do not
    // hold their O(E) of values until some later miss.
    for (auto e = graph_cache.begin(); e != graph_cache.end();) {
        e = e->second->dead() ? graph_cache.erase(e) : std::next(e);
    }
    const CacheKey key{indptr.data_ptr(), indices.data_ptr(), indptr.numel() - 1, indices.numel(), norm, dtype};
    auto it = graph_cache.find(key);
    if (it != graph_cache.end() && it->second->describes(indptr, indices)) {
        return *it->second;
    }
    auto entry    = std::make_unique<CacheEntry>(indptr, indices);
    entry->values = normalized_values(indptr, indices, torch::Tensor(), norm, dtype, block_dim);
    entry->alg    = alg;
    create_csr(entry->mat, indptr, indices, entry->values);
    entry->published(at::cuda::getCurrentCUDAStream());
    CacheEntry& ref  = *entry;
    graph_cache[key] = std::move(entry);
    return ref;
}

// out = op(A) @ B with A the sparse descriptor, on the current stream.
void spmm(cusparseSpMatDescr_t A, const torch::Tensor& B, torch::Tensor& C, cusparseOperation_t op, cusparseSpMMAlg_t alg) {
    const int64_t m   = B.size(0);
    const int64_t n   = B.size(1);
    const auto type   = value_type(B.scalar_type());
    auto handle       = at::cuda::getCurrentCUDASparseHandle();
    const float alpha = 1.0f;
    const float beta  = 0.0f;
    DnMat matB, matC;
    CHECK_CUSPARSE(cusparseCreateDnMat(&matB.d, m, n, n, B.data_ptr(), type, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matC.d, m, n, n, C.data_ptr(), type, CUSPARSE_ORDER_ROW));
    alg = alg_for(alg, op);
    // Sized for this very call -- the buffer depends on the dense shapes and the algorithm -- and taken
    // from the stream-ordered caching allocator, so it is neither a device-wide sync nor too small for
    // a later call of another width.
    size_t bytes = 0;
    CHECK_CUSPARSE(
        cusparseSpMM_bufferSize(handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, matB.d, &beta, matC.d, CUDA_R_32F, alg, &bytes)
    );
    torch::Tensor workspace = torch::empty({static_cast<int64_t>(bytes)}, B.options().dtype(torch::kUInt8));
    CHECK_CUSPARSE(cusparseSpMM(
        handle, op, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, A, matB.d, &beta, matC.d, CUDA_R_32F, alg,
        bytes > 0 ? workspace.data_ptr() : nullptr
    ));
}

}  // namespace

torch::Tensor csr_SPMM_normalized(
    const torch::Tensor& indptr,
    const torch::Tensor& indices,
    const torch::Tensor& features_in,
    const torch::Tensor& edge_weights,
    const std::string& norm_str,
    int algorithm,
    bool use_cache,
    bool do_transpose_a,
    int block_dim
) {
    check_graph(indptr, indices, features_in);
    const at::cuda::CUDAGuard device_guard(features_in.device());
    const torch::Tensor features = features_in.contiguous();
    const NormType norm          = parse_norm(norm_str);
    const bool has_edge_weights  = edge_weights.defined() && edge_weights.numel() > 0;
    if (has_edge_weights) {
        TORCH_CHECK(
            edge_weights.is_cuda() && edge_weights.device() == features.device(), "csr_SPMM: edge_weights must be on the device of the features"
        );
        TORCH_CHECK(edge_weights.numel() == indices.numel(), "csr_SPMM: edge weights must have one entry per edge");
    }

    const int64_t m = indptr.numel() - 1;
    const int64_t n = features.size(1);
    if (m == 0 || n == 0 || indices.numel() == 0) {
        return torch::zeros({m, n}, features.options());
    }
    torch::Tensor out            = torch::empty({m, n}, features.options());
    const cusparseOperation_t op = do_transpose_a ? CUSPARSE_OPERATION_TRANSPOSE : CUSPARSE_OPERATION_NON_TRANSPOSE;
    const cusparseSpMMAlg_t alg  = parse_alg(algorithm);

    if (use_cache && !has_edge_weights && cacheable(indptr, indices)) {
        std::lock_guard<std::mutex> lock(cache_mutex);
        const at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        CacheEntry& entry                 = cached_entry(indptr, indices, norm, features.scalar_type(), alg, block_dim);
        const cusparseSpMMAlg_t used_alg  = algorithm == -1 ? entry.alg : alg;  // -1: the algorithm the entry was made with
        const bool explicit_t = do_transpose_a && (features.scalar_type() != c10::ScalarType::Float || n >= kExplicitTransposeMinColumns);
        if (explicit_t && !entry.transposed) {
            entry.ready.block(stream);  // the transpose reads values
            entry.transposed = std::make_unique<TransposedCsr>();
            build_transposed(*entry.transposed, indptr, indices, entry.values);
            entry.published(stream);
            entry.users.clear();  // the new tensors are recorded on every reading stream afresh
        }
        entry.used_on(stream);
        if (explicit_t) {
            spmm(entry.transposed->mat.d, features, out, CUSPARSE_OPERATION_NON_TRANSPOSE, used_alg);
        } else {
            spmm(entry.mat.d, features, out, op, used_alg);
        }
        return out;
    }

    const torch::Tensor values =
        normalized_values(indptr, indices, has_edge_weights ? edge_weights : torch::Tensor(), norm, features.scalar_type(), block_dim);
    if (do_transpose_a && features.scalar_type() != c10::ScalarType::Float) {
        // 16-bit A^T has no transposed product (see TransposedCsr). Uncached, fp32 keeps it: the
        // transpose's sort would cost more than it saves on one product.
        TransposedCsr t;
        build_transposed(t, indptr, indices, values);
        spmm(t.mat.d, features, out, CUSPARSE_OPERATION_NON_TRANSPOSE, alg);
        return out;
    }
    SpMat mat;
    create_csr(mat, indptr, indices, values);
    spmm(mat.d, features, out, op, alg);
    return out;
}

int find_best_algorithm_normalized(
    const torch::Tensor& indptr,
    const torch::Tensor& indices,
    const torch::Tensor& features_in,
    const torch::Tensor& edge_weights,
    const std::string& norm_str,
    int block_dim
) {
    check_graph(indptr, indices, features_in);
    const at::cuda::CUDAGuard device_guard(features_in.device());
    const torch::Tensor features = features_in.contiguous();
    const NormType norm          = parse_norm(norm_str);
    const bool has_edge_weights  = edge_weights.defined() && edge_weights.numel() > 0;
    if (features.size(0) == 0 || features.size(1) == 0 || indices.numel() == 0) {
        return 0;
    }

    const torch::Tensor values =
        normalized_values(indptr, indices, has_edge_weights ? edge_weights : torch::Tensor(), norm, features.scalar_type(), block_dim);
    SpMat mat;
    create_csr(mat, indptr, indices, values);
    torch::Tensor out = torch::empty_like(features);

    const std::vector<std::pair<int, cusparseSpMMAlg_t>> algorithms = {
        {0, CUSPARSE_SPMM_ALG_DEFAULT}, {1, CUSPARSE_SPMM_CSR_ALG1}, {2, CUSPARSE_SPMM_CSR_ALG2}, {3, CUSPARSE_SPMM_CSR_ALG3}};
    const auto stream = at::cuda::getCurrentCUDAStream();
    int best_alg_id   = 0;
    float best_time   = std::numeric_limits<float>::max();
    for (auto& [alg_id, alg] : algorithms) {
        try {
            for (int i = 0; i < 3; ++i) {
                spmm(mat.d, features, out, CUSPARSE_OPERATION_NON_TRANSPOSE, alg);  // warmup
            }
            // Timed on the stream the products run on.
            at::cuda::CUDAEvent start(cudaEventDefault), stop(cudaEventDefault);
            start.record(stream);
            for (int i = 0; i < 10; ++i) {
                spmm(mat.d, features, out, CUSPARSE_OPERATION_NON_TRANSPOSE, alg);
            }
            stop.record(stream);
            stop.synchronize();
            const float ms = start.elapsed_time(stop);
            if (ms < best_time) {
                best_time   = ms;
                best_alg_id = alg_id;
            }
        } catch (const c10::Error&) {
            // Not supported for this matrix and dtype: skip it.
        }
    }
    return best_alg_id;
}

// ************************* BACKWARD COMPATIBILITY STARTS *************************
int find_best_algorithm(const torch::Tensor& indptr, const torch::Tensor& indices, const torch::Tensor& features, int block_dim) {
    return find_best_algorithm_normalized(indptr, indices, features, torch::Tensor(), "none", block_dim);
}

torch::Tensor csr_SPMM(
    const torch::Tensor& indptr,
    const torch::Tensor& indices,
    const torch::Tensor& features,
    int algorithm,
    bool use_cache,
    bool do_transpose_a,
    int block_dim
) {
    return csr_SPMM_normalized(indptr, indices, features, torch::Tensor(), "none", algorithm, use_cache, do_transpose_a, block_dim);
}
// ************************* BACKWARD COMPATIBILITY ENDS *************************

void clear_graph_cache() {
    std::lock_guard<std::mutex> lock(cache_mutex);
    graph_cache.clear();
}
