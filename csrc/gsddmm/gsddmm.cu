#include <cstdint>

#include "common/misc.cuh"
#include "gsddmm/gsddmm.cuh"

namespace gsddmm {

// =============================================================================
// Elementwise GSDDMM op on vector tiles. Dot is reduced separately (it is not
// elementwise); Copy never touches the right operand.
// =============================================================================
template <GSDDMM_OP op, size_t TW, FloatingNum cuda_t>
struct GsddmmVecOp {
    using vec_t = VecFloat<TW, cuda_t>;

    static constexpr __device__ __forceinline__ vec_t apply(vec_t l, vec_t r) {
        if constexpr (op == GSDDMM_OP::Add) {
            l.add_(r);
        } else if constexpr (op == GSDDMM_OP::Sub) {
            l.sub_(r);
        } else if constexpr (op == GSDDMM_OP::Mul) {
            l.mul_(r);
        } else if constexpr (op == GSDDMM_OP::Div) {
            l.div_(r);
        } else if constexpr (op == GSDDMM_OP::Copy) {
            // left operand passes through unchanged
        } else if constexpr (op == GSDDMM_OP::Dot) {
            // operators are unchanged there
        } else {
            __builtin_unreachable();
        }
        return l;
    }
};

// =============================================================================
// cp.async pipeline over the edge list of one CSR row: warp-strided loop with
// NUM_STAGES-deep prefetch of NUM_ROWS per-edge rows (Src_V rows indexed by the
// neighbor id j = col_idx[e], Edge rows indexed by the edge position e).
// Modeled on pipelined_neighbor_row_loop (pipeline.cuh); the addressing differs
// (edge-indexed rows cannot be expressed through col_idx), which is why this
// variant exists.
//
// consume(e, rows): rows[r] is operand r's prefetched row in shared memory;
// valid only inside the call (the slot is recycled on return).
//
// dbuf: this warp's private shared scratch, NUM_ROWS * NUM_STAGES * D_CONST elements.
// =============================================================================
template <size_t N_PER_BLOCK, size_t D_CONST, size_t NUM_STAGES, size_t NUM_ROWS, FloatingNum cuda_t, typename index_t, typename ConsumeFn>
__device__ __forceinline__ void gsddmm_pipelined_edge_loop(
    size_t warp_id,
    size_t lane,
    size_t num_edges,
    index_t edge_start,
    index_t const *__restrict__ col_idx,
    cuda_t const *__restrict__ const (&row_bases)[NUM_ROWS],
    bool const (&row_edge_indexed)[NUM_ROWS],
    cuda_t *__restrict__ dbuf,
    ConsumeFn&& consume
) {
    const size_t loop_iters = (num_edges > warp_id) ? ceil_div(num_edges - warp_id, N_PER_BLOCK) : 0;
    if (loop_iters == 0) {
        return;
    }

    cuda_t *rows[NUM_ROWS][NUM_STAGES];
#pragma unroll
    for (size_t r = 0; r < NUM_ROWS; ++r) {
#pragma unroll
        for (size_t s = 0; s < NUM_STAGES; ++s) {
            rows[r][s] = dbuf + (r * NUM_STAGES + s) * D_CONST;
        }
    }

    index_t edge_pos_buf[NUM_STAGES];

    cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

    auto prefetch = [&](size_t it) {
        pipe.producer_acquire();
        if (it < loop_iters) {
            const size_t k                = warp_id + it * N_PER_BLOCK;
            const index_t e               = edge_start + static_cast<index_t>(k);
            edge_pos_buf[it % NUM_STAGES] = e;
            const index_t j               = col_idx[e];
#pragma unroll
            for (size_t r = 0; r < NUM_ROWS; ++r) {
                const size_t row_id = row_edge_indexed[r] ? static_cast<size_t>(e) : static_cast<size_t>(j);
                async_copy_row_warp<D_CONST, cuda_t>(rows[r][it % NUM_STAGES], row_bases[r] + row_id * D_CONST, pipe, lane);
            }
        }
        pipe.producer_commit();
    };

#pragma unroll
    for (size_t s = 0; s < NUM_STAGES; ++s) {
        prefetch(s);
    }

    for (size_t iter = 0; iter < loop_iters; ++iter) {
        cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
        // Thread-scope wait covers only this lane's own cp.async; a lane may read
        // chunks copied by other lanes (tile < 16B, or row < 512B leaves lanes idle).
        __syncwarp();

        const size_t slot = iter % NUM_STAGES;
        cuda_t const *cur_rows[NUM_ROWS];
#pragma unroll
        for (size_t r = 0; r < NUM_ROWS; ++r) {
            cur_rows[r] = rows[r][slot];
        }

        consume(static_cast<size_t>(edge_pos_buf[slot]), cur_rows);

        // All lanes must finish reading the slot before prefetch() reuses it.
        __syncwarp();
        pipe.consumer_release();
        prefetch(iter + NUM_STAGES);
    }
}

// =============================================================================
// GSDDMM forward kernel. See gsddmm.cuh for semantics and conventions.
// =============================================================================
template <GSDDMM_OP op, GSDDMM_MEMBER ll, GSDDMM_MEMBER rr, size_t N_PER_BLOCK, size_t D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t, int PIPELINE_STAGES>
__global__ void __launch_bounds__(N_PER_BLOCK *kWarpSize) GSDDMM_forward_edge_block( // no-format
    size_t N,
    cuda_t const *__restrict__ L, cuda_t const *__restrict__ R, cuda_t *__restrict__ O,
    index_t const *__restrict__ row_ptr, index_t const *__restrict__ col_idx,
    index_t const *__restrict__ node_indices
) {
    static_assert(D_CONST % 32 == 0, "D_CONST must be a multiple of 32 so a warp covers the row an integral number of times");
    static_assert(std::popcount(D_CONST / 32) == 1, "D_CONST / 32 must be a power of two for the tile decomposition");
    static_assert((D_CONST * sizeof(cuda_t)) % 16 == 0, "Row width in bytes must be a multiple of 16 for wide copies");

    using Plan = GsddmmPlan<op, ll, rr>;

    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr size_t TW = TW_SELECTOR::value;  // Tile width
    static_assert(D_CONST % TW == 0, "Feature dim should be divisible by Tile width");
    constexpr size_t TILES            = D_CONST / TW;
    constexpr size_t TILES_PER_THREAD = ceil_div(TILES, kWarpSize);

    using Tile  = TileOps<TW, cuda_t, accum_t>;
    using vec_t = typename Tile::vec_t;
    using VecOp = GsddmmVecOp<op, TW, cuda_t>;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr int NUM_STAGES = PIPELINE_STAGES;
    // With no per-edge rows (both operands are Dst_V, or Copy from Dst_V) there
    // is nothing to prefetch — the pipeline degenerates to the plain loop.
    constexpr bool USE_PIPELINE = (PIPELINE_STAGES > 0) && (Plan::NUM_EDGE_ROWS > 0);

    constexpr size_t ROWS_CAP = Plan::NUM_EDGE_ROWS > 0 ? Plan::NUM_EDGE_ROWS : 1;  // arrays need a nonzero bound

    const size_t node_i = static_cast<size_t>(node_indices[blockIdx.x]);

    __builtin_assume(threadIdx.y < static_cast<unsigned>(N_PER_BLOCK));
    const size_t lane_id = threadIdx.x;
    __builtin_assume(lane_id < static_cast<size_t>(kWarpSize));
    const size_t warp_id = threadIdx.y;

    if (node_i >= N) [[unlikely]] {
        return;
    }

    const index_t edge_start = row_ptr[node_i];
    const index_t edge_end   = row_ptr[node_i + 1];
    const size_t num_edges   = static_cast<size_t>(edge_end - edge_start);

    // Isolated node: no edges, hence no output rows — nothing to do.
    if (num_edges == 0) [[unlikely]] {
        return;
    }

    // Shared memory layout (everything written through 16-byte vectors comes first):
    //   dst_sh[NUM_DST_ROWS * D_CONST] as cuda_t
    //       -- Dst_V operand rows, identical for every edge of this CSR row
    //   edge_dbuf[N_PER_BLOCK * NUM_EDGE_ROWS * NUM_STAGES * D_CONST] as cuda_t
    //       -- per-warp ping-pong for gathered Src_V/Edge rows, only when USE_PIPELINE
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *const dst_sh    = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *const edge_dbuf = dst_sh + Plan::NUM_DST_ROWS * D_CONST;

    // Cooperative staging of the Dst_V operand rows via 16-byte copies (all threads).
    if constexpr (Plan::NUM_DST_ROWS > 0) {
        constexpr size_t ELEMS_PER_F4 = sizeof(float4) / sizeof(cuda_t);  // remainder is guaranteed to be zero
        constexpr size_t NUM_LOADS    = D_CONST / ELEMS_PER_F4;
        const size_t tid              = warp_id * kWarpSize + lane_id;
        constexpr size_t NUM_THREADS  = N_PER_BLOCK * kWarpSize;

        if constexpr (Plan::L_DST) {
            float4 const *const src = reinterpret_cast<float4 const *>(L + node_i * D_CONST);
            float4 *const dst       = reinterpret_cast<float4 *>(dst_sh);
            for (size_t i = tid; i < NUM_LOADS; i += NUM_THREADS) {
                dst[i] = src[i];
            }
        }
        if constexpr (Plan::R_DST) {
            float4 const *const src = reinterpret_cast<float4 const *>(R + node_i * D_CONST);
            float4 *const dst       = reinterpret_cast<float4 *>(dst_sh + (Plan::L_DST ? D_CONST : 0));
            for (size_t i = tid; i < NUM_LOADS; i += NUM_THREADS) {
                dst[i] = src[i];
            }
        }
        __syncthreads();
    }

    cuda_t const *const sh_l = dst_sh;
    cuda_t const *const sh_r = dst_sh + (Plan::L_DST ? D_CONST : 0);

    cuda_t *const my_dbuf = edge_dbuf + warp_id * Plan::NUM_EDGE_ROWS * NUM_STAGES * D_CONST;

    // consume(e, rows): apply the op for edge e. rows[] holds the per-edge
    // operand rows (shared-memory slots when pipelined, global rows otherwise);
    // Dst_V operands are taken from the staged shared rows instead.
    auto consume = [&](size_t e, cuda_t const *const(&rows)[ROWS_CAP]) {
        if constexpr (!Plan::IS_DOT) {
            cuda_t *const o_row = O + e * D_CONST;
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t v = lane_id + kWarpSize * t;
                if (v < TILES) [[likely]] {
                    const vec_t lv = Plan::L_DST ? Tile::read(sh_l, v) : Tile::read(rows[Plan::L_SLOT], v);
                    if constexpr (op == GSDDMM_OP::Copy) {
                        Tile::write(o_row, v, lv);
                    } else {
                        const vec_t rv = Plan::R_DST ? Tile::read(sh_r, v) : Tile::read(rows[Plan::R_SLOT], v);
                        Tile::write(o_row, v, VecOp::apply(lv, rv));
                    }
                }
            }
        } else {
            accum_t partial{};
#pragma unroll
            for (size_t t = 0; t < TILES_PER_THREAD; ++t) {
                const size_t v = lane_id + kWarpSize * t;
                if (v < TILES) [[likely]] {
                    const vec_t lv = Plan::L_DST ? Tile::read(sh_l, v) : Tile::read(rows[Plan::L_SLOT], v);
                    const vec_t rv = Plan::R_DST ? Tile::read(sh_r, v) : Tile::read(rows[Plan::R_SLOT], v);
                    lv.template dot_product_<accum_t>(&partial, rv);
                }
            }
            partial = warp_reduce_sum(partial);
            if (lane_id == 0) {
                O[e] = static_cast<cuda_t>(partial);
            }
        }
    };

    if constexpr (USE_PIPELINE) {
        cuda_t const *row_bases[ROWS_CAP];
        bool row_edge_indexed[ROWS_CAP];
        if constexpr (!Plan::L_DST) {
            row_bases[Plan::L_SLOT]        = L;
            row_edge_indexed[Plan::L_SLOT] = Plan::L_EDGE_INDEXED;
        }
        if constexpr (!Plan::R_DST) {
            row_bases[Plan::R_SLOT]        = R;
            row_edge_indexed[Plan::R_SLOT] = Plan::R_EDGE_INDEXED;
        }
        gsddmm_pipelined_edge_loop<N_PER_BLOCK, D_CONST, NUM_STAGES, Plan::NUM_EDGE_ROWS, cuda_t, index_t>(
            warp_id, lane_id, num_edges, edge_start, col_idx, row_bases, row_edge_indexed, my_dbuf, consume
        );
    } else {
        const size_t rounds = ceil_div(num_edges, N_PER_BLOCK);
        for (size_t r = 0; r < rounds; ++r) {
            const size_t k = r * N_PER_BLOCK + warp_id;
            if (k >= num_edges) [[unlikely]] {
                break;
            }
            const index_t e = edge_start + static_cast<index_t>(k);
            const index_t j = col_idx[e];

            cuda_t const *rows[ROWS_CAP];
            if constexpr (!Plan::L_DST) {
                rows[Plan::L_SLOT] = L + (Plan::L_EDGE_INDEXED ? static_cast<size_t>(e) : static_cast<size_t>(j)) * D_CONST;
            }
            if constexpr (!Plan::R_DST) {
                rows[Plan::R_SLOT] = R + (Plan::R_EDGE_INDEXED ? static_cast<size_t>(e) : static_cast<size_t>(j)) * D_CONST;
            }
            consume(static_cast<size_t>(e), rows);
        }
    }
}

};  // namespace gsddmm
