#pragma once

#include <cstdint>

#include "common.cuh"
#include "common/gspmm_ops.cuh"
#include "common/row_group.cuh"
#include "common/row_shape.cuh"

// =============================================================================
// g-SpMM sum backward: the gradient w.r.t. the edge operand, for any feature dim D.
//
// For the edges e = (u -> v) of forward-CSR row v, with g = grad_out[v]:
//
//     add, copy_e  dE[e] = g                 sub  dE[e] = -g
//     mul          dE[e] = g * X[u]          div  dE[e] = -g * X[u] / E[e]^2
//
// and, for an edge operand broadcast over the features ([E] or [E, 1]), the sum of that over D:
// the row sum of g for add/sub/copy_e, <g, X[u]> for mul and div.
//
// Work is laid out as the forward's (reduction/reduction_forward.cuh): the light rows and the slices
// of the heavy rows, each a work item of a lane group (rows of up to 16 vectors) or a warp (longer
// rows, swept in passes, every pass an item of its own). A group holds its row's g in registers and
// walks its edges UNROLL at a time, so that many edge rows are in flight. Every edge belongs to
// exactly one item, so each slot of dE is written exactly once, straight in the operand dtype: no
// zero fill, no atomics. A broadcast gradient needs the whole row per edge, so there an item covers
// every pass of its row, and the group's lanes add their parts with shuffles among the group's own
// lanes: the groups of a warp walk different rows and are not in step with each other.
// =============================================================================

namespace turbo_gnn::gspmm {

template <int CODE>
using Shape = rows::Shape<CODE>;

template <FloatingNum cuda_t, IntegralNum index_t>
struct EdgeGradParams {
    cuda_t const *G;  // grad_out: [N_dst, D] rows at stride_g
    int64_t stride_g;
    cuda_t const *X;  // the node operand: [N_src, D] rows at stride_x; mul and div only
    int64_t stride_x;
    cuda_t const *E;  // the edge operand: [E, D] rows, or [E] scalars; div only
    cuda_t *dE;       // its gradient, shaped as E

    uint32_t D;
    uint32_t nvec;    // vectors per row
    uint32_t passes;  // items per row: its passes, or 1 for a broadcast gradient

    index_t const *row_ptr;
    index_t const *col_idx;
    int64_t num_slices;
    int64_t num_light;
    index_t const *light_nodes;
    index_t const *heavy_nodes;
    int const *chunk_node;   // [num_slices]: the slice's slot in heavy_nodes
    int const *chunk_start;  // [num_slices]: its first edge, relative to the row
    uint32_t slice_size;
};

inline constexpr uint32_t kEdgeGradWarps = 8;

// Edges walked together, their rows in flight at once.
inline constexpr uint32_t kEdgeGradUnroll = 4;

// Resident 8-warp blocks requested per SM. The kernels keep a row of g, UNROLL edge rows and their
// addresses live; spill-checked on sm_80-sm_100 (see the build notes in gspmm.cu).
template <BinaryOp BOP, bool BCAST, int SHAPE, FloatingNum cuda_t>
inline constexpr int kEdgeGradMinBlocks = [] {
    using Sh = Shape<SHAPE>;
    if (Sh::S > 1) {
        return 1;
    }
    constexpr bool kReadsRows = BOP == BinaryOp::MUL || BOP == BinaryOp::DIV;
    return std::min(kReadsRows ? 4 : 6, kMaxThreadsPerSM / static_cast<int>(kEdgeGradWarps * kWarpSize));
}();

// Sum over the G lanes of the calling lane's group, among those lanes only.
template <size_t G>
__device__ __forceinline__ float group_sum(float x, uint32_t lane) {
    if constexpr (G > 1) {
        const unsigned mask = G == kWarpSize ? FULL_WARP_MASK : ((1u << G) - 1u) << (lane & ~static_cast<uint32_t>(G - 1));
#pragma unroll
        for (uint32_t off = G / 2; off > 0; off >>= 1) {
            x += __shfl_xor_sync(mask, x, off);
        }
    }
    return x;
}

template <BinaryOp BOP, bool BCAST, int SHAPE, FloatingNum cuda_t, IntegralNum index_t>
__global__ void __launch_bounds__(kEdgeGradWarps *kWarpSize, kEdgeGradMinBlocks<BOP, BCAST, SHAPE, cuda_t>) edge_grad_kernel(
    EdgeGradParams<cuda_t, index_t> const p
) {
    using Sh                        = Shape<SHAPE>;
    static constexpr size_t G       = Sh::G;
    static constexpr size_t S       = Sh::S;
    static constexpr size_t VEC     = Sh::template vec<cuda_t>;
    using RG                        = RowGroup<G, S, VEC, cuda_t, float>;
    using Frag                      = typename RG::Frag;
    using Tile                      = typename RG::Tile;
    static constexpr size_t kAccs   = RG::kAccs;
    static constexpr uint32_t kPass = rows::kPassVecs<SHAPE>;
    static constexpr bool kWarpRows = G == kWarpSize;
    // A broadcast gradient sums the whole row per edge: one item takes every pass of its row.
    static constexpr bool kPassItems = kWarpRows && !BCAST;
    static constexpr bool kReadsX    = BOP == BinaryOp::MUL || BOP == BinaryOp::DIV;
    static constexpr bool kReadsE    = BOP == BinaryOp::DIV;
    static constexpr float kSign     = BOP == BinaryOp::SUB || BOP == BinaryOp::DIV ? -1.0f : 1.0f;
    static_assert(BOP != BinaryOp::COPY_U, "copy_u has no edge operand");

    const uint32_t lane    = threadIdx.x % kWarpSize;
    const uint32_t r       = lane % G;
    const int64_t warp     = static_cast<int64_t>(blockIdx.x) * (blockDim.x / kWarpSize) + threadIdx.x / kWarpSize;
    const int64_t worker   = warp * static_cast<int64_t>(RG::kGroupsPerWarp) + lane / G;
    const int64_t row_item = kPassItems ? worker / p.passes : worker;
    const uint32_t pass    = kPassItems ? static_cast<uint32_t>(worker - row_item * p.passes) : 0;
    if (row_item >= p.num_slices + p.num_light) {
        return;  // whole groups
    }

    const bool sliced       = row_item < p.num_slices;
    const int64_t node      = static_cast<int64_t>(sliced ? p.heavy_nodes[p.chunk_node[row_item]] : p.light_nodes[row_item - p.num_slices]);
    const index_t row_begin = p.row_ptr[node];
    const uint32_t degree   = static_cast<uint32_t>(p.row_ptr[node + 1] - row_begin);
    const uint32_t lo       = sliced ? min(static_cast<uint32_t>(p.chunk_start[row_item]), degree) : 0u;
    const uint32_t hi       = sliced ? lo + min(p.slice_size, degree - lo) : degree;
    if (lo >= hi) {
        return;  // whole groups, and no shuffle below concerns another group
    }
    index_t const *const __restrict__ ids = p.col_idx + row_begin;
    cuda_t const *const g_row             = p.G + node * p.stride_g;
    using uidx_t                          = std::make_unsigned_t<index_t>;
    const uint32_t stride_x               = static_cast<uint32_t>(p.stride_x);

    if constexpr (!BCAST && !kReadsX && VEC == 1) {
        // ---- add, sub, copy_e on rows of single elements, in one pass: the row's +-g in every edge
        // row of a contiguous block, written flat. Lane by lane in edge rows, a 28-byte row (D = 7
        // fp32) made each warp store four scattered 28-byte pieces: measured 500 us against the former
        // kernels' 370 on twitch-views. Flat, consecutive lanes write consecutive elements, and the
        // lane's feature advances by G mod D per step.
        if (p.passes == 1) {
            const uint64_t first = static_cast<uint64_t>(row_begin + static_cast<index_t>(lo)) * p.D;
            const uint64_t total = static_cast<uint64_t>(hi - lo) * p.D;
            const uint32_t step  = static_cast<uint32_t>(G % p.D);
            uint32_t f           = r % p.D;
#pragma unroll 4
            for (uint64_t k = r; k < total; k += G) {
                p.dE[first + k] = static_cast<cuda_t>(kSign * static_cast<float>(g_row[f]));
                f += step;
                f -= f >= p.D ? p.D : 0u;
            }
            return;
        }
    }
    if constexpr (!BCAST) {
        // ---- One gradient row per edge: the pass's lanes of g, combined with the edge's rows. ----
        const uint32_t v0    = pass * kPass;
        const uint32_t cnvec = min(p.nvec - v0, kPass);
        Frag gf;
        RG::load(gf, g_row + static_cast<int64_t>(v0) * VEC, r, cnvec);
        alignas(16) float g[kAccs];
        RG::to_accum(gf, g);
        if constexpr (!kReadsX) {
            // add, sub, copy_e: every edge of the row takes the same row, +-g.
#pragma unroll
            for (size_t k = 0; k < kAccs; ++k) {
                g[k] *= kSign;
            }
        }
        // Hoisted addressing, as in the forward: a lane's vector offset within a row is fixed, and a
        // lane past the end of the pass reads the pass's last vector (and stores nothing).
        cuda_t const *lane_x[S];
        cuda_t const *lane_e[S];
#pragma unroll
        for (size_t s = 0; s < S; ++s) {
            const int64_t at = static_cast<int64_t>(v0 + min(RG::vec_index(r, s), cnvec - 1)) * VEC;
            lane_x[s]        = p.X + at;
            lane_e[s]        = p.E + at;
        }
        auto read_row = [&](cuda_t const *const(&lanes)[S], uint64_t off) {
            Frag f;
#pragma unroll
            for (size_t s = 0; s < S; ++s) {
                f.v[s] = Tile::read(lanes[s] + off, 0);
            }
            return f;
        };
        auto store = [&](index_t e, float const(&out)[kAccs]) {
            cuda_t *const row = p.dE + static_cast<int64_t>(e) * p.D + static_cast<int64_t>(v0) * VEC;
#pragma unroll
            for (size_t s = 0; s < S; ++s) {
                const uint32_t v = RG::vec_index(r, s);
                if (v < cnvec) {
                    Tile::write(row, v, Tile::convert_from_accum(&out[s * VEC]));
                }
            }
        };

        if constexpr (!kReadsX) {
            for (uint32_t slot = lo; slot < hi; ++slot) {
                store(row_begin + static_cast<index_t>(slot), g);
            }
        } else {
            for (uint32_t base = lo; base < hi; base += kEdgeGradUnroll) {
                const uint32_t n = min(kEdgeGradUnroll, hi - base);
                Frag xf[kEdgeGradUnroll];
                Frag ef[kEdgeGradUnroll];
#pragma unroll
                for (uint32_t u = 0; u < kEdgeGradUnroll; ++u) {
                    // A slot past the end re-reads the last edge's rows: no predicate on the loads.
                    const uint32_t slot = base + min(u, n - 1);
                    const uint64_t j    = static_cast<uidx_t>(ids[slot]);
                    xf[u]               = read_row(lane_x, j * stride_x);
                    if constexpr (kReadsE) {
                        ef[u] = read_row(lane_e, static_cast<uint64_t>(static_cast<uidx_t>(row_begin + static_cast<index_t>(slot))) * p.D);
                    }
                }
#pragma unroll
                for (uint32_t u = 0; u < kEdgeGradUnroll; ++u) {
                    if (u < n) {
                        alignas(16) float out[kAccs];
                        RG::to_accum(xf[u], out);
                        if constexpr (kReadsE) {
                            alignas(16) float ev[kAccs];
                            RG::to_accum(ef[u], ev);
#pragma unroll
                            for (size_t k = 0; k < kAccs; ++k) {
                                out[k] = BinaryOps<BOP>::grad_rhs(out[k], ev[k], g[k]);
                            }
                        } else {
#pragma unroll
                            for (size_t k = 0; k < kAccs; ++k) {
                                out[k] *= g[k];
                            }
                        }
                        store(row_begin + static_cast<index_t>(base + u), out);
                    }
                }
            }
        }
    } else {
        // ---- A scalar per edge: the sum over the row of its gradient row. ----
        const uint32_t passes = kWarpRows ? ceil_div(p.nvec, kPass) : 1u;
        auto g_part           = [&](uint32_t ps, Frag& gf) {
            const uint32_t v0 = ps * kPass;
            RG::load(gf, g_row + static_cast<int64_t>(v0) * VEC, r, min(p.nvec - v0, kPass));
        };
        if constexpr (!kReadsX) {
            // add, sub, copy_e: the row sum of g, the same for every edge of the row. The group's
            // lanes then store it for consecutive edges, one each.
            float total = 0.0f;
            for (uint32_t ps = 0; ps < passes; ++ps) {
                Frag gf;
                g_part(ps, gf);
                alignas(16) float g[kAccs];
                RG::to_accum(gf, g);
#pragma unroll
                for (size_t k = 0; k < kAccs; ++k) {
                    total += g[k];
                }
            }
            const cuda_t val = static_cast<cuda_t>(kSign * group_sum<G>(total, lane));
            for (uint32_t slot = lo + r; slot < hi; slot += G) {
                p.dE[row_begin + static_cast<index_t>(slot)] = val;
            }
        } else {
            // mul, div: <g, X[u]> per edge, UNROLL edges at a time.
            Frag g1;
            if (passes == 1) {
                g_part(0, g1);  // one pass: g stays in registers across the row's edges
            }
            for (uint32_t base = lo; base < hi; base += kEdgeGradUnroll) {
                const uint32_t n = min(kEdgeGradUnroll, hi - base);
                float dots[kEdgeGradUnroll];
                uint64_t jrow[kEdgeGradUnroll];
#pragma unroll
                for (uint32_t u = 0; u < kEdgeGradUnroll; ++u) {
                    dots[u] = 0.0f;
                    jrow[u] = static_cast<uint64_t>(static_cast<uidx_t>(ids[base + min(u, n - 1)])) * p.stride_x;
                }
                for (uint32_t ps = 0; ps < passes; ++ps) {
                    const uint32_t v0    = ps * kPass;
                    const uint32_t cnvec = min(p.nvec - v0, kPass);
                    Frag gf              = g1;
                    if (passes > 1) {
                        g_part(ps, gf);
                    }
                    Frag xf[kEdgeGradUnroll];
#pragma unroll
                    for (uint32_t u = 0; u < kEdgeGradUnroll; ++u) {
                        RG::load(xf[u], p.X + jrow[u] + static_cast<int64_t>(v0) * VEC, r, cnvec);
                    }
#pragma unroll
                    for (uint32_t u = 0; u < kEdgeGradUnroll; ++u) {
                        dots[u] += RG::dot(gf, xf[u]);
                    }
                }
#pragma unroll
                for (uint32_t u = 0; u < kEdgeGradUnroll; ++u) {
                    if (u < n) {
                        const index_t e = row_begin + static_cast<index_t>(base + u);
                        float val       = group_sum<G>(dots[u], lane);
                        if constexpr (kReadsE) {
                            const float ev = static_cast<float>(p.E[e]);
                            val            = -val / (ev * ev);
                        }
                        if (r == 0) {
                            p.dE[e] = static_cast<cuda_t>(val);
                        }
                    }
                }
            }
        }
    }
}

}  // namespace turbo_gnn::gspmm
