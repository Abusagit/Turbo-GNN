#pragma once

#include <cstdint>
#include <cuda/pipeline>

#include "common.cuh"

/// Registers cap occupancy for this kernel at every D: the compiler settles on 64 registers,
/// which at 8 warps allows only 65536/(64*32*8) = 4 blocks per SM -- 50% occupancy. Supplying
/// `minBlocksPerMultiprocessor` forces the allocator lower: asking for B blocks of T threads
/// caps it at 65536/(B*T) registers per thread.
///
/// Applied only at D_CONST >= 256, where the measured d=256 results are flat while d=128 gains
/// 1.0055x: at d=128 each lane owns exactly one tile and the kernel is already at its best
/// occupancy for the register count, so constraining it there would only risk spills. The target
/// is ~51 registers (a 5th block at 8 warps, 50% -> 62%), expressed as 40/WARPS_PER_BLOCK so
/// wider blocks are not asked for something unsatisfiable -- 16 warps would otherwise be capped
/// at 25 registers, which buys occupancy back with a local-memory spill and loses.
template <int WARPS_PER_BLOCK, int D_CONST, FloatingNum cuda_t>
inline constexpr int kGATv2FwdMinBlocksPerSM = [] {
    // Condition the register budget on tile geometry, not on D_CONST. TILES = D/TW, and TW
    // depends on the dtype, so D_CONST alone does not describe the kernel's register profile:
    //   TILES == 32 (one tile per lane, saturated) -- fp32/d=128, fp16/d=256. Already the
    //       winning shape; squeezing registers here spilled 14 of 16 depth-6 instantiations
    //       and cost fp16/d=256 3% (1.0091x -> 0.9773x). Leave it at nvcc's own 64.
    //   TILES  > 32 (two or more tiles per lane) -- fp32/d=256. Cannot saturate, and the extra
    //       block per SM is worth +0.3% there (0.9988x -> 1.0024x). Cap to 48.
    // Never emit 1: that asserts a single resident block suffices and the allocator spends
    // freely (measured 64 -> 119 registers, occupancy 50% -> 25%).
    constexpr int kTiles = static_cast<int>(D_CONST / SelectTW<D_CONST, cuda_t>::value);
    constexpr int want   = (kTiles > 32) ? 40 : 32;   // 40/W => 48 regs, 32/W => 64 regs
    constexpr int blocks = want / WARPS_PER_BLOCK;
    return blocks < 1 ? 1 : (blocks > 32 ? 32 : blocks);
}();


// =============================================================================
// GATv2 Kernel with CSR Graph Format
// =============================================================================

template <
    turbo_gnn::sched::ScheduleKind SK, int WARPS_PER_BLOCK, int D_CONST, FloatingNum cuda_t, typename index_t,
    FloatingNum accum_t = float,
    // PIPELINE_STAGES == 0 disables the async-copy pipeline (plain warp-strided loop);
    // PIPELINE_STAGES >= 1 enables it with that many ping-pong stages for r[j].
    int PIPELINE_STAGES = 0>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize, kGATv2FwdMinBlocksPerSM<WARPS_PER_BLOCK, D_CONST, cuda_t>) GATv2Forward_Kernel(
    size_t N,
    size_t H,
    size_t D,
    const cuda_t *__restrict__ d_l,
    const cuda_t *__restrict__ d_r,
    int64_t stride_l_n,
    int64_t stride_l_h,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr,
    const index_t *__restrict__ d_col_idx,
    turbo_gnn::sched::SchedulerParams<index_t> sched_params,
    const cuda_t *__restrict__ d_attn_vec,
    cuda_t *__restrict__ d_h_out,
    float *__restrict__ d_logsumexp_out,
    float negative_slope
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;

    constexpr int TW               = TW_SELECTOR::value;                                                     // Tile width
    constexpr int TILES            = (D_CONST + TW - 1) / TW;                                                // Total tiles count
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;  // Tiles per thread
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;                                                  // Accumulatores used by one thread

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    using vec_t = typename Tile::vec_t;

    const int head_h  = blockIdx.y;

    // Body in a lambda: its `return`s become per-node `continue` semantics.
    auto process_node = [&](const int node_i) {
    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    if (node_i >= static_cast<int>(N) || head_h >= static_cast<int>(H)) [[unlikely]] {
        return;
    }

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    cuda_t *h_out_base = d_h_out + (static_cast<int64_t>(node_i) * H + head_h) * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < TILES; v += kWarpSize) {
                Tile::write_zero(h_out_base, v);
            }
            if (lane == 0) {
                d_logsumexp_out[static_cast<int64_t>(node_i) * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    const cuda_t *l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t *a_base = d_attn_vec + head_h * D_CONST;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0;
    constexpr int NUM_STAGES    = PIPELINE_STAGES;

    // Shared memory layout:
    //   l_sh:      D_CONST * sizeof(cuda_t)                              -- read-only
    //   r_dbuf:    WARPS_PER_BLOCK * NUM_STAGES * D_CONST * sizeof(cuda_t) -- per-warp ping-pong for async r[j], only when USE_PIPELINE
    //   warp_out:  WARPS_PER_BLOCK * D_CONST * sizeof(accum_t)           -- per-warp output accum
    //   warp_max:  WARPS_PER_BLOCK * sizeof(accum_t)                     -- per-warp softmax max
    //   warp_sum:  WARPS_PER_BLOCK * sizeof(accum_t)                     -- per-warp softmax sum_exp
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *l_sh   = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *r_dbuf = l_sh + D_CONST;  // only meaningful when USE_PIPELINE

    constexpr size_t r_dbuf_bytes = USE_PIPELINE ? WARPS_PER_BLOCK * NUM_STAGES * D_CONST * sizeof(cuda_t) : 0;
    accum_t *warp_out             = reinterpret_cast<accum_t *>(sh_raw + D_CONST * sizeof(cuda_t) + r_dbuf_bytes);
    accum_t *warp_max             = warp_out + WARPS_PER_BLOCK * D_CONST;
    accum_t *warp_sum             = warp_max + WARPS_PER_BLOCK;

    accum_t *my_out = warp_out + warp_id * D_CONST;

    // Cooperative load of l into shared memory using all threads
    {
        static_assert((D_CONST * sizeof(cuda_t)) % 16 == 0, "D_CONST in bytes must be a multiple of 16 for float4 vectorized loads.");
        constexpr int f4_count = (D_CONST * static_cast<int>(sizeof(cuda_t))) / 16;
        const float4 *l_src4   = reinterpret_cast<const float4 *>(l_base);
        float4 *l_sh4          = reinterpret_cast<float4 *>(l_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kWarpSize) {
            l_sh4[i] = l_src4[i];
        }
    }
    __syncthreads();

    // Per-warp register accumulators
    accum_t h_acc[ACCS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
        h_acc[i] = accum_t{};
    }

    OnlineSoftmaxState softmax_state;

    if constexpr (USE_PIPELINE) {
        // Blocked, not strided: each warp owns a contiguous run of this node's edges. That makes
        // the warp's upcoming column indices contiguous in memory, so all 32 can be fetched in one
        // coalesced transaction and broadcast from registers -- see idx_of() below.
        const int per_warp  = (num_neighbors + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int my_start  = warp_id * per_warp;
        const int my_end    = min(my_start + per_warp, static_cast<int>(num_neighbors));
        const int loop_iters = (my_end > my_start) ? (my_end - my_start) : 0;

        if (loop_iters > 0) {
            // Affine in the slot, so compute the address instead of holding an array indexed by
            // a runtime value -- such an array lands in local memory and the spill grows with depth.
            auto slot_row = [r_dbuf, warp_id](int slot) -> cuda_t * {
                return r_dbuf + (warp_id * NUM_STAGES + slot) * D_CONST;
            };

            // The column index used to be loaded synchronously inside prefetch(), so the cp.async
            // for the neighbour row could not even be *issued* until that load returned from DRAM:
            // the two-hop chain col_idx[e] -> X[col_idx[e]] was only half covered. Here lane L
            // pre-loads the index for edge (base + L) in a single coalesced 32-wide load, and each
            // edge's index is then broadcast from a register. The async issue no longer waits on
            // global memory at all.
            index_t idx_reg{};
            int idx_base = -1;
            auto idx_of = [&idx_reg, &idx_base, d_col_idx, edge_start, my_start, my_end, lane](int i) -> index_t {
                const int b = i & ~31;
                if (b != idx_base) {
                    const int pos = my_start + b + lane;
                    idx_reg  = (pos < my_end) ? d_col_idx[edge_start + static_cast<index_t>(pos)] : index_t{};
                    idx_base = b;
                }
                return __shfl_sync(0xffffffffu, idx_reg, i & 31);
            };

            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            auto prefetch = [&pipe, &idx_of, loop_iters, d_r, stride_r_n, head_h, stride_r_h, slot_row, lane](int it) {
                pipe.producer_acquire();
                if (it < loop_iters) {
                    const index_t j     = idx_of(it);
                    const cuda_t *r_src = d_r + j * stride_r_n + head_h * stride_r_h;
                    async_copy_row_warp<D_CONST, cuda_t>(slot_row(it % NUM_STAGES), r_src, pipe, lane);
                }
                pipe.producer_commit();
            };

#pragma unroll
            for (int s = 0; s < NUM_STAGES; ++s) {
                prefetch(s);
            }

            // Two edges per iteration when the pipeline holds at least two rows. This is the
            // reason to pipeline at all here: the per-edge critical path is a five-step
            // warp_reduce_sum shuffle chain, and a single edge cannot overlap it with anything.
            // Two staged rows give two *independent* reduction chains that the scheduler
            // interleaves, and l[i]/a are read once per pair instead of once per edge. The
            // register path cannot do this -- holding two neighbour rows in registers costs
            // 2 * TILES_PER_THREAD vectors in a kernel where registers already bind occupancy.
            int iter = 0;
            if constexpr (NUM_STAGES >= 2) {
                for (; iter + 1 < loop_iters; iter += 2) {
                    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 2>(pipe);
                    __syncwarp();
                    cuda_t const *r0 = slot_row(iter % NUM_STAGES);
                    cuda_t const *r1 = slot_row((iter + 1) % NUM_STAGES);

                    accum_t d0{}, d1{};
#pragma unroll
                    for (int t = 0; t < TILES_PER_THREAD; ++t) {
                        int v = lane + kWarpSize * t;
                        if (v < TILES) {
                            const vec_t lv = Tile::read(l_sh, v);
                            const vec_t av = Tile::read(a_base, v);
                            d0 += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r0, v), av, negative_slope);
                            d1 += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r1, v), av, negative_slope);
                        }
                    }
                    // Independent, so the two shuffle chains overlap instead of serialising.
                    const accum_t dot0 = warp_reduce_sum(d0);
                    const accum_t dot1 = warp_reduce_sum(d1);

                    const accum_t rescale0 = softmax_state.update(dot0);
#pragma unroll
                    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                        h_acc[i] *= rescale0;
                    }
                    const accum_t c0 = AccumOps::exp(dot0 - softmax_state.max_val);
#pragma unroll
                    for (int t = 0; t < TILES_PER_THREAD; ++t) {
                        int v = lane + kWarpSize * t;
                        if (v < TILES) {
                            Tile::read(r0, v).template weighted_accum_<accum_t>(&h_acc[t * TW], c0);
                        }
                    }

                    const accum_t rescale1 = softmax_state.update(dot1);
#pragma unroll
                    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                        h_acc[i] *= rescale1;
                    }
                    const accum_t c1 = AccumOps::exp(dot1 - softmax_state.max_val);
#pragma unroll
                    for (int t = 0; t < TILES_PER_THREAD; ++t) {
                        int v = lane + kWarpSize * t;
                        if (v < TILES) {
                            Tile::read(r1, v).template weighted_accum_<accum_t>(&h_acc[t * TW], c1);
                        }
                    }

                    __syncwarp();
                    pipe.consumer_release();
                    pipe.consumer_release();
                    prefetch(iter + NUM_STAGES);
                    prefetch(iter + NUM_STAGES + 1);
                }
            }

            // Odd tail (and the whole loop when NUM_STAGES == 1, which no longer occurs).
            for (; iter < loop_iters; ++iter) {
                cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
                __syncwarp();
                cuda_t const *r_cur = slot_row(iter % NUM_STAGES);

                accum_t dot_lane{};
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        const vec_t lv = Tile::read(l_sh, v);
                        const vec_t av = Tile::read(a_base, v);
                        dot_lane += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r_cur, v), av, negative_slope);
                    }
                }
                const accum_t dot = warp_reduce_sum(dot_lane);

                const accum_t rescale = softmax_state.update(dot);
#pragma unroll
                for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                    h_acc[i] *= rescale;
                }

                const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        Tile::read(r_cur, v).template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                    }
                }

                __syncwarp();
                pipe.consumer_release();
                prefetch(iter + NUM_STAGES);
            }
        }
    } else {
        // Warp-strided neighbor loop
        for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
            index_t neighbor_j   = d_col_idx[edge_start + static_cast<index_t>(k)];
            const cuda_t *r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

            vec_t r_regs[TILES_PER_THREAD];

            accum_t dot_lane{};
#pragma unroll
            for (int t = 0; t < TILES_PER_THREAD; ++t) {
                int v = lane + kWarpSize * t;
                if (v < TILES) {
                    const vec_t lv = Tile::read(l_sh, v);
                    r_regs[t]      = Tile::read(r_base, v);
                    const vec_t av = Tile::read(a_base, v);
                    dot_lane += Tile::gatv2_dot_leaky_relu(lv, r_regs[t], av, negative_slope);
                }
            }
            const accum_t dot = warp_reduce_sum(dot_lane);

            const accum_t rescale = softmax_state.update(dot);
#pragma unroll
            for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                h_acc[i] *= rescale;
            }

            const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
            for (int t = 0; t < TILES_PER_THREAD; ++t) {
                int v = lane + kWarpSize * t;
                if (v < TILES) {
                    r_regs[t].template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                }
            }
        }
    }

// Write per-warp results to shared memory
#pragma unroll
    for (int t = 0; t < TILES_PER_THREAD; ++t) {
        const int v = lane + kWarpSize * t;
        if (v < TILES) {
            constexpr size_t compact_N  = std::min<size_t>(TW, VecFloat<1, cuda_t>::max_vec_size_bytes / std::max(sizeof(cuda_t), sizeof(accum_t)));
            constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
            for (size_t i = 0; i < repeat_cnt; ++i) {
                TileOps<compact_N, accum_t>::write(
                    my_out, v * repeat_cnt + i, reinterpret_cast<VecFloat<compact_N, accum_t> const *>(h_acc)[t * repeat_cnt + i]
                );
            }
        }
    }

    if (lane == 0) {
        warp_max[warp_id] = softmax_state.max_val;
        warp_sum[warp_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // Cross-warp online-softmax reduction (warp 0 only)
    if (warp_id == 0) {
        accum_t global_max = -FLT_MAX;
        accum_t global_sum{};
        accum_t inv_sum{};

        if (lane == 0) {
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_max = AccumOps::max(global_max, warp_max[w]);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_sum = AccumOps::fma(warp_sum[w], AccumOps::exp(warp_max[w] - global_max), global_sum);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                warp_sum[w] = AccumOps::exp(warp_max[w] - global_max);
            }
            inv_sum                                       = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
            d_logsumexp_out[static_cast<int64_t>(node_i) * H + head_h] =
                (global_sum > 0.0f) ? (global_max + AccumOps::log(global_sum)) : -INFINITY;
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

// Combine all warps' outputs with proper rescaling
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
                accum_t combined[TW];
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    combined[ep]    = accum_t{};
                    const int d_idx = v * TW + ep;
#pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                        combined[ep] = AccumOps::fma(warp_sum[w], warp_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_convert_from_accum(&h_out_base[v * TW], combined);
            }
        }
    }
    };  // process_node

    using Sched = turbo_gnn::sched::NodeScheduler<SK, index_t, /*SyncBlock=*/true>;
    __shared__ typename Sched::SharedStorage sched_smem;
    Sched sched(sched_params, sched_smem);
    for (auto work = sched.first(); sched.valid(work); work = sched.next(work)) {
        process_node(static_cast<int>(sched.node(work)));
    }
}

// ================================================================================================
// Split-K heavy path. Mirrors the GT version in csrc/gt/gt_forward.cu; see the rationale there.
// One block per fixed-size slice of a heavy node's edge list, partials merged by a second kernel,
// so the heavy bucket's grid is sized by its edge count rather than its node count.
// ================================================================================================

template <
    int WARPS_PER_BLOCK, int D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float,
    // PIPELINE_STAGES == 0 keeps the plain warp-strided loop; >= 1 stages r[j] through shared
    // memory with that many ping-pong buffers, as GATv2Forward_Kernel does for the light bucket.
    int PIPELINE_STAGES = 0>
__global__ void __launch_bounds__(WARPS_PER_BLOCK *kWarpSize, kGATv2FwdMinBlocksPerSM<WARPS_PER_BLOCK, D_CONST, cuda_t>) GATv2ForwardSlice_Kernel(
    size_t N, size_t H, size_t D,
    const cuda_t *__restrict__ d_l, const cuda_t *__restrict__ d_r,
    int64_t stride_l_n, int64_t stride_l_h, int64_t stride_r_n, int64_t stride_r_h,
    const index_t *__restrict__ d_row_ptr, const index_t *__restrict__ d_col_idx,
    const index_t *__restrict__ heavy_nodes,
    const int *__restrict__ chunk_node, const int *__restrict__ chunk_start,
    int slice_size, int num_slices,
    const cuda_t *__restrict__ d_attn_vec,
    accum_t *__restrict__ part_o, accum_t *__restrict__ part_ml,
    float negative_slope
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;
    constexpr int TW               = TW_SELECTOR::value;
    constexpr int TILES            = (D_CONST + TW - 1) / TW;
    constexpr int TILES_PER_THREAD = (TILES + TW_SELECTOR::threads_per_d - 1) / TW_SELECTOR::threads_per_d;
    constexpr int ACCS_PER_THREAD  = TW * TILES_PER_THREAD;

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;
    using vec_t    = typename Tile::vec_t;

    const int slice_id = blockIdx.x;
    const int head_h   = blockIdx.y;
    if (slice_id >= num_slices || head_h >= static_cast<int>(H)) [[unlikely]] {
        return;
    }

    const int warp_id = threadIdx.x / kWarpSize;
    const int lane    = threadIdx.x % kWarpSize;

    const int slot     = chunk_node[slice_id];
    const int node_i   = static_cast<int>(heavy_nodes[slot]);
    if (node_i >= static_cast<int>(N)) [[unlikely]] {
        return;
    }

    const index_t edge_start = d_row_ptr[node_i];
    const int num_neighbors  = static_cast<int>(d_row_ptr[node_i + 1] - edge_start);

    const int local_start = chunk_start[slice_id];
    const int local_end   = min(local_start + slice_size, num_neighbors);

    const int64_t part_base = (int64_t)slice_id * H + head_h;

    if (local_start >= local_end) [[unlikely]] {
        if (warp_id == 0) {
            // Scalar: partials are accum_t, whose width need not match the input dtype's TW.
            for (int f = lane; f < D_CONST; f += kWarpSize) {
                part_o[part_base * D_CONST + f] = accum_t{};
            }
            if (lane == 0) {
                part_ml[part_base * 2 + 0] = -FLT_MAX;
                part_ml[part_base * 2 + 1] = accum_t{};
            }
        }
        return;
    }

    const cuda_t *l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t *a_base = d_attn_vec + head_h * D_CONST;

    static_assert(PIPELINE_STAGES >= 0, "pipeline_stages must be >= 0 (0 disables the pipeline)");
    constexpr bool USE_PIPELINE = PIPELINE_STAGES > 0;
    constexpr int NUM_STAGES    = PIPELINE_STAGES;

    // Layout: l_sh | r_dbuf (only when USE_PIPELINE) | warp_out | warp_max | warp_sum
    extern __shared__ __align__(16) uint8_t sh_raw[];
    cuda_t *l_sh   = reinterpret_cast<cuda_t *>(sh_raw);
    cuda_t *r_dbuf = l_sh + D_CONST;  // only meaningful when USE_PIPELINE

    constexpr size_t r_dbuf_bytes = USE_PIPELINE ? WARPS_PER_BLOCK * NUM_STAGES * D_CONST * sizeof(cuda_t) : 0;
    accum_t *warp_out = reinterpret_cast<accum_t *>(sh_raw + D_CONST * sizeof(cuda_t) + r_dbuf_bytes);
    accum_t *warp_max = warp_out + WARPS_PER_BLOCK * D_CONST;
    accum_t *warp_sum = warp_max + WARPS_PER_BLOCK;
    accum_t *my_out   = warp_out + warp_id * D_CONST;

    {
        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4 *l_src4   = reinterpret_cast<const float4 *>(l_base);
        float4 *l_sh4          = reinterpret_cast<float4 *>(l_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kWarpSize) {
            l_sh4[i] = l_src4[i];
        }
    }
    __syncthreads();

    accum_t h_acc[ACCS_PER_THREAD];
#pragma unroll
    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
        h_acc[i] = accum_t{};
    }

    OnlineSoftmaxState softmax_state;

    if constexpr (USE_PIPELINE) {
        // Blocked within this slice so each warp's column indices are contiguous and can be
        // fetched 32 at a time in one coalesced load (see idx_of()).
        const int slice_len = local_end - local_start;
        const int per_warp  = (slice_len + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK;
        const int my_start  = local_start + warp_id * per_warp;
        const int my_end    = min(my_start + per_warp, local_end);
        const int loop_iters = (my_end > my_start) ? (my_end - my_start) : 0;

        if (loop_iters > 0) {
            // Affine in the slot, so compute the address instead of holding an array indexed by
            // a runtime value -- such an array lands in local memory and the spill grows with depth.
            auto slot_row = [r_dbuf, warp_id](int slot) -> cuda_t * {
                return r_dbuf + (warp_id * NUM_STAGES + slot) * D_CONST;
            };

            // Index prefetch: one coalesced 32-wide load per 32 edges, broadcast from a register,
            // so issuing the row copy never waits on a global load. Without this the cp.async is
            // gated by col_idx[e] and only half of the two-hop chain is actually covered.
            index_t idx_reg{};
            int idx_base = -1;
            auto idx_of = [&idx_reg, &idx_base, d_col_idx, edge_start, my_start, my_end, lane](int i) -> index_t {
                const int b = i & ~31;
                if (b != idx_base) {
                    const int pos = my_start + b + lane;
                    idx_reg  = (pos < my_end) ? d_col_idx[edge_start + static_cast<index_t>(pos)] : index_t{};
                    idx_base = b;
                }
                return __shfl_sync(0xffffffffu, idx_reg, i & 31);
            };

            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            auto prefetch = [&pipe, &idx_of, loop_iters, d_r, stride_r_n, head_h, stride_r_h, slot_row, lane](int it) {
                pipe.producer_acquire();
                if (it < loop_iters) {
                    const index_t j     = idx_of(it);
                    const cuda_t *r_src = d_r + j * stride_r_n + head_h * stride_r_h;
                    async_copy_row_warp<D_CONST, cuda_t>(slot_row(it % NUM_STAGES), r_src, pipe, lane);
                }
                pipe.producer_commit();
            };

#pragma unroll
            for (int st = 0; st < NUM_STAGES; ++st) {
                prefetch(st);
            }

            // Same paired-edge structure as the light-bucket kernel: two staged rows give two
            // independent reduction chains. See there for the rationale. Two edges per iteration
            // reason to pipeline at all here: the per-edge critical path is a five-step
            // warp_reduce_sum shuffle chain, and a single edge cannot overlap it with anything.
            // Two staged rows give two *independent* reduction chains that the scheduler
            // interleaves, and l[i]/a are read once per pair instead of once per edge. The
            // register path cannot do this -- holding two neighbour rows in registers costs
            // 2 * TILES_PER_THREAD vectors in a kernel where registers already bind occupancy.
            int iter = 0;
            if constexpr (NUM_STAGES >= 2) {
                for (; iter + 1 < loop_iters; iter += 2) {
                    cuda::pipeline_consumer_wait_prior<NUM_STAGES - 2>(pipe);
                    __syncwarp();
                    cuda_t const *r0 = slot_row(iter % NUM_STAGES);
                    cuda_t const *r1 = slot_row((iter + 1) % NUM_STAGES);

                    accum_t d0{}, d1{};
#pragma unroll
                    for (int t = 0; t < TILES_PER_THREAD; ++t) {
                        int v = lane + kWarpSize * t;
                        if (v < TILES) {
                            const vec_t lv = Tile::read(l_sh, v);
                            const vec_t av = Tile::read(a_base, v);
                            d0 += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r0, v), av, negative_slope);
                            d1 += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r1, v), av, negative_slope);
                        }
                    }
                    // Independent, so the two shuffle chains overlap instead of serialising.
                    const accum_t dot0 = warp_reduce_sum(d0);
                    const accum_t dot1 = warp_reduce_sum(d1);

                    const accum_t rescale0 = softmax_state.update(dot0);
#pragma unroll
                    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                        h_acc[i] *= rescale0;
                    }
                    const accum_t c0 = AccumOps::exp(dot0 - softmax_state.max_val);
#pragma unroll
                    for (int t = 0; t < TILES_PER_THREAD; ++t) {
                        int v = lane + kWarpSize * t;
                        if (v < TILES) {
                            Tile::read(r0, v).template weighted_accum_<accum_t>(&h_acc[t * TW], c0);
                        }
                    }

                    const accum_t rescale1 = softmax_state.update(dot1);
#pragma unroll
                    for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                        h_acc[i] *= rescale1;
                    }
                    const accum_t c1 = AccumOps::exp(dot1 - softmax_state.max_val);
#pragma unroll
                    for (int t = 0; t < TILES_PER_THREAD; ++t) {
                        int v = lane + kWarpSize * t;
                        if (v < TILES) {
                            Tile::read(r1, v).template weighted_accum_<accum_t>(&h_acc[t * TW], c1);
                        }
                    }

                    __syncwarp();
                    pipe.consumer_release();
                    pipe.consumer_release();
                    prefetch(iter + NUM_STAGES);
                    prefetch(iter + NUM_STAGES + 1);
                }
            }

            // Odd tail (and the whole loop when NUM_STAGES == 1, which no longer occurs).
            for (; iter < loop_iters; ++iter) {
                cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
                __syncwarp();
                cuda_t const *r_cur = slot_row(iter % NUM_STAGES);

                accum_t dot_lane{};
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        const vec_t lv = Tile::read(l_sh, v);
                        const vec_t av = Tile::read(a_base, v);
                        dot_lane += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r_cur, v), av, negative_slope);
                    }
                }
                const accum_t dot = warp_reduce_sum(dot_lane);

                const accum_t rescale = softmax_state.update(dot);
#pragma unroll
                for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                    h_acc[i] *= rescale;
                }

                const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        Tile::read(r_cur, v).template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                    }
                }

                __syncwarp();
                pipe.consumer_release();
                prefetch(iter + NUM_STAGES);
            }

            // Odd tail: one edge left over when loop_iters is odd.
            for (; iter < loop_iters; ++iter) {
                cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);
                __syncwarp();
                cuda_t const *r_cur = slot_row(iter % NUM_STAGES);

                accum_t dot_lane{};
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        const vec_t lv = Tile::read(l_sh, v);
                        const vec_t av = Tile::read(a_base, v);
                        dot_lane += Tile::gatv2_dot_leaky_relu(lv, Tile::read(r_cur, v), av, negative_slope);
                    }
                }
                const accum_t dot = warp_reduce_sum(dot_lane);

                const accum_t rescale = softmax_state.update(dot);
#pragma unroll
                for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                    h_acc[i] *= rescale;
                }

                const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
                for (int t = 0; t < TILES_PER_THREAD; ++t) {
                    int v = lane + kWarpSize * t;
                    if (v < TILES) {
                        Tile::read(r_cur, v).template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                    }
                }

                __syncwarp();
                pipe.consumer_release();
                prefetch(iter + NUM_STAGES);
            }
        }
    } else {
        for (int k = local_start + warp_id; k < local_end; k += WARPS_PER_BLOCK) {
            index_t neighbor_j   = d_col_idx[edge_start + static_cast<index_t>(k)];
            const cuda_t *r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

            accum_t dot_lane{};
#pragma unroll
            for (int t = 0; t < TILES_PER_THREAD; ++t) {
                int v = lane + kWarpSize * t;
                if (v < TILES) {
                    const vec_t lv = Tile::read(l_sh, v);
                    const vec_t rv = Tile::read(r_base, v);
                    const vec_t av = Tile::read(a_base, v);
                    dot_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, negative_slope);
                }
            }
            const accum_t dot = warp_reduce_sum(dot_lane);

            const accum_t rescale = softmax_state.update(dot);
#pragma unroll
            for (int i = 0; i < ACCS_PER_THREAD; ++i) {
                h_acc[i] *= rescale;
            }

            const accum_t contrib = AccumOps::exp(dot - softmax_state.max_val);
#pragma unroll
            for (int t = 0; t < TILES_PER_THREAD; ++t) {
                int v = lane + kWarpSize * t;
                if (v < TILES) {
                    const vec_t rv = Tile::read(r_base, v);
                    rv.template weighted_accum_<accum_t>(&h_acc[t * TW], contrib);
                }
            }
        }
    }

#pragma unroll
    for (int t = 0; t < TILES_PER_THREAD; ++t) {
        const int v = lane + kWarpSize * t;
        if (v < TILES) {
            constexpr size_t compact_N =
                std::min<size_t>(TW, VecFloat<1, cuda_t>::max_vec_size_bytes / std::max(sizeof(cuda_t), sizeof(accum_t)));
            constexpr size_t repeat_cnt = TW / compact_N;
#pragma unroll
            for (size_t i = 0; i < repeat_cnt; ++i) {
                TileOps<compact_N, accum_t>::write(
                    my_out, v * repeat_cnt + i, reinterpret_cast<VecFloat<compact_N, accum_t> const *>(h_acc)[t * repeat_cnt + i]
                );
            }
        }
    }

    if (lane == 0) {
        warp_max[warp_id] = softmax_state.max_val;
        warp_sum[warp_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // Cross-warp merge, left un-normalised: dividing by a slice-local sum would have to be undone.
    if (warp_id == 0) {
        accum_t slice_max = -FLT_MAX;
        accum_t slice_sum{};

        if (lane == 0) {
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                slice_max = AccumOps::max(slice_max, warp_max[w]);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                slice_sum = AccumOps::fma(warp_sum[w], AccumOps::exp(warp_max[w] - slice_max), slice_sum);
            }
#pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                warp_sum[w] = AccumOps::exp(warp_max[w] - slice_max);
            }
            part_ml[part_base * 2 + 0] = slice_max;
            part_ml[part_base * 2 + 1] = slice_sum;
        }

        // Lane 0 rewrote warp_sum in shared memory and every lane reads it below. A shuffle
        // converges the warp but is not a memory fence; under Volta's independent thread
        // scheduling that ordering has to be established explicitly. (The in-place kernel above
        // is missing this, and the same hazard was confirmed in GT forward.)
        __syncwarp();

        accum_t *const o_base = part_o + part_base * D_CONST;
#pragma unroll
        for (int t = 0; t < TILES_PER_THREAD; ++t) {
            int v = lane + kWarpSize * t;
            if (v < TILES) {
#pragma unroll
                for (int ep = 0; ep < TW; ++ep) {
                    accum_t acc     = accum_t{};
                    const int d_idx = v * TW + ep;
#pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                        acc = AccumOps::fma(warp_sum[w], warp_out[w * D_CONST + d_idx], acc);
                    }
                    o_base[d_idx] = acc;
                }
            }
        }
    }
}

/// Merge every slice of one heavy node into its final GATv2 output row.
/// Grid (num_heavy, H), one warp per block. Keeps GATv2's own `(sum > 0)` guard convention,
/// which differs from GT's `max(1/sum, 0)` form.
template <int D_CONST, FloatingNum cuda_t, typename index_t, FloatingNum accum_t = float>
__global__ void __launch_bounds__(kWarpSize) GATv2MergeSlices_Kernel(
    size_t H,
    const index_t *__restrict__ d_row_ptr, const index_t *__restrict__ heavy_nodes,
    const int *__restrict__ node_chunk_offset,
    const accum_t *__restrict__ part_o, const accum_t *__restrict__ part_ml,
    cuda_t *__restrict__ d_h_out, float *__restrict__ d_logsumexp_out, int num_heavy
) {
    using TW_SELECTOR = SelectTW<D_CONST, cuda_t>;
    constexpr int TW    = TW_SELECTOR::value;
    constexpr int TILES = (D_CONST + TW - 1) / TW;

    using AccumOps = AdOps<accum_t>;
    using Tile     = TileOps<TW, cuda_t, accum_t>;

    const int slot   = blockIdx.x;
    const int head_h = blockIdx.y;
    if (slot >= num_heavy || head_h >= static_cast<int>(H)) [[unlikely]] {
        return;
    }

    const int lane   = threadIdx.x;
    const int node_i = static_cast<int>(heavy_nodes[slot]);
    cuda_t *h_out_base = d_h_out + ((int64_t)node_i * H + head_h) * D_CONST;

    if (d_row_ptr[node_i + 1] == d_row_ptr[node_i]) [[unlikely]] {
        for (int v = lane; v < TILES; v += kWarpSize) {
            Tile::write_zero(h_out_base, v);
        }
        if (lane == 0) {
            d_logsumexp_out[(int64_t)node_i * H + head_h] = -INFINITY;
        }
        return;
    }

    const int lo = node_chunk_offset[slot];
    const int hi = node_chunk_offset[slot + 1];

    accum_t global_max = -FLT_MAX;
    for (int s = lo; s < hi; ++s) {
        global_max = AccumOps::max(global_max, part_ml[((int64_t)s * H + head_h) * 2 + 0]);
    }
    accum_t global_sum{};
    for (int s = lo; s < hi; ++s) {
        const int64_t b = ((int64_t)s * H + head_h) * 2;
        global_sum      = AccumOps::fma(part_ml[b + 1], AccumOps::exp(part_ml[b + 0] - global_max), global_sum);
    }
    const accum_t inv_sum = (global_sum > 0.0f) ? (accum_t{1} / global_sum) : accum_t{};

    if (lane == 0) {
        d_logsumexp_out[(int64_t)node_i * H + head_h] =
            (global_sum > 0.0f) ? (global_max + AccumOps::log(global_sum)) : -INFINITY;
    }

    for (int v = lane; v < TILES; v += kWarpSize) {
        accum_t combined[TW];
#pragma unroll
        for (int ep = 0; ep < TW; ++ep) {
            combined[ep] = accum_t{};
        }
        for (int s = lo; s < hi; ++s) {
            const accum_t sc         = AccumOps::exp(part_ml[((int64_t)s * H + head_h) * 2 + 0] - global_max);
            const accum_t *const o_s = part_o + ((int64_t)s * H + head_h) * D_CONST;
#pragma unroll
            for (int ep = 0; ep < TW; ++ep) {
                combined[ep] = AccumOps::fma(sc, o_s[v * TW + ep], combined[ep]);
            }
        }
#pragma unroll
        for (int ep = 0; ep < TW; ++ep) {
            combined[ep] *= inv_sum;
        }
        Tile::write_convert_from_accum(&h_out_base[v * TW], combined);
    }
}
