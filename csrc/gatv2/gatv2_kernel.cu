#include "common.cuh"

// =============================================================================
// GATv2 Kernel with CSR Graph Format
// =============================================================================

template<int WARPS_PER_BLOCK, int D_CONST, bool USE_PIPELINE, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kMaxThreadsInWarp)
GATv2Forward_Kernel(
    size_t N,
    size_t H,
    size_t D,
    const cuda_t* __restrict__ d_l,
    const cuda_t* __restrict__ d_r,
    int64_t stride_l_n,
    int64_t stride_l_h,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const index_t* __restrict__ d_row_ptr,
    const index_t* __restrict__ d_col_idx,
    const index_t* __restrict__ node_indices,   // node indirection
    const cuda_t* __restrict__ d_attn_vec,
    cuda_t* __restrict__ d_h_out,
    float* __restrict__ d_logsumexp_out,
    float negative_slope
) {
    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    constexpr int EPV            = Tile::ELEM_PER_VEC;
    constexpr int NUM_VECS       = D_CONST / EPV;
    constexpr int VECS_PER_LANE  = (NUM_VECS + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;
    constexpr int ACCS_PER_LANE  = VECS_PER_LANE * EPV;

    // only meaningful when USE_PIPELINE
    constexpr int R_BYTES        = D_CONST * (int)sizeof(cuda_t);
    constexpr int F4_PER_R       = R_BYTES / 16;
    constexpr int NUM_STAGES     = 2;

    const int node_i = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h = blockIdx.y;
    const int warp_id = threadIdx.x / kMaxThreadsInWarp;
    const int lane    = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= (int)N || head_h >= (int)H) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    cuda_t* h_out_base = d_h_out + ((int64_t)node_i * H + head_h) * D_CONST;

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < NUM_VECS; v += kMaxThreadsInWarp) {
                Tile::write_zero(h_out_base, v);
            }
            if (lane == 0) {
                d_logsumexp_out[(int64_t)node_i * H + head_h] = -INFINITY;
            }
        }
        return;
    }

    const cuda_t* l_base = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t* a_base = d_attn_vec + head_h * D_CONST;

    // Shared memory layout:
    //   l_sh:      D_CONST * sizeof(cuda_t)                        -- read-only
    //   r_dbuf:    WARPS_PER_BLOCK * 2 * D_CONST * sizeof(cuda_t)  -- per-warp ping-pong for async r[j]
    //   warp_out:  WARPS_PER_BLOCK * D_CONST * sizeof(float)       -- per-warp output accum
    //   warp_max:  WARPS_PER_BLOCK * sizeof(float)                 -- per-warp softmax max
    //   warp_sum:  WARPS_PER_BLOCK * sizeof(float)                 -- per-warp softmax sum_exp
    extern __shared__ char sh_raw[];
    cuda_t* l_sh     = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* r_dbuf   = l_sh + D_CONST; // only meaningful when USE_PIPELINE
    float*  warp_out = USE_PIPELINE ? reinterpret_cast<float*>(r_dbuf + WARPS_PER_BLOCK * 2 * D_CONST) : reinterpret_cast<float*>(l_sh + D_CONST);
    float*  warp_max = warp_out + WARPS_PER_BLOCK * D_CONST;
    float*  warp_sum = warp_max + WARPS_PER_BLOCK;

    float* my_out = warp_out + warp_id * D_CONST;

    cuda_t* r0_ptr = r_dbuf + warp_id * 2 * D_CONST;
    cuda_t* r1_ptr = r0_ptr + D_CONST;

    // Cooperative load of l into shared memory using all threads
    {
        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4* l_src4 = reinterpret_cast<const float4*>(l_base);
        float4* l_sh4 = reinterpret_cast<float4*>(l_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kMaxThreadsInWarp) {
            l_sh4[i] = l_src4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    // Per-warp register accumulators
    float h_acc[ACCS_PER_LANE];
    #pragma unroll
    for (int i = 0; i < ACCS_PER_LANE; ++i) {
        h_acc[i] = 0.f;
    }

    OnlineSoftmaxState softmax_state;

    if constexpr (USE_PIPELINE) {
        int loop_iters = (num_neighbors > warp_id) ? (num_neighbors - warp_id + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK : 0;

        if (loop_iters > 0) {
            cuda::pipeline<cuda::thread_scope_thread> pipe = cuda::make_pipeline();

            auto async_copy_r = [&](cuda_t* dst, const cuda_t* src) {
                for (int i = lane; i < F4_PER_R; i += kMaxThreadsInWarp) {
                    cuda::memcpy_async(reinterpret_cast<char*>(dst) + i * 16, reinterpret_cast<const char*>(src) + i * 16, cuda::aligned_size_t<16>(16), pipe);
                }
            };

            {
                index_t j0 = d_col_idx[edge_start + static_cast<index_t>(warp_id)];
                const cuda_t* r0_src = d_r + j0 * stride_r_n + head_h * stride_r_h;
                pipe.producer_acquire();
                async_copy_r(r0_ptr, r0_src);
                pipe.producer_commit();
            }

            for (int iter = 0; iter < loop_iters; ++iter) {
                int k = warp_id + iter * WARPS_PER_BLOCK;
                cuda_t* r_cur  = (iter & 1) ? r1_ptr : r0_ptr;
                cuda_t* r_next = (iter & 1) ? r0_ptr : r1_ptr;

                pipe.producer_acquire();
                int k_next = k + WARPS_PER_BLOCK;
                if (k_next < num_neighbors) {
                    index_t j_next = d_col_idx[edge_start + static_cast<index_t>(k_next)];
                    const cuda_t* r_next_src = d_r + j_next * stride_r_n + head_h * stride_r_h;
                    async_copy_r(r_next, r_next_src);
                }
                pipe.producer_commit();

                cuda::pipeline_consumer_wait_prior<NUM_STAGES - 1>(pipe);

                float dot_lane = 0.f;
                #pragma unroll
                for (int t = 0; t < VECS_PER_LANE; ++t) {
                    int v = lane + kMaxThreadsInWarp * t;
                    if (v < NUM_VECS) {
                        vec_t lv = Tile::load(l_sh, v);
                        vec_t rv = Tile::load(r_cur, v);
                        vec_t av = Tile::load(a_base, v);
                        dot_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                    }
                }
                float dot = warp_reduce_sum(dot_lane);

                float rescale = softmax_state.update(dot);
                #pragma unroll
                for (int i = 0; i < ACCS_PER_LANE; ++i) h_acc[i] *= rescale;

                float contrib = __expf(dot - softmax_state.max_val);
                #pragma unroll
                for (int t = 0; t < VECS_PER_LANE; ++t) {
                    int v = lane + kMaxThreadsInWarp * t;
                    if (v < NUM_VECS) {
                        vec_t rv = Tile::load(r_cur, v);
                        Tile::weighted_accum(&h_acc[t * EPV], contrib, rv);
                    }
                }
                pipe.consumer_release();
            }
        }
    } else {
        // Warp-strided neighbor loop
        for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
            index_t neighbor_j = d_col_idx[edge_start + static_cast<index_t>(k)];
            const cuda_t* r_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

            float dot_lane = 0.f;
            #pragma unroll
            for (int t = 0; t < VECS_PER_LANE; ++t) {
                int v = lane + kMaxThreadsInWarp * t;
                if (v < NUM_VECS) {
                    vec_t lv = Tile::load(l_sh, v);
                    vec_t rv = Tile::load(r_base, v);
                    vec_t av = Tile::load(a_base, v);
                    dot_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                }
            }
            float dot = warp_reduce_sum(dot_lane);

            float rescale = softmax_state.update(dot);
            #pragma unroll
            for (int i = 0; i < ACCS_PER_LANE; ++i) {
                h_acc[i] *= rescale;
            }

            float contrib = __expf(dot - softmax_state.max_val);
            #pragma unroll
            for (int t = 0; t < VECS_PER_LANE; ++t) {
                int v = lane + kMaxThreadsInWarp * t;
                if (v < NUM_VECS) {
                    vec_t rv = Tile::load(r_base, v);
                    Tile::weighted_accum(&h_acc[t * EPV], contrib, rv);
                }
            }
        }
    }

    // Write per-warp results to shared memory
    #pragma unroll
    for (int t = 0; t < VECS_PER_LANE; ++t) {
        int v = lane + kMaxThreadsInWarp * t;
        if (v < NUM_VECS) {
            Tile::write_float(my_out, v, &h_acc[t * EPV]);
        }
    }

    if (lane == 0) {
        warp_max[warp_id] = softmax_state.max_val;
        warp_sum[warp_id] = softmax_state.sum_exp;
    }
    __syncthreads();

    // Cross-warp online-softmax reduction (warp 0 only)
    if (warp_id == 0) {
        float global_max = -FLT_MAX;
        float global_sum = 0.0f;
        float inv_sum    = 0.0f;

        if (lane == 0) {
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_max = fmaxf(global_max, warp_max[w]);
            }
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                global_sum = fmaf(warp_sum[w], __expf(warp_max[w] - global_max), global_sum);
            }
            #pragma unroll
            for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                warp_sum[w] = __expf(warp_max[w] - global_max);
            }
            inv_sum = (global_sum > 0.0f) ? (1.0f / global_sum) : 0.0f;
            d_logsumexp_out[(int64_t)node_i * H + head_h] =
                (global_sum > 0.0f) ? (global_max + logf(global_sum)) : -INFINITY;
        }

        inv_sum = __shfl_sync(FULL_WARP_MASK, inv_sum, 0);

        // Combine all warps' outputs with proper rescaling
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                float combined[EPV];
                #pragma unroll
                for (int ep = 0; ep < EPV; ++ep) {
                    combined[ep] = 0.0f;
                    int d_idx = v * EPV + ep;
                    #pragma unroll
                    for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                        combined[ep] = fmaf(warp_sum[w], warp_out[w * D_CONST + d_idx], combined[ep]);
                    }
                    combined[ep] *= inv_sum;
                }
                Tile::write_typed(h_out_base, v, combined);
            }
        }
    }
}

// =============================================================================
// Unified GATv2 Backward AL kernel (computes grad_a, grad_l, G)
// =============================================================================
template<int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kMaxThreadsInWarp)
GATv2Backward_AL(
    size_t N, size_t H, size_t D,
    const cuda_t* __restrict__ grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,
    const cuda_t* __restrict__ d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,
    const cuda_t* __restrict__ d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const index_t* __restrict__ d_row_ptr,
    const index_t* __restrict__ d_col_idx,
    const index_t* __restrict__ node_indices,   // node indirection
    const cuda_t* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,   // [N, H]
    float negative_slope,
    float* __restrict__ grad_a,  // [N, H, D] always float32
    cuda_t* __restrict__ grad_l, // [N, H, D]
    float* __restrict__ d_G      // [N, H]
) {
    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    constexpr int EPV            = Tile::ELEM_PER_VEC;
    constexpr int NUM_VECS       = D_CONST / EPV;
    constexpr int VECS_PER_LANE  = (NUM_VECS + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;

    const int node_i = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h = blockIdx.y;
    const int warp_id = threadIdx.x / kMaxThreadsInWarp;
    const int lane    = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= (int)N || head_h >= (int)H) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    // Shared memory layout:
    //   li_sh:      D_CONST * sizeof(cuda_t)                       -- read-only
    //   ghi_sh:     D_CONST * sizeof(cuda_t)                       -- read-only
    //   warp_grada: WARPS_PER_BLOCK * D_CONST * sizeof(float)      -- per-warp
    //   warp_gradl: WARPS_PER_BLOCK * D_CONST * sizeof(float)      -- per-warp
    //   warp_G:     WARPS_PER_BLOCK * sizeof(float)                -- per-warp G partial
    //   G_broadcast: sizeof(float)                                  -- broadcast slot
    extern __shared__ char sh_raw[];
    cuda_t* li_sh      = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* ghi_sh     = li_sh + D_CONST;
    float*  warp_grada = reinterpret_cast<float*>(ghi_sh + D_CONST);
    float*  warp_gradl = warp_grada + WARPS_PER_BLOCK * D_CONST;
    float*  warp_G     = warp_gradl + WARPS_PER_BLOCK * D_CONST;
    float*  G_broadcast = warp_G + WARPS_PER_BLOCK;

    float* my_grada = warp_grada + warp_id * D_CONST;
    float* my_gradl = warp_gradl + warp_id * D_CONST;

    cuda_t* grad_l_base = grad_l + ((int64_t)(node_i * H + head_h) * D_CONST);
    float*  grad_a_base = grad_a + ((int64_t)(node_i * H + head_h) * D_CONST);

    // handle isolated nodes
    if (num_neighbors == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < NUM_VECS; v += kMaxThreadsInWarp) {
                Tile::write_zero(grad_l_base, v);
            }
            constexpr int f4_count_f = D_CONST / 4;
            float4* ga_f4 = reinterpret_cast<float4*>(grad_a_base);
            for (int i = lane; i < f4_count_f; i += kMaxThreadsInWarp) {
                ga_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            }
            if (lane == 0) {
                d_G[node_i * H + head_h] = 0.f;
            }
        }
        return;
    }

    float L_i = d_logsumexp[node_i * H + head_h];

    const cuda_t* li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t* ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const cuda_t* a_base   = d_attn_vec + head_h * D_CONST;

    // Zero per-warp accumulators and cooperatively load li, ghi
    {
        constexpr int f4_count_f = D_CONST / 4;
        float4* my_grada_f4 = reinterpret_cast<float4*>(my_grada);
        float4* my_gradl_f4 = reinterpret_cast<float4*>(my_gradl);
        for (int i = lane; i < f4_count_f; i += kMaxThreadsInWarp) {
            my_grada_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            my_gradl_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4* li_src_f4  = reinterpret_cast<const float4*>(li_base);
        const float4* ghi_src_f4 = reinterpret_cast<const float4*>(ghi_base);
        float4* li_sh_f4  = reinterpret_cast<float4*>(li_sh);
        float4* ghi_sh_f4 = reinterpret_cast<float4*>(ghi_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kMaxThreadsInWarp) {
            li_sh_f4[i]  = li_src_f4[i];
            ghi_sh_f4[i] = ghi_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    // pass 1: compute G_{i,h} = sum_j alpha_ij * <grad_h_i, r_j> (warp-strided)
    float G_partial = 0.f;

    for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
        index_t neighbor_j = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t* rj_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij = recompute_alpha(e_ij, L_i);
        G_partial = fmaf(alpha_ij, p_ij, G_partial);
    }

    // Cross-warp reduction for G
    if (lane == 0) warp_G[warp_id] = G_partial;
    __syncthreads();

    float G_i_h = 0.f;
    if (warp_id == 0 && lane == 0) {
        for (int w = 0; w < WARPS_PER_BLOCK; ++w)
            G_i_h += warp_G[w];
        *G_broadcast = G_i_h;
        d_G[node_i * H + head_h] = G_i_h;
    }
    __syncthreads();
    G_i_h = *G_broadcast;

    // pass 2: accumulate gradients (warp-strided)
    for (int k = warp_id; k < num_neighbors; k += WARPS_PER_BLOCK) {
        index_t neighbor_j = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t* rj_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij  = recompute_alpha(e_ij, L_i);
        float grad_e_ij = alpha_ij * (p_ij - G_i_h);

        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv = Tile::load(li_sh, v);
                vec_t rv = Tile::load(rj_base, v);
                vec_t av = Tile::load(a_base, v);
                int base_f = v * EPV;
                Tile::gatv2_accum_grad_al(&my_grada[base_f], &my_gradl[base_f],
                                    grad_e_ij, lv, rv, av, negative_slope);
            }
        }
    }

    // Cross-warp reduction: warp 0 sums all per-warp accumulators
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                int base_f = v * EPV;
                float ga_sum[EPV];
                float gl_sum[EPV];
                #pragma unroll
                for (int ep = 0; ep < EPV; ++ep) {
                    ga_sum[ep] = 0.f;
                    gl_sum[ep] = 0.f;
                }
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                    #pragma unroll
                    for (int ep = 0; ep < EPV; ++ep) {
                        ga_sum[ep] += warp_grada[w * D_CONST + base_f + ep];
                        gl_sum[ep] += warp_gradl[w * D_CONST + base_f + ep];
                    }
                }
                Tile::write_typed(grad_l_base, v, gl_sum);
                Tile::write_float(grad_a_base, v, ga_sum);
            }
        }
    }
}

// =============================================================================
// Unified GATv2 Backward R kernel (computes grad_r)
// =============================================================================
template<int WARPS_PER_BLOCK, int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(WARPS_PER_BLOCK * kMaxThreadsInWarp)
GATv2Backward_R(
    size_t N, size_t H, size_t D,
    const cuda_t* __restrict__ grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,
    const cuda_t* __restrict__ d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,
    const cuda_t* __restrict__ d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,
    const index_t* __restrict__ d_row_ptr_T,
    const index_t* __restrict__ d_col_idx_T,
    const index_t* __restrict__ node_indices,   // node indirection
    const cuda_t* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,   // [N, H]
    const float* __restrict__ d_G,           // [N, H]
    float negative_slope,
    cuda_t* __restrict__ grad_r              // [N, H, D]
) {
    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    constexpr int EPV            = Tile::ELEM_PER_VEC;
    constexpr int NUM_VECS       = D_CONST / EPV;
    constexpr int VECS_PER_LANE  = (NUM_VECS + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;

    const int node_j = static_cast<int>(node_indices[blockIdx.x]);
    const int head_h = blockIdx.y;
    const int warp_id = threadIdx.x / kMaxThreadsInWarp;
    const int lane    = threadIdx.x % kMaxThreadsInWarp;

    if (node_j >= (int)N || head_h >= (int)H) return;

    index_t edge_start = d_row_ptr_T[node_j];
    index_t edge_end   = d_row_ptr_T[node_j + 1];
    int num_incoming   = static_cast<int>(edge_end - edge_start);

    // Shared memory layout:
    //   rj_sh:       D_CONST * sizeof(cuda_t)                      -- read-only
    //   warp_gradr:  WARPS_PER_BLOCK * D_CONST * sizeof(float)     -- per-warp
    extern __shared__ char sh_raw[];
    cuda_t* rj_sh      = reinterpret_cast<cuda_t*>(sh_raw);
    float*  warp_gradr  = reinterpret_cast<float*>(rj_sh + D_CONST);

    float* my_gradr = warp_gradr + warp_id * D_CONST;

    cuda_t* grad_r_base = grad_r + ((int64_t)(node_j * H + head_h) * D_CONST);

    // Handle isolated nodes
    if (num_incoming == 0) {
        if (warp_id == 0) {
            for (int v = lane; v < NUM_VECS; v += kMaxThreadsInWarp) {
                Tile::write_zero(grad_r_base, v);
            }
        }
        return;
    }

    const cuda_t* rj_base = d_r + node_j * stride_r_n + head_h * stride_r_h;
    const cuda_t* a_base  = d_attn_vec + head_h * D_CONST;

    // Zero per-warp accumulators and cooperatively load rj
    {
        constexpr int f4_count_f = D_CONST / 4;
        float4* my_gradr_f4 = reinterpret_cast<float4*>(my_gradr);
        for (int i = lane; i < f4_count_f; i += kMaxThreadsInWarp) {
            my_gradr_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4* rj_src_f4 = reinterpret_cast<const float4*>(rj_base);
        float4* rj_sh_f4 = reinterpret_cast<float4*>(rj_sh);
        for (int i = threadIdx.x; i < f4_count; i += WARPS_PER_BLOCK * kMaxThreadsInWarp) {
            rj_sh_f4[i] = rj_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    // Warp-strided edge loop
    for (int idx = warp_id; idx < num_incoming; idx += WARPS_PER_BLOCK) {
        index_t node_i = d_col_idx_T[edge_start + static_cast<index_t>(idx)];
        const cuda_t* li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
        const cuda_t* ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;

        float L_i_h = d_logsumexp[node_i * H + head_h];
        float G_i_h = d_G[node_i * H + head_h];

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_base, v);
                vec_t rv  = Tile::load(rj_sh, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_base, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij  = recompute_alpha(e_ij, L_i_h);
        float grad_e_ij = alpha_ij * (p_ij - G_i_h);

        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_base, v);
                vec_t rv  = Tile::load(rj_sh, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_base, v);
                int base_f = v * EPV;
                Tile::gatv2_accum_grad_r(&my_gradr[base_f], alpha_ij, ghv,
                                   grad_e_ij, lv, rv, av, negative_slope);
            }
        }
    }

    // Cross-warp reduction: warp 0 sums per-warp accumulators
    __syncthreads();

    if (warp_id == 0) {
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                int base_f = v * EPV;
                float gr_sum[EPV];
                #pragma unroll
                for (int ep = 0; ep < EPV; ++ep)
                    gr_sum[ep] = 0.f;
                #pragma unroll
                for (int w = 0; w < WARPS_PER_BLOCK; ++w) {
                    #pragma unroll
                    for (int ep = 0; ep < EPV; ++ep)
                        gr_sum[ep] += warp_gradr[w * D_CONST + base_f + ep];
                }
                Tile::write_typed(grad_r_base, v, gr_sum);
            }
        }
    }
}


template<int grad_A_reduce_row_chunk_size, typename cuda_t>
__global__ void __launch_bounds__(kMaxThreadsInWarp * kMaxThreadsInWarp)
ReduceGradAKernel(
    size_t N, size_t H, size_t D,

    const float* __restrict__ grad_a,        // [N, H, D] always float32
    float* __restrict__ d_grad_a_reduced_out // [H, D] output in float32
){

    //head inbex
    int head_h = blockIdx.z; // 0..H-1

    // define feature chunk and node chunk to reduce
    int row_chunk_start     = grad_A_reduce_row_chunk_size * blockIdx.x;
    int feature_chunk_start = blockDim.y * blockIdx.y;


    // define thread-specific indices and feature locations
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int fx = feature_chunk_start + tx;


    // define shared memory chunk and accumulatur
    __shared__ float tile_reduce[kMaxThreadsInWarp][kMaxThreadsInWarp + 1];
    __shared__ float result_accum[kMaxThreadsInWarp];

    float accum = 0.0f;

    // looped logic across row chunks:
    const int row_chunk_end = min((int)N, (int)(row_chunk_start + grad_A_reduce_row_chunk_size));
    for (int base_row_offset = row_chunk_start; base_row_offset < row_chunk_end; base_row_offset += blockDim.y){

        int row_to_load = base_row_offset + ty; // node index
        if (row_to_load < (int)N && fx < (int)D && head_h < (int)H){
            // grad_a layout: [N, H, D] contiguous
            // idx = (n * H + h) * D + d
            size_t idx = ((size_t)row_to_load * H + (size_t)head_h) * D + (size_t)fx;

            tile_reduce[tx][ty] = grad_a[idx];
        } else {
            tile_reduce[tx][ty] = 0.0f;
        }

        __syncthreads();

        // transpose tile for warp-level reduction:
        //   * each warp (fixed ty) reduces over rows for one feature (fx)
        float value = tile_reduce[ty][tx];
        accum += warp_reduce_sum(value);

    }
    // each first lane in a warp write its results into the sshared memory for the first warp to finally reduce it into HBM:
    if (tx == 0){
        result_accum[ty] = accum;
    }

    __syncthreads();
    // now  threads with ty==0 and tx selecting feature within chunk
    // write out the final reduced result

    if (ty == 0 && fx < (int)D && head_h < (int)H) {
        // output layout: [H, D] contiguous
        size_t out_idx = (size_t)head_h * D + (size_t)fx;
        atomicAdd(d_grad_a_reduced_out + out_idx, result_accum[tx]);
    }
}

// =============================================================================
// Undirected GATv2 backward: G computation kernel (extracts pass 1 of AL)
// =============================================================================
template<int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kMaxThreadsInWarp)
GATv2Backward_G_Kernel(
    size_t N, size_t H, size_t D,
    const cuda_t* __restrict__ grad_h,
    int64_t stride_gh_n, int64_t stride_gh_h,
    const cuda_t* __restrict__ d_l,
    int64_t stride_l_n, int64_t stride_l_h,
    const cuda_t* __restrict__ d_r,
    int64_t stride_r_n, int64_t stride_r_h,
    const index_t* __restrict__ d_row_ptr,
    const index_t* __restrict__ d_col_idx,
    const cuda_t* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,   // [N, H]
    float negative_slope,
    float* __restrict__ d_G                  // [N, H] output
) {
    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    constexpr int NUM_VECS       = D_CONST / Tile::ELEM_PER_VEC;
    constexpr int VECS_PER_LANE  = (NUM_VECS + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= (int)N || head_h >= (int)H) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    if (num_neighbors == 0) {
        if (lane == 0) d_G[node_i * H + head_h] = 0.f;
        return;
    }

    float L_i = d_logsumexp[node_i * H + head_h];

    // Shared memory: li_sh + ghi_sh
    extern __shared__ char sh_raw[];
    cuda_t* li_sh  = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* ghi_sh = li_sh + D_CONST;

    const cuda_t* li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t* ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const cuda_t* a_base   = d_attn_vec + head_h * D_CONST;

    // Load li, ghi via 128-bit transactions
    {
        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4* li_src_f4  = reinterpret_cast<const float4*>(li_base);
        const float4* ghi_src_f4 = reinterpret_cast<const float4*>(ghi_base);
        float4* li_sh_f4  = reinterpret_cast<float4*>(li_sh);
        float4* ghi_sh_f4 = reinterpret_cast<float4*>(ghi_sh);
        for (int i = lane; i < f4_count; i += kMaxThreadsInWarp) {
            li_sh_f4[i]  = li_src_f4[i];
            ghi_sh_f4[i] = ghi_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    float G_i_h = 0.f;
    for (int k = 0; k < num_neighbors; ++k) {
        index_t neighbor_j = d_col_idx[edge_start + static_cast<index_t>(k)];
        const cuda_t* rj_base = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;

        float e_lane = 0.f;
        float p_lane = 0.f;
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);
                e_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_lane += Tile::dot_product(ghv, rv);
            }
        }
        float e_ij = warp_reduce_sum(e_lane);
        float p_ij = warp_reduce_sum(p_lane);

        float alpha_ij = recompute_alpha(e_ij, L_i);
        G_i_h = fmaf(alpha_ij, p_ij, G_i_h);
    }

    if (lane == 0) {
        d_G[node_i * H + head_h] = G_i_h;
    }
}


// =============================================================================
// Undirected GATv2 backward: fused ALR kernel
// Computes grad_a[i], grad_l[i] (forward direction) and grad_r[i] (reverse
// direction) in a single pass over forward CSR neighbors.
// Requires G[j] to be pre-computed globally.
// =============================================================================
template<int D_CONST, typename cuda_t, typename index_t>
__global__ void __launch_bounds__(kMaxThreadsInWarp)
GATv2Backward_ALR_Undirected(
    size_t N, size_t H, size_t D,
    const cuda_t* __restrict__ grad_h,
    int64_t stride_gh_n, int64_t stride_gh_h,
    const cuda_t* __restrict__ d_l,
    int64_t stride_l_n, int64_t stride_l_h,
    const cuda_t* __restrict__ d_r,
    int64_t stride_r_n, int64_t stride_r_h,
    const index_t* __restrict__ d_row_ptr,
    const index_t* __restrict__ d_col_idx,
    const cuda_t* __restrict__ d_attn_vec,   // [H, D]
    const float* __restrict__ d_logsumexp,   // [N, H]
    const float* __restrict__ d_G,           // [N, H] (pre-computed)
    float negative_slope,
    float* __restrict__ grad_a,              // [N, H, D] always float32
    cuda_t* __restrict__ grad_l,             // [N, H, D]
    cuda_t* __restrict__ grad_r              // [N, H, D]
) {
    constexpr int VW = SelectVW<D_CONST, cuda_t>::value;
    using Tile = TileOps<VW, cuda_t>;
    using vec_t = typename Tile::vec_t;
    using ns_t  = typename Tile::ns_t;

    constexpr int NUM_VECS       = D_CONST / Tile::ELEM_PER_VEC;
    constexpr int VECS_PER_LANE  = (NUM_VECS + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp;

    int node_i = blockIdx.x;
    int head_h = blockIdx.y;
    int lane   = threadIdx.x % kMaxThreadsInWarp;

    if (node_i >= (int)N || head_h >= (int)H) return;

    index_t edge_start = d_row_ptr[node_i];
    index_t edge_end   = d_row_ptr[node_i + 1];
    int num_neighbors  = static_cast<int>(edge_end - edge_start);

    // Shared memory layout:
    //   li_sh:      D_CONST * sizeof(cuda_t)   -- l[i]
    //   ri_sh:      D_CONST * sizeof(cuda_t)   -- r[i]
    //   ghi_sh:     D_CONST * sizeof(cuda_t)   -- grad_h[i]
    //   grada_sh:   D_CONST * sizeof(float)    -- accumulator for grad_a[i]
    //   gradli_sh:  D_CONST * sizeof(float)    -- accumulator for grad_l[i]
    //   gradri_sh:  D_CONST * sizeof(float)    -- accumulator for grad_r[i]
    extern __shared__ char sh_raw[];
    cuda_t* li_sh      = reinterpret_cast<cuda_t*>(sh_raw);
    cuda_t* ri_sh      = li_sh + D_CONST;
    cuda_t* ghi_sh     = ri_sh + D_CONST;
    float*  grada_sh   = reinterpret_cast<float*>(ghi_sh + D_CONST);
    float*  gradli_sh  = grada_sh + D_CONST;
    float*  gradri_sh  = gradli_sh + D_CONST;

    cuda_t* grad_l_base = grad_l + ((int64_t)(node_i * H + head_h) * D_CONST);
    cuda_t* grad_r_base = grad_r + ((int64_t)(node_i * H + head_h) * D_CONST);
    float*  grad_a_base = grad_a + ((int64_t)(node_i * H + head_h) * D_CONST);

    // Handle isolated nodes: write zeros
    if (num_neighbors == 0) {
        for (int v = lane; v < NUM_VECS; v += kMaxThreadsInWarp) {
            Tile::write_zero(grad_l_base, v);
            Tile::write_zero(grad_r_base, v);
        }
        constexpr int f4_count_f = D_CONST / 4;
        float4* ga_f4 = reinterpret_cast<float4*>(grad_a_base);
        for (int i = lane; i < f4_count_f; i += kMaxThreadsInWarp) {
            ga_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }
        return;
    }

    float L_i   = d_logsumexp[node_i * H + head_h];
    float G_i_h = d_G[node_i * H + head_h];

    const cuda_t* li_base  = d_l + node_i * stride_l_n + head_h * stride_l_h;
    const cuda_t* ri_base  = d_r + node_i * stride_r_n + head_h * stride_r_h;
    const cuda_t* ghi_base = grad_h + node_i * stride_gh_n + head_h * stride_gh_h;
    const cuda_t* a_base   = d_attn_vec + head_h * D_CONST;

    // Zero accumulators and load li, ri, ghi via 128-bit transactions
    {
        constexpr int f4_count_f = D_CONST / 4;
        float4* grada_f4  = reinterpret_cast<float4*>(grada_sh);
        float4* gradli_f4 = reinterpret_cast<float4*>(gradli_sh);
        float4* gradri_f4 = reinterpret_cast<float4*>(gradri_sh);
        for (int i = lane; i < f4_count_f; i += kMaxThreadsInWarp) {
            grada_f4[i]  = make_float4(0.f, 0.f, 0.f, 0.f);
            gradli_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
            gradri_f4[i] = make_float4(0.f, 0.f, 0.f, 0.f);
        }

        constexpr int f4_count = (D_CONST * (int)sizeof(cuda_t)) / 16;
        const float4* li_src_f4  = reinterpret_cast<const float4*>(li_base);
        const float4* ri_src_f4  = reinterpret_cast<const float4*>(ri_base);
        const float4* ghi_src_f4 = reinterpret_cast<const float4*>(ghi_base);
        float4* li_sh_f4  = reinterpret_cast<float4*>(li_sh);
        float4* ri_sh_f4  = reinterpret_cast<float4*>(ri_sh);
        float4* ghi_sh_f4 = reinterpret_cast<float4*>(ghi_sh);
        for (int i = lane; i < f4_count; i += kMaxThreadsInWarp) {
            li_sh_f4[i]  = li_src_f4[i];
            ri_sh_f4[i]  = ri_src_f4[i];
            ghi_sh_f4[i] = ghi_src_f4[i];
        }
    }
    __syncthreads();

    ns_t ns = Tile::make_ns(negative_slope);

    for (int k = 0; k < num_neighbors; ++k) {
        index_t neighbor_j = d_col_idx[edge_start + static_cast<index_t>(k)];

        const cuda_t* rj_base  = d_r + neighbor_j * stride_r_n + head_h * stride_r_h;
        const cuda_t* lj_base  = d_l + neighbor_j * stride_l_n + head_h * stride_l_h;
        const cuda_t* ghj_base = grad_h + neighbor_j * stride_gh_n + head_h * stride_gh_h;

        // ── Forward direction: score(i,j) = a^T . LeakyReLU(l[i] + r[j]) ──
        float e_fwd_lane = 0.f;
        float p_fwd_lane = 0.f;
        // ── Reverse direction: score(j,i) = a^T . LeakyReLU(l[j] + r[i]) ──
        float e_rev_lane = 0.f;
        float p_rev_lane = 0.f;

        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_sh, v);   // l[i]
                vec_t rv  = Tile::load(rj_base, v);  // r[j]
                vec_t av  = Tile::load(a_base, v);
                vec_t ghv = Tile::load(ghi_sh, v);   // grad_h[i]

                vec_t ljv = Tile::load(lj_base, v);  // l[j]
                vec_t riv = Tile::load(ri_sh, v);    // r[i]
                vec_t ghjv = Tile::load(ghj_base, v); // grad_h[j]

                // Forward: e(i,j) and <grad_h[i], r[j]>
                e_fwd_lane += Tile::gatv2_dot_leaky_relu(lv, rv, av, ns);
                p_fwd_lane += Tile::dot_product(ghv, rv);

                // Reverse: e(j,i) and <grad_h[j], r[i]>
                e_rev_lane += Tile::gatv2_dot_leaky_relu(ljv, riv, av, ns);
                p_rev_lane += Tile::dot_product(ghjv, riv);
            }
        }

        float e_fwd = warp_reduce_sum(e_fwd_lane);
        float p_fwd = warp_reduce_sum(p_fwd_lane);
        float e_rev = warp_reduce_sum(e_rev_lane);
        float p_rev = warp_reduce_sum(p_rev_lane);

        // Forward: alpha(i,j), grad_e(i,j)
        float alpha_fwd  = recompute_alpha(e_fwd, L_i);
        float grad_e_fwd = alpha_fwd * (p_fwd - G_i_h);

        // Reverse: alpha(j,i), grad_e(j,i) — uses L[j] and G[j]
        float L_j   = d_logsumexp[neighbor_j * H + head_h];
        float G_j_h = d_G[neighbor_j * H + head_h];
        float alpha_rev  = recompute_alpha(e_rev, L_j);
        float grad_e_rev = alpha_rev * (p_rev - G_j_h);

        // Accumulate gradients
        #pragma unroll
        for (int t = 0; t < VECS_PER_LANE; ++t) {
            int v = lane + kMaxThreadsInWarp * t;
            if (v < NUM_VECS) {
                vec_t lv  = Tile::load(li_sh, v);
                vec_t rv  = Tile::load(rj_base, v);
                vec_t av  = Tile::load(a_base, v);
                int base_f = v * Tile::ELEM_PER_VEC;

                // Forward: grad_a[i], grad_l[i] from score(i,j)
                Tile::gatv2_accum_grad_al(&grada_sh[base_f], &gradli_sh[base_f],
                                          grad_e_fwd, lv, rv, av, negative_slope);

                // Reverse: grad_r[i] from score(j,i) = a^T . LeakyReLU(l[j] + r[i])
                vec_t ljv  = Tile::load(lj_base, v);
                vec_t riv  = Tile::load(ri_sh, v);
                vec_t ghjv = Tile::load(ghj_base, v);
                Tile::gatv2_accum_grad_r(&gradri_sh[base_f], alpha_rev, ghjv,
                                         grad_e_rev, ljv, riv, av, negative_slope);
            }
        }
    }

    __syncthreads();

    // Write grad_l (cuda_t), grad_a (float32), grad_r (cuda_t)
    #pragma unroll
    for (int t = 0; t < VECS_PER_LANE; ++t) {
        int v = lane + kMaxThreadsInWarp * t;
        if (v < NUM_VECS) {
            int base_f = v * Tile::ELEM_PER_VEC;
            Tile::write_typed(grad_l_base, v, &gradli_sh[base_f]);
            Tile::write_float(grad_a_base, v, &grada_sh[base_f]);
            Tile::write_typed(grad_r_base, v, &gradri_sh[base_f]);
        }
    }
}


// =============================================================================
// Undirected GATv2 backward impl: G kernel + fused ALR kernel + ReduceGradA
// =============================================================================
template<int D_CONST, typename cuda_t, typename index_t>
void GATv2Backward_CSR_Undirected_Impl(
    size_t N, size_t H, size_t D,
    const cuda_t* grad_h,
    int64_t stride_gh_n, int64_t stride_gh_h,
    const cuda_t* d_l,
    int64_t stride_l_n, int64_t stride_l_h,
    const cuda_t* d_r,
    int64_t stride_r_n, int64_t stride_r_h,
    const index_t* d_row_ptr,
    const index_t* d_col_idx,
    const cuda_t* d_attn_vec,
    const float* d_logsumexp,
    float negative_slope,
    int grad_A_reduce_row_chunk_size,
    cudaStream_t stream,
    cuda_t* grad_l,
    cuda_t* grad_r,
    float* grad_a,
    float* d_grad_a_reduced
) {
    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N, H);

    // 1) Compute G[i,h] for all nodes
    float* d_G;
    CUDA_CHECK(cudaMalloc(&d_G, N * H * sizeof(float)));

    // G kernel shared: li (cuda_t) + ghi (cuda_t)
    size_t sh_g = 2 * D_CONST * sizeof(cuda_t);

    GATv2Backward_G_Kernel<D_CONST, cuda_t, index_t><<<nBlocks, nThreads, sh_g, stream>>>(
        N, H, D, grad_h, stride_gh_n, stride_gh_h,
        d_l, stride_l_n, stride_l_h,
        d_r, stride_r_n, stride_r_h,
        d_row_ptr, d_col_idx,
        d_attn_vec, d_logsumexp, negative_slope,
        d_G);

    // 2) Fused ALR kernel: grad_a, grad_l, grad_r using forward CSR only
    // Shared: li + ri + ghi (cuda_t) + grada + gradli + gradri (float)
    size_t sh_alr = 3 * D_CONST * sizeof(cuda_t) + 3 * D_CONST * sizeof(float);

    GATv2Backward_ALR_Undirected<D_CONST, cuda_t, index_t><<<nBlocks, nThreads, sh_alr, stream>>>(
        N, H, D, grad_h, stride_gh_n, stride_gh_h,
        d_l, stride_l_n, stride_l_h,
        d_r, stride_r_n, stride_r_h,
        d_row_ptr, d_col_idx,
        d_attn_vec, d_logsumexp, d_G, negative_slope,
        grad_a, grad_l, grad_r);

    // 3) Reduce grad_a [N, H, D] -> [H, D]
    size_t shmem_gradA_reduce_size = (kMaxThreadsInWarp * (kMaxThreadsInWarp + 2)) * sizeof(float);
    dim3 grad_A_reduce_blockDim(kMaxThreadsInWarp, kMaxThreadsInWarp);

    std::visit([&](auto chunk_c) {
        constexpr int CHUNK = decltype(chunk_c)::value;
        dim3 grad_A_reduce_gridDim(
            (N + CHUNK - 1) / CHUNK,
            (D + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp,
            H
        );
        ReduceGradAKernel<CHUNK, cuda_t><<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(
            N, H, D, grad_a, d_grad_a_reduced
        );
    }, MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(grad_A_reduce_row_chunk_size));

    CUDA_CHECK(cudaFree(d_G));
}


// =============================================================================
// Launcher for backward pass (directed, with light/heavy node dispatch)
// =============================================================================

// Legacy impl kept for reference; actual dispatch is in gatv2_backward_cuda below.
template<int D_CONST, typename cuda_t, typename index_t>
void GATv2Backward_CSR_Impl_UNUSED(
    // inputs
    size_t N, size_t H, size_t D,

    const cuda_t* grad_h,
    int64_t stride_gh_n,
    int64_t stride_gh_h,

    const cuda_t* d_l,
    int64_t stride_l_n,
    int64_t stride_l_h,

    const cuda_t* d_r,
    int64_t stride_r_n,
    int64_t stride_r_h,

    const index_t* d_row_ptr,
    const index_t* d_col_idx,
    const index_t* d_row_ptr_T,
    const index_t* d_col_idx_T,
    const cuda_t* d_attn_vec,
    const float* d_logsumexp,  // [N, H]
    float negative_slope,
    int grad_A_reduce_row_chunk_size,
    cudaStream_t stream,

    // outputs
    cuda_t* grad_l,            // [N, H, D]
    cuda_t* grad_r,            // [N, H, D]
    float* grad_a,             // [N, H, D] always float32
    float* d_grad_a_reduced    // [H, D] output in float32
) {
    dim3 nThreads(kMaxThreadsInWarp);
    dim3 nBlocks(N, H);

    // G has shape [N, H]
    float* d_G;
    CUDA_CHECK(cudaMalloc(&d_G, N * H * sizeof(float)));

    // AL shared: li (cuda_t) + ghi (cuda_t) + grada (float) + gradli (float)
    size_t sh_al = 2 * D_CONST * sizeof(cuda_t) + 2 * D_CONST * sizeof(float);

    // 1: AL kernel - computes grad_a, grad_l, G
    GATv2Backward_AL<D_CONST, cuda_t, index_t><<<nBlocks, nThreads, sh_al, stream>>>(
        N, H, D, grad_h, stride_gh_n, stride_gh_h,
        d_l, stride_l_n, stride_l_h,
        d_r, stride_r_n, stride_r_h,
        d_row_ptr, d_col_idx,
        d_attn_vec, d_logsumexp, negative_slope,
        grad_a, grad_l, d_G);

    // R shared: rj (cuda_t) + gradr (float)
    size_t sh_r = D_CONST * sizeof(cuda_t) + D_CONST * sizeof(float);

    // 2: R kernel - computes grad_r
    GATv2Backward_R<D_CONST, cuda_t, index_t><<<nBlocks, nThreads, sh_r, stream>>>(
        N, H, D, grad_h, stride_gh_n, stride_gh_h,
        d_l, stride_l_n, stride_l_h,
        d_r, stride_r_n, stride_r_h,
        d_row_ptr_T, d_col_idx_T,
        d_attn_vec, d_logsumexp, d_G, negative_slope, grad_r);

    // 3: sum-reduce grad_a [N, H, D] over N into [H, D] (always float32)
    size_t shmem_gradA_reduce_size = (kMaxThreadsInWarp * (kMaxThreadsInWarp + 2)) * sizeof(float);
    dim3 grad_A_reduce_blockDim(kMaxThreadsInWarp, kMaxThreadsInWarp);

    std::visit([&](auto chunk_c) {
        constexpr int CHUNK = decltype(chunk_c)::value;
        dim3 grad_A_reduce_gridDim(
            (N + CHUNK - 1) / CHUNK,
            (D + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp,
            H
        );
        ReduceGradAKernel<CHUNK, cuda_t><<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(
            N, H, D, grad_a, d_grad_a_reduced
        );
    }, MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(grad_A_reduce_row_chunk_size));

    CUDA_CHECK(cudaFree(d_G));
}


std::vector<torch::Tensor> gatv2_forward_cuda(
    torch::Tensor l,              // [N, H, D] - left features
    torch::Tensor r,              // [N, H, D] - right features
    torch::Tensor row_ptr,        // [N+1] - CSR row pointers
    torch::Tensor col_idx,        // [E] - CSR column indices
    torch::Tensor attn_vec,       // [H, D] - contiguous attention vector
    float negative_slope,
    torch::Tensor light_nodes,
    torch::Tensor heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool use_pipeline
) {

    TORCH_CHECK(l.is_cuda() && r.is_cuda(), "l, r must be CUDA");
    TORCH_CHECK(l.dim() == 3 && r.dim() == 3, "l, r must be [N, H, D]");
    TORCH_CHECK(l.sizes() == r.sizes(), "l, r sizes must match");

    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(attn_vec.is_cuda(), "attn_vec must be a CUDA tensor");

    TORCH_CHECK(l.size(0) == r.size(0), "l and r must have same number of nodes");
    TORCH_CHECK(l.size(1) == r.size(1), "l and r must have same head dimension");
    TORCH_CHECK(l.size(2) == r.size(2), "l and r must have same feature dimension");
    TORCH_CHECK(l.size(2) == attn_vec.size(1), "attn_vec dimension must match features");

    auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype),
                "row_ptr must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");

    TORCH_CHECK(l.dtype() == r.dtype() && l.dtype() == attn_vec.dtype(),
                "l, r, and attn_vec must have the same dtype");
    TORCH_CHECK(l.dtype() == torch::kFloat32 || l.dtype() == torch::kFloat16 || l.dtype() == torch::kBFloat16,
                "l must be float32, float16, or bfloat16");

    const int64_t N = l.size(0);
    const int64_t H = l.size(1);
    const int64_t D = l.size(2);

    TORCH_CHECK(attn_vec.dim() == 2, "attn_vec must be [H, D]");
    TORCH_CHECK(attn_vec.size(0) == H, "attn_vec H mismatch");
    TORCH_CHECK(attn_vec.size(1) == D, "attn_vec D mismatch");
    TORCH_CHECK(D % 4 == 0, "head_dim (D) must be divisible by 4");

    TORCH_CHECK(attn_vec.is_contiguous(), "attn_vec must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");

    auto l_strides = l.strides();
    auto r_strides = r.strides();

    int64_t stride_l_n = l_strides[0];
    int64_t stride_l_h = l_strides[1];
    int64_t stride_l_d = l_strides[2];

    int64_t stride_r_n = r_strides[0];
    int64_t stride_r_h = r_strides[1];
    int64_t stride_r_d = r_strides[2];

    TORCH_CHECK(stride_l_d == 1 && stride_r_d == 1, "Feature dim (D) must be contiguous (stride_d == 1)");

    // h_out matches input dtype, logsumexp always float32
    torch::Tensor h_out     = torch::empty({N, H, D}, torch::TensorOptions().dtype(l.dtype()).device(l.device()));
    torch::Tensor logsumexp = torch::empty({N, H}, torch::TensorOptions().dtype(torch::kFloat32).device(l.device()));

    float* d_logsumexp = logsumexp.data_ptr<float>();
    cudaStream_t stream = 0;

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "GATv2 forward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    auto launch_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
        if (num_nodes_bucket == 0) return;

        std::visit([&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c, auto pipe_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t = typename decltype(typeInfo)::CudaType;
            constexpr int DC = decltype(d_c)::value;
            constexpr int W = decltype(warp_c)::value;
            constexpr bool PIPE = decltype(pipe_c)::value;

            auto* l_ptr     = reinterpret_cast<const cuda_t*>(l.data_ptr<torch_t>());
            auto* r_ptr     = reinterpret_cast<const cuda_t*>(r.data_ptr<torch_t>());
            auto* attn_ptr  = reinterpret_cast<const cuda_t*>(attn_vec.data_ptr<torch_t>());
            auto* h_out_ptr = reinterpret_cast<cuda_t*>(h_out.data_ptr<torch_t>());

            // l_sh + r_dbuf (if PIPE) + W * D float + 2 * W float
            size_t shmem = DC * sizeof(cuda_t)
             + (PIPE ? W * 2 * DC * sizeof(cuda_t) : 0)
             + W * DC * sizeof(float)
             + 2 * W * sizeof(float);

            dim3 blocks(num_nodes_bucket, H);
            dim3 threads(W * kMaxThreadsInWarp);

            GATv2Forward_Kernel<W, DC, PIPE, cuda_t, index_t><<<blocks, threads, shmem, stream>>>(
                N, H, DC, l_ptr, r_ptr, stride_l_n, stride_l_h, stride_r_n, stride_r_h,
                index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
                index_ptr<index_t>(node_indices),
                attn_ptr, h_out_ptr, d_logsumexp, negative_slope);
        }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
           MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
           MakeIntVariant<32, 64, 128, 256>((int)D),
           warp_variant,
           MakeBoolVariant<true, false>(use_pipeline));
    };

    launch_bucket(light_nodes, light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block));
    launch_bucket(heavy_nodes, heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block));

    CUDA_KERNEL_CHECK();

    return {h_out, logsumexp};
}


std::vector<torch::Tensor> gatv2_backward_cuda(
    torch::Tensor grad_h,         // [N, H, D] - gradient from output
    torch::Tensor l,              // [N, H, D] - left features (saved)
    torch::Tensor r,              // [N, H, D] - right features (saved)
    torch::Tensor row_ptr,        // [N+1] - CSR row pointers
    torch::Tensor col_idx,        // [E] - CSR column indices
    torch::Tensor row_ptr_T,      // [N+1] - CSR^T row pointers
    torch::Tensor col_idx_T,      // [E] - CSR^T column indices
    torch::Tensor attn_vec,       // [H, D] - attention vector (saved)
    torch::Tensor logsumexp,      // [N, H] - logsumexp (saved)
    float negative_slope,
    int grad_A_reduce_row_chunk_size,
    torch::Tensor fwd_light_nodes,
    torch::Tensor fwd_heavy_nodes,
    torch::Tensor bwd_light_nodes,
    torch::Tensor bwd_heavy_nodes,
    int light_warps_per_block,
    int heavy_warps_per_block,
    bool is_directed
) {
    TORCH_CHECK(grad_h.is_cuda(), "grad_h must be a CUDA tensor");
    TORCH_CHECK(l.is_cuda(), "l must be a CUDA tensor");
    TORCH_CHECK(r.is_cuda(), "r must be a CUDA tensor");
    TORCH_CHECK(row_ptr.is_cuda(), "row_ptr must be a CUDA tensor");
    TORCH_CHECK(col_idx.is_cuda(), "col_idx must be a CUDA tensor");
    TORCH_CHECK(row_ptr_T.is_cuda(), "row_ptr_T must be a CUDA tensor");
    TORCH_CHECK(col_idx_T.is_cuda(), "col_idx_T must be a CUDA tensor");
    TORCH_CHECK(attn_vec.is_cuda(), "attn_vec must be a CUDA tensor");
    TORCH_CHECK(logsumexp.is_cuda(), "logsumexp must be a CUDA tensor");

    TORCH_CHECK(grad_h.dtype() == l.dtype() && l.dtype() == r.dtype() && l.dtype() == attn_vec.dtype(),
                "grad_h, l, r, and attn_vec must have the same dtype");
    TORCH_CHECK(l.dtype() == torch::kFloat32 || l.dtype() == torch::kFloat16 || l.dtype() == torch::kBFloat16,
                "l must be float32, float16, or bfloat16");
    TORCH_CHECK(logsumexp.dtype() == torch::kFloat32, "logsumexp must be float32");
    auto idx_dtype = row_ptr.scalar_type();
    TORCH_CHECK(is_supported_index_type(idx_dtype),
                "index tensors must be int32, int64, uint32, or uint64");
    TORCH_CHECK(col_idx.scalar_type() == idx_dtype, "col_idx must have same dtype as row_ptr");
    TORCH_CHECK(row_ptr_T.scalar_type() == idx_dtype, "row_ptr_T must have same dtype as row_ptr");
    TORCH_CHECK(col_idx_T.scalar_type() == idx_dtype, "col_idx_T must have same dtype as row_ptr");

    TORCH_CHECK(attn_vec.is_contiguous(), "attn_vec must be contiguous");
    TORCH_CHECK(logsumexp.is_contiguous(), "logsumexp must be contiguous");
    TORCH_CHECK(row_ptr.is_contiguous(), "row_ptr must be contiguous");
    TORCH_CHECK(col_idx.is_contiguous(), "col_idx must be contiguous");
    TORCH_CHECK(row_ptr_T.is_contiguous(), "row_ptr_T must be contiguous");
    TORCH_CHECK(col_idx_T.is_contiguous(), "col_idx_T must be contiguous");

    TORCH_CHECK(grad_h.dim() == 3 && l.dim() == 3 && r.dim() == 3, "grad_h, l, r must be [N, H, D]");
    TORCH_CHECK(grad_h.sizes() == l.sizes() && l.sizes() == r.sizes(), "grad_h, l, r sizes must match");
    TORCH_CHECK(attn_vec.dim() == 2, "attn_vec must be [H, D]");
    TORCH_CHECK(logsumexp.dim() == 2, "logsumexp must be [N, H]");

    const int64_t N = l.size(0);
    const int64_t H = l.size(1);
    const int64_t D = l.size(2);

    TORCH_CHECK(attn_vec.size(0) == H && attn_vec.size(1) == D, "attn_vec must be [H, D]");
    TORCH_CHECK(logsumexp.size(0) == N && logsumexp.size(1) == H, "logsumexp must be [N, H]");
    TORCH_CHECK(D % 4 == 0, "head_dim (D) must be divisible by 4");

    auto gh_strides = grad_h.strides();
    auto l_strides  = l.strides();
    auto r_strides  = r.strides();

    int64_t stride_gh_n = gh_strides[0];
    int64_t stride_gh_h = gh_strides[1];
    int64_t stride_gh_d = gh_strides[2];

    int64_t stride_l_n = l_strides[0];
    int64_t stride_l_h = l_strides[1];
    int64_t stride_l_d = l_strides[2];

    int64_t stride_r_n = r_strides[0];
    int64_t stride_r_h = r_strides[1];
    int64_t stride_r_d = r_strides[2];

    TORCH_CHECK(stride_gh_d == 1 && stride_l_d == 1 && stride_r_d == 1,
                "For now, feature dim (D) must be contiguous (stride_d == 1) for grad_h, l, r");

    auto input_dtype = l.dtype();
    auto f32_options = torch::TensorOptions().dtype(torch::kFloat32).device(l.device());
    auto typed_options = torch::TensorOptions().dtype(input_dtype).device(l.device());

    // grad_l, grad_r: match input dtype
    torch::Tensor grad_l = torch::empty({N, H, D}, typed_options);
    torch::Tensor grad_r = torch::empty({N, H, D}, typed_options);
    // grad_a: always float32 (internal)
    torch::Tensor grad_a = torch::empty({N, H, D}, f32_options);
    // grad_a_reduced: accumulate in float32 to avoid bf16/fp16 atomicAdd contention
    torch::Tensor grad_a_reduced_f32 = torch::zeros({H, D}, f32_options);

    const float* d_logsumexp = logsumexp.data_ptr<float>();
    float* d_grad_a          = grad_a.data_ptr<float>();
    cudaStream_t stream = 0;

    TORCH_CHECK(D == 32 || D == 64 || D == 128 || D == 256,
                "GATv2 backward: unsupported head dim D=", D, "; supported: 32, 64, 128, 256");

    if (is_directed) {
        // Directed path: warp-parallel bucketed AL + R kernels

        // Allocate G [N, H] for all nodes (both light + heavy AL write into it)
        torch::Tensor G_tensor = torch::empty({N, H}, f32_options);
        float* d_G = G_tensor.data_ptr<float>();

        // Lambda to launch AL kernel for a bucket
        auto launch_al_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
            if (num_nodes_bucket == 0) return;
            std::visit([&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                using index_t = typename decltype(idxInfo)::Type;
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;
                constexpr int W = decltype(warp_c)::value;

                auto* grad_h_ptr = reinterpret_cast<const cuda_t*>(grad_h.data_ptr<torch_t>());
                auto* l_ptr      = reinterpret_cast<const cuda_t*>(l.data_ptr<torch_t>());
                auto* r_ptr      = reinterpret_cast<const cuda_t*>(r.data_ptr<torch_t>());
                auto* attn_ptr   = reinterpret_cast<const cuda_t*>(attn_vec.data_ptr<torch_t>());
                auto* grad_l_ptr = reinterpret_cast<cuda_t*>(grad_l.data_ptr<torch_t>());

                size_t sh_al = 2 * DC * sizeof(cuda_t)
                             + W * 2 * DC * sizeof(float)
                             + (W + 1) * sizeof(float);

                dim3 blocks(num_nodes_bucket, H);
                dim3 threads(W * kMaxThreadsInWarp);

                GATv2Backward_AL<W, DC, cuda_t, index_t><<<blocks, threads, sh_al, stream>>>(
                    N, H, D, grad_h_ptr, stride_gh_n, stride_gh_h,
                    l_ptr, stride_l_n, stride_l_h,
                    r_ptr, stride_r_n, stride_r_h,
                    index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
                    index_ptr<index_t>(node_indices),
                    attn_ptr, d_logsumexp, negative_slope,
                    d_grad_a, grad_l_ptr, d_G);
            }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
               MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
               MakeIntVariant<32, 64, 128, 256>((int)D),
               warp_variant);
        };

        // Lambda to launch R kernel for a bucket
        auto launch_r_bucket = [&](torch::Tensor& node_indices, int num_nodes_bucket, auto warp_variant) {
            if (num_nodes_bucket == 0) return;
            std::visit([&](auto idxInfo, auto typeInfo, auto d_c, auto warp_c) {
                using index_t = typename decltype(idxInfo)::Type;
                using torch_t = typename decltype(typeInfo)::TorchType;
                using cuda_t = typename decltype(typeInfo)::CudaType;
                constexpr int DC = decltype(d_c)::value;
                constexpr int W = decltype(warp_c)::value;

                auto* grad_h_ptr = reinterpret_cast<const cuda_t*>(grad_h.data_ptr<torch_t>());
                auto* l_ptr      = reinterpret_cast<const cuda_t*>(l.data_ptr<torch_t>());
                auto* r_ptr      = reinterpret_cast<const cuda_t*>(r.data_ptr<torch_t>());
                auto* attn_ptr   = reinterpret_cast<const cuda_t*>(attn_vec.data_ptr<torch_t>());
                auto* grad_r_ptr = reinterpret_cast<cuda_t*>(grad_r.data_ptr<torch_t>());

                size_t sh_r = DC * sizeof(cuda_t) + W * DC * sizeof(float);

                dim3 blocks(num_nodes_bucket, H);
                dim3 threads(W * kMaxThreadsInWarp);

                GATv2Backward_R<W, DC, cuda_t, index_t><<<blocks, threads, sh_r, stream>>>(
                    N, H, D, grad_h_ptr, stride_gh_n, stride_gh_h,
                    l_ptr, stride_l_n, stride_l_h,
                    r_ptr, stride_r_n, stride_r_h,
                    index_ptr<index_t>(row_ptr_T), index_ptr<index_t>(col_idx_T),
                    index_ptr<index_t>(node_indices),
                    attn_ptr, d_logsumexp, d_G, negative_slope, grad_r_ptr);
            }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
               MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
               MakeIntVariant<32, 64, 128, 256>((int)D),
               warp_variant);
        };

        // 1: AL kernel (forward CSR direction) - light + heavy
        launch_al_bucket(fwd_light_nodes, fwd_light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block));
        launch_al_bucket(fwd_heavy_nodes, fwd_heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block));

        // 2: R kernel (backward CSR direction) - light + heavy
        launch_r_bucket(bwd_light_nodes, bwd_light_nodes.numel(), MakeIntVariant<1, 2, 4>(light_warps_per_block));
        launch_r_bucket(bwd_heavy_nodes, bwd_heavy_nodes.numel(), MakeIntVariant<8, 16, 32>(heavy_warps_per_block));

        // 3: ReduceGradA
        {
            size_t shmem_gradA_reduce_size = (kMaxThreadsInWarp * (kMaxThreadsInWarp + 2)) * sizeof(float);
            dim3 grad_A_reduce_blockDim(kMaxThreadsInWarp, kMaxThreadsInWarp);

            std::visit([&](auto typeInfo, auto chunk_c) {
                using cuda_t = typename decltype(typeInfo)::CudaType;
                constexpr int CHUNK = decltype(chunk_c)::value;
                dim3 grad_A_reduce_gridDim(
                    (N + CHUNK - 1) / CHUNK,
                    (D + kMaxThreadsInWarp - 1) / kMaxThreadsInWarp,
                    H
                );
                ReduceGradAKernel<CHUNK, cuda_t><<<grad_A_reduce_gridDim, grad_A_reduce_blockDim, shmem_gradA_reduce_size>>>(
                    N, H, D, d_grad_a, grad_a_reduced_f32.data_ptr<float>()
                );
            }, MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
               MakeIntVariant<32, 64, 128, 256, 512, 1024, 2048>(grad_A_reduce_row_chunk_size));
        }
    } else {
        // Undirected path: fused G + ALR kernel (no bucketing, no CSR^T)
        std::visit([&](auto idxInfo, auto typeInfo, auto d_c) {
            using index_t = typename decltype(idxInfo)::Type;
            using torch_t = typename decltype(typeInfo)::TorchType;
            using cuda_t = typename decltype(typeInfo)::CudaType;
            constexpr int DC = decltype(d_c)::value;

            auto* grad_h_ptr     = reinterpret_cast<const cuda_t*>(grad_h.data_ptr<torch_t>());
            auto* l_ptr          = reinterpret_cast<const cuda_t*>(l.data_ptr<torch_t>());
            auto* r_ptr          = reinterpret_cast<const cuda_t*>(r.data_ptr<torch_t>());
            auto* attn_ptr       = reinterpret_cast<const cuda_t*>(attn_vec.data_ptr<torch_t>());
            auto* grad_l_ptr     = reinterpret_cast<cuda_t*>(grad_l.data_ptr<torch_t>());
            auto* grad_r_ptr     = reinterpret_cast<cuda_t*>(grad_r.data_ptr<torch_t>());
            float* grad_a_reduced_ptr = grad_a_reduced_f32.data_ptr<float>();

            GATv2Backward_CSR_Undirected_Impl<DC, cuda_t, index_t>(
                N, H, D,
                grad_h_ptr, stride_gh_n, stride_gh_h,
                l_ptr, stride_l_n, stride_l_h,
                r_ptr, stride_r_n, stride_r_h,
                index_ptr<index_t>(row_ptr), index_ptr<index_t>(col_idx),
                attn_ptr, d_logsumexp,
                negative_slope,
                grad_A_reduce_row_chunk_size,
                stream,
                grad_l_ptr, grad_r_ptr, d_grad_a, grad_a_reduced_ptr
            );
        }, MakeIndexVariant<int32_t, int64_t, uint32_t, uint64_t>(idx_dtype),
           MakeTypeVariant<float, at::Half, at::BFloat16>(l.scalar_type()),
           MakeIntVariant<32, 64, 128, 256>((int)D));
    }

    CUDA_KERNEL_CHECK();

    torch::Tensor grad_a_reduced = grad_a_reduced_f32.to(input_dtype);

    return {grad_l, grad_r, grad_a_reduced};
}
