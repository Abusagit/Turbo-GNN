#pragma once

#include <cuda/pipeline>

#ifndef TGNN_PIPE_MODE
#define TGNN_PIPE_MODE 0
#endif

#include "common/misc.cuh"
#include "common/traits.cuh"

template <size_t STAGES, size_t ROW_ELEMS, FloatingNum num_type>
class RowPipeline {
   public:
    static constexpr size_t kCopyBytes = 16;
    static constexpr size_t kRowBytes  = ROW_ELEMS * sizeof(num_type);

    static constexpr size_t kChunksPerRow = kRowBytes / kCopyBytes;

    static constexpr size_t smem_bytes(size_t workers) {
        return STAGES == 1 ? 0 : workers * STAGES * kRowBytes;
    }
    static constexpr size_t smem_elems(size_t workers) {
        return smem_bytes(workers) / sizeof(num_type);
    }

    __device__ __forceinline__ RowPipeline(num_type *const stage_base, size_t lane, size_t lane_cnt)
        : stage_base_(stage_base),
          lane_(lane),
          lane_cnt_(lane_cnt),
          prod_idx_(0),
          cons_idx_(0),
          direct_(nullptr),
          pipe_(cuda::make_pipeline()) {}

    __device__ __forceinline__ void prefetch(num_type const *const src) {
        if constexpr (STAGES == 1) {
            direct_ = src;
        } else {
            src_ring_[prod_idx_] = src;
#if TGNN_PIPE_MODE == 2
            if (src != nullptr) {
                uint4 *const dst        = reinterpret_cast<uint4 *>(stage_base_ + prod_idx_ * ROW_ELEMS);
                uint4 const *const from = reinterpret_cast<uint4 const *>(src);
                for (size_t chunk = lane_; chunk < kChunksPerRow; chunk += lane_cnt_) {
                    dst[chunk] = from[chunk];
                }
            }
#else
            pipe_.producer_acquire();
            if (src != nullptr) {
                char *const dst        = reinterpret_cast<char *>(stage_base_ + prod_idx_ * ROW_ELEMS);
                char const *const from = reinterpret_cast<char const *>(src);
                for (size_t chunk = lane_; chunk < kChunksPerRow; chunk += lane_cnt_) {
                    cuda::memcpy_async(
                        dst + chunk * kCopyBytes, from + chunk * kCopyBytes, cuda::aligned_size_t<kCopyBytes>(kCopyBytes), pipe_
                    );
                }
            }
            pipe_.producer_commit();
#endif
            prod_idx_ = advance(prod_idx_);
        }
    }

    __device__ __forceinline__ num_type const *consume() {
        if constexpr (STAGES == 1) {
            return direct_;
        } else {
#if TGNN_PIPE_MODE != 2
            cuda::pipeline_consumer_wait_prior<STAGES - 1>(pipe_);
#endif
            __syncwarp();
#if TGNN_PIPE_MODE == 1
            return src_ring_[cons_idx_];
#else
            return stage_base_ + cons_idx_ * ROW_ELEMS;
#endif
        }
    }

    __device__ __forceinline__ void release() {
        if constexpr (STAGES > 1) {
#if TGNN_PIPE_MODE != 2
            pipe_.consumer_release();
#endif
            cons_idx_ = advance(cons_idx_);
        }
    }

   private:
    static __device__ __forceinline__ size_t advance(size_t idx) { return idx + 1 == STAGES ? 0 : idx + 1; }

    num_type *const stage_base_;
    const size_t lane_;
    const size_t lane_cnt_;
    size_t prod_idx_;
    size_t cons_idx_;
    num_type const *direct_;
    num_type const *src_ring_[STAGES];
    cuda::pipeline<cuda::thread_scope_thread> pipe_;
};
