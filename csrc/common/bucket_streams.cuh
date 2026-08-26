#pragma once

#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <c10/cuda/CUDAStream.h>

#include "common/misc.cuh"

// Concurrent light/heavy bucket launches.
//
// These convolutions split nodes into a *light* and a *heavy* bucket by degree quantile and
// launch a kernel for each. Historically the two went out back to back on one stream, so the
// heavy launch could not begin until the light one had fully drained -- even though the two
// touch disjoint sets of output rows and have no data dependence whatsoever.
//
// That costs two things. The tail of each launch under-fills the device, and the buckets are
// deliberately unbalanced: the heavy bucket is a handful of very expensive nodes (1% of the
// graph at the default quantile) while the light bucket is everything else. Running heavy
// first and letting light fill in around it is the standard way to shorten a makespan when one
// job is much longer than the rest.
//
// `Concurrent` puts the two on separate streams so the hardware can overlap them. Ordering is
// handled with events rather than synchronisation:
//
//   * at construction, an event recorded on the caller's stream is waited on by the side
//     stream, so neither bucket can start before whatever produced its inputs has finished;
//   * at `join()`, an event recorded on the side stream is waited on by the caller's stream,
//     so the caller's next operation cannot start before both buckets are done.
//
// The heavy bucket keeps the caller's stream and is issued first; the light bucket goes to the
// side stream. Any tensor allocated on the caller's stream but read or written on the side one
// must be passed to `record()`, or the caching allocator may hand its memory to someone else
// while the side stream is still using it.
namespace turbo_gnn::streams {

enum class BucketLaunch : int {
    Sequential = 0,  ///< light then heavy on one stream -- the historical behaviour
    Concurrent = 1,  ///< heavy and light on separate streams, heavy issued first
};

inline constexpr int kNumBucketLaunchModes = 2;

inline BucketLaunch bucket_launch_from_int(int value) {
    TORCH_CHECK(
        value >= 0 && value < kNumBucketLaunchModes, "bucket_launch must be 0 (sequential) or 1 (concurrent), got ", value
    );
    return static_cast<BucketLaunch>(value);
}

/// Hands each bucket the stream it should launch on, and joins them before the caller resumes.
///
/// Joining happens in the destructor as well as in `join()`, so an early return or a thrown
/// check cannot leave the caller's stream running ahead of work still queued on the side one.
class BucketStreams {
  public:
    BucketStreams(BucketLaunch mode, at::Device device)
        : mode_(mode), main_(at::cuda::getCurrentCUDAStream(device.index())), side_(main_) {
        if (mode_ != BucketLaunch::Concurrent) {
            return;
        }
        side_ = at::cuda::getStreamFromPool(/*isHighPriority=*/false, device.index());
        at::cuda::CUDAEvent fork;
        fork.record(main_);
        fork.block(side_);
        forked_ = true;
    }

    BucketStreams(BucketStreams const &)            = delete;
    BucketStreams &operator=(BucketStreams const &) = delete;

    ~BucketStreams() { join(); }

    /// The expensive bucket keeps the caller's stream, and is the one issued first.
    at::cuda::CUDAStream heavy() const { return main_; }
    at::cuda::CUDAStream light() const { return side_; }

    /// True when the caller should issue the heavy bucket before the light one.
    ///
    /// Only `Concurrent` does. Reordering the two on a *single* stream was measured and does
    /// nothing -- the second launch still waits for the first to drain -- so that mode was
    /// removed rather than kept as a tunable: 0.964 geomean over 192 cells, worse than leaving
    /// the order alone. All of the gain comes from the overlap.
    bool heavy_first() const { return mode_ == BucketLaunch::Concurrent; }

    /// True when the two buckets are on different streams, so the light bucket's host-side
    /// prep must run under a stream guard.
    bool overlapped() const { return forked_; }

    /// Mark `t` as in use on the light bucket's stream. No-op unless the streams differ.
    ///
    /// Required for every tensor that was allocated on the caller's stream and is touched by
    /// the light kernel: without it the caching allocator may reuse that block as soon as the
    /// caller's stream is done with it, while the side stream is still reading.
    void record(at::Tensor const &t) const {
        if (!forked_ || !t.defined() || !t.is_cuda()) {
            return;
        }
        c10::cuda::CUDACachingAllocator::recordStream(t.storage().data_ptr(), side_);
    }

    template <typename... Tensors>
    void record_all(Tensors const &...tensors) const {
        (record(tensors), ...);
    }

    /// Make the caller's stream wait for the light bucket. Idempotent.
    void join() {
        if (!forked_) {
            return;
        }
        at::cuda::CUDAEvent done;
        done.record(side_);
        done.block(main_);
        forked_ = false;
    }

  private:
    BucketLaunch mode_;
    at::cuda::CUDAStream main_;
    at::cuda::CUDAStream side_;
    bool forked_ = false;
};

/// Issue a light/heavy pair in the order and on the streams `b` dictates, then join.
///
/// Each callable receives the stream its bucket should launch on. Keeping the call here rather
/// than at every site means the ordering rule and the join live in exactly one place.
template <typename LightFn, typename HeavyFn>
inline void run_buckets(BucketStreams &b, LightFn &&light, HeavyFn &&heavy) {
    if (b.heavy_first()) {
        heavy(b.heavy());
        light(b.light());
    } else {
        light(b.light());
        heavy(b.heavy());
    }
    b.join();
}

}  // namespace turbo_gnn::streams
