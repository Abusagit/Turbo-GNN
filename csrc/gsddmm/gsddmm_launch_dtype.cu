// Routes every per-op GSDDMM entry point to the shard compiled for the operands' dtype.
#include "gsddmm/gsddmm_launch.cuh"

namespace gsddmm {

void gsddmm_forward_launch_add_f32(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_add_f16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_add_bf16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_add(const GsddmmLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_launch_add_f32(args);
        case at::kHalf:
            return gsddmm_forward_launch_add_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_launch_add_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_launch_add: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_edge_launch_add_f32(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_add_f16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_add_bf16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_add(const GsddmmLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_edge_launch_add_f32(args);
        case at::kHalf:
            return gsddmm_forward_edge_launch_add_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_edge_launch_add_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_edge_launch_add: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_launch_add_f32(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_add_f16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_add_bf16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_add(const GsddmmBackwardLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_launch_add_f32(args);
        case at::kHalf:
            return gsddmm_backward_launch_add_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_launch_add_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_launch_add: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_edge_launch_add_f32(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_add_f16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_add_bf16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_add(const GsddmmBackwardLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_edge_launch_add_f32(args);
        case at::kHalf:
            return gsddmm_backward_edge_launch_add_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_edge_launch_add_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_edge_launch_add: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_launch_sub_f32(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_sub_f16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_sub_bf16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_sub(const GsddmmLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_launch_sub_f32(args);
        case at::kHalf:
            return gsddmm_forward_launch_sub_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_launch_sub_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_launch_sub: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_edge_launch_sub_f32(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_sub_f16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_sub_bf16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_sub(const GsddmmLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_edge_launch_sub_f32(args);
        case at::kHalf:
            return gsddmm_forward_edge_launch_sub_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_edge_launch_sub_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_edge_launch_sub: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_launch_sub_f32(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_sub_f16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_sub_bf16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_sub(const GsddmmBackwardLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_launch_sub_f32(args);
        case at::kHalf:
            return gsddmm_backward_launch_sub_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_launch_sub_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_launch_sub: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_edge_launch_sub_f32(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_sub_f16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_sub_bf16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_sub(const GsddmmBackwardLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_edge_launch_sub_f32(args);
        case at::kHalf:
            return gsddmm_backward_edge_launch_sub_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_edge_launch_sub_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_edge_launch_sub: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_launch_mul_f32(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_mul_f16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_mul_bf16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_mul(const GsddmmLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_launch_mul_f32(args);
        case at::kHalf:
            return gsddmm_forward_launch_mul_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_launch_mul_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_launch_mul: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_edge_launch_mul_f32(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_mul_f16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_mul_bf16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_mul(const GsddmmLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_edge_launch_mul_f32(args);
        case at::kHalf:
            return gsddmm_forward_edge_launch_mul_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_edge_launch_mul_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_edge_launch_mul: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_launch_mul_f32(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_mul_f16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_mul_bf16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_mul(const GsddmmBackwardLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_launch_mul_f32(args);
        case at::kHalf:
            return gsddmm_backward_launch_mul_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_launch_mul_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_launch_mul: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_edge_launch_mul_f32(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_mul_f16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_mul_bf16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_mul(const GsddmmBackwardLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_edge_launch_mul_f32(args);
        case at::kHalf:
            return gsddmm_backward_edge_launch_mul_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_edge_launch_mul_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_edge_launch_mul: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_launch_div_f32(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_div_f16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_div_bf16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_div(const GsddmmLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_launch_div_f32(args);
        case at::kHalf:
            return gsddmm_forward_launch_div_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_launch_div_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_launch_div: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_edge_launch_div_f32(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_div_f16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_div_bf16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_div(const GsddmmLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_edge_launch_div_f32(args);
        case at::kHalf:
            return gsddmm_forward_edge_launch_div_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_edge_launch_div_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_edge_launch_div: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_launch_div_f32(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_div_f16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_div_bf16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_div(const GsddmmBackwardLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_launch_div_f32(args);
        case at::kHalf:
            return gsddmm_backward_launch_div_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_launch_div_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_launch_div: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_edge_launch_div_f32(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_div_f16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_div_bf16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_div(const GsddmmBackwardLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_edge_launch_div_f32(args);
        case at::kHalf:
            return gsddmm_backward_edge_launch_div_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_edge_launch_div_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_edge_launch_div: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_launch_dot_f32(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_dot_f16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_dot_bf16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_dot(const GsddmmLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_launch_dot_f32(args);
        case at::kHalf:
            return gsddmm_forward_launch_dot_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_launch_dot_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_launch_dot: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_edge_launch_dot_f32(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_dot_f16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_dot_bf16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_dot(const GsddmmLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_edge_launch_dot_f32(args);
        case at::kHalf:
            return gsddmm_forward_edge_launch_dot_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_edge_launch_dot_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_edge_launch_dot: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_launch_dot_f32(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_dot_f16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_dot_bf16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_dot(const GsddmmBackwardLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_launch_dot_f32(args);
        case at::kHalf:
            return gsddmm_backward_launch_dot_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_launch_dot_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_launch_dot: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_edge_launch_dot_f32(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_dot_f16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_dot_bf16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_dot(const GsddmmBackwardLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_edge_launch_dot_f32(args);
        case at::kHalf:
            return gsddmm_backward_edge_launch_dot_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_edge_launch_dot_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_edge_launch_dot: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_launch_copy_f32(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_copy_f16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_copy_bf16(const GsddmmLaunchArgs& args);
void gsddmm_forward_launch_copy(const GsddmmLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_launch_copy_f32(args);
        case at::kHalf:
            return gsddmm_forward_launch_copy_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_launch_copy_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_launch_copy: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_forward_edge_launch_copy_f32(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_copy_f16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_copy_bf16(const GsddmmLaunchArgsEdge& args);
void gsddmm_forward_edge_launch_copy(const GsddmmLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_forward_edge_launch_copy_f32(args);
        case at::kHalf:
            return gsddmm_forward_edge_launch_copy_f16(args);
        case at::kBFloat16:
            return gsddmm_forward_edge_launch_copy_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_forward_edge_launch_copy: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_launch_copy_f32(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_copy_f16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_copy_bf16(const GsddmmBackwardLaunchArgs& args);
void gsddmm_backward_launch_copy(const GsddmmBackwardLaunchArgs& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_launch_copy_f32(args);
        case at::kHalf:
            return gsddmm_backward_launch_copy_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_launch_copy_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_launch_copy: unsupported dtype ", args.L.scalar_type());
    }
}

void gsddmm_backward_edge_launch_copy_f32(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_copy_f16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_copy_bf16(const GsddmmBackwardLaunchArgsEdge& args);
void gsddmm_backward_edge_launch_copy(const GsddmmBackwardLaunchArgsEdge& args) {
    switch (args.L.scalar_type()) {
        case at::kFloat:
            return gsddmm_backward_edge_launch_copy_f32(args);
        case at::kHalf:
            return gsddmm_backward_edge_launch_copy_f16(args);
        case at::kBFloat16:
            return gsddmm_backward_edge_launch_copy_bf16(args);
        default:
            TORCH_CHECK(false, "gsddmm_backward_edge_launch_copy: unsupported dtype ", args.L.scalar_type());
    }
}

}  // namespace gsddmm
