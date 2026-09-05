#pragma once

#include <stdexcept>
#include <string>

#include "misc.cuh"

enum class BinaryOp {
    COPY_U,
    COPY_E,
    ADD,
    SUB,
    MUL,
    DIV
};

template <BinaryOp Op>
struct BinaryOps;

template <>
struct BinaryOps<BinaryOp::COPY_U> {
    static constexpr bool USE_LHS = true;
    static constexpr bool USE_RHS = false;
    static constexpr bool GRAD_USES_OPERANDS = false;

    template <typename val_t>
    static __device__ __forceinline__ val_t call(val_t u, val_t /*e*/) {
        return u;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_lhs(val_t /*u*/, val_t /*e*/, val_t g) {
        return g;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_rhs(val_t /*u*/, val_t /*e*/, val_t /*g*/) {
        return val_t{};
    }
};

template <>
struct BinaryOps<BinaryOp::COPY_E> {
    static constexpr bool USE_LHS = false;
    static constexpr bool USE_RHS = true;
    static constexpr bool GRAD_USES_OPERANDS = false;

    template <typename val_t>
    static __device__ __forceinline__ val_t call(val_t /*u*/, val_t e) {
        return e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_lhs(val_t /*u*/, val_t /*e*/, val_t /*g*/) {
        return val_t{};
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_rhs(val_t /*u*/, val_t /*e*/, val_t g) {
        return g;
    }
};

template <>
struct BinaryOps<BinaryOp::ADD> {
    static constexpr bool USE_LHS = true;
    static constexpr bool USE_RHS = true;
    static constexpr bool GRAD_USES_OPERANDS = false;

    template <typename val_t>
    static __device__ __forceinline__ val_t call(val_t u, val_t e) {
        return u + e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_lhs(val_t /*u*/, val_t /*e*/, val_t g) {
        return g;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_rhs(val_t /*u*/, val_t /*e*/, val_t g) {
        return g;
    }
};

template <>
struct BinaryOps<BinaryOp::SUB> {
    static constexpr bool USE_LHS = true;
    static constexpr bool USE_RHS = true;
    static constexpr bool GRAD_USES_OPERANDS = false;

    template <typename val_t>
    static __device__ __forceinline__ val_t call(val_t u, val_t e) {
        return u - e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_lhs(val_t /*u*/, val_t /*e*/, val_t g) {
        return g;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_rhs(val_t /*u*/, val_t /*e*/, val_t g) {
        return -g;
    }
};

template <>
struct BinaryOps<BinaryOp::MUL> {
    static constexpr bool USE_LHS = true;
    static constexpr bool USE_RHS = true;
    static constexpr bool GRAD_USES_OPERANDS = true;

    template <typename val_t>
    static __device__ __forceinline__ val_t call(val_t u, val_t e) {
        return u * e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_lhs(val_t /*u*/, val_t e, val_t g) {
        return g * e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_rhs(val_t u, val_t /*e*/, val_t g) {
        return g * u;
    }
};

template <>
struct BinaryOps<BinaryOp::DIV> {
    static constexpr bool USE_LHS = true;
    static constexpr bool USE_RHS = true;
    static constexpr bool GRAD_USES_OPERANDS = true;

    template <typename val_t>
    static __device__ __forceinline__ val_t call(val_t u, val_t e) {
        return u / e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_lhs(val_t /*u*/, val_t e, val_t g) {
        return g / e;
    }
    template <typename val_t>
    static __device__ __forceinline__ val_t grad_rhs(val_t u, val_t e, val_t g) {
        return -g * u / (e * e);
    }
};

inline BinaryOp binary_op_from_string(const std::string& op) {
    if (op == "copy_u") return BinaryOp::COPY_U;
    if (op == "copy_e") return BinaryOp::COPY_E;
    if (op == "add") return BinaryOp::ADD;
    if (op == "sub") return BinaryOp::SUB;
    if (op == "mul") return BinaryOp::MUL;
    if (op == "div") return BinaryOp::DIV;
    throw std::runtime_error(std::string("Unknown op: ") + op);
}
