#ifndef ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_POWERSGD_BODY_H
#define ZQ_CPP_LIB_GRADIENT_COMPRESSION_BODY_POWERSGD_BODY_H

// #include <ATen/cuda/CUDABlas.h>
// #include <ATen/cuda/CUDAContext.h>
#include "../operate_memory/get_policy_general.h"

namespace zq_cpp_lib {
namespace gradient_compression_body {

template <typename policy_t>
int powersgd_encode1_body(
    const torch::Tensor& grad,
    const torch::Tensor& residual,
    const torch::Tensor& Q,
    torch::Tensor& M,
    torch::Tensor& P,
    policy_t policy,
    void* stream
) {
    // input: Grad, Residual, Q
    // output: M, P
    int64_t N = grad.numel();

    (grad.view(-1) + residual).copy_(M.view(-1).slice(0, 0, N));
    // zt.record("M[:N] = Grad + Residual");

    // torch::mm_out(P, M, Q);
    torch::mm(M, Q).copy_(P);
    // zt.record("P = M * Q");
    return 0;
}

template <typename policy_t>
int powersgd_encode2_body(
    torch::Tensor& P,
    const torch::Tensor& M,
    torch::Tensor& Q,
    policy_t policy,
    void* stream
) {
    // input: M
    // output: P, Q

    torch::Tensor p1, p2;
    std::tie(p1, p2) = torch::geqrf(P);
    torch::orgqr(p1, p2).copy_(P);
    // zt.record("P = Orthogonal(P)");

    // std::tie(P, std::ignore) = torch::qr(P);
    // std::tie(P, std::ignore) = torch::linalg::qr(P);

    // torch::mm_out(Q, M.t(), P);
    torch::mm(M.t(), P).copy_(Q);
    // zt.record("Q = M^T * P");
    return 0;
}

template <typename policy_t>
int powersgd_decode_body(
    const torch::Tensor& P,
    const torch::Tensor& Q,
    torch::Tensor& M,
    torch::Tensor& residual,
    torch::Tensor& grad,
    policy_t policy,
    void* stream
) {
    // input: P, Q
    // output: M, Residual, Grad
    int64_t N = grad.numel();

    // torch::mm_out(M, P, Q.t());
    torch::mm(P, Q.t()).copy_(M);
    // zt.record("M = P * Q^T");

    (grad.view(-1) - M.view(-1).slice(0, 0, N)).copy_(residual);
    // zt.record("Residual = Grad - M[:N]");

    M.view(-1).slice(0, 0, N).copy_(grad.view(-1));
    // zt.record("Grad = M[:N]");
    return 0;
}

}
}

#endif