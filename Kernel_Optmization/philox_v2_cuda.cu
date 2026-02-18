/*
 * Optimized Philox RNG CUDA Kernels for ZO Optimizer Perturbation
 * ===============================================================
 *
 * New benchmarking variants — do NOT replace existing kernels.
 *
 * Optimizations over existing fused_perturb.cu / zo2_cuda_kernels.cu:
 *   A. Template<ROUNDS> — compile-time specialization for 7 or 10 rounds
 *   B. sincosf() — hardware-fused sin+cos → 4 normals from 1 Philox call
 *   C. Stateless inline Philox — no struct overhead, minimal registers
 *   D. __launch_bounds__(256) — compiler hint for occupancy control
 *   E. fmaf() — fused multiply-add for perturbation step
 *   F. __umulhi() — native 32-bit integer multiply (no FP64 pipeline)
 *
 * Kernel variants:
 *   scalar<10>: 1 element/thread, 10 rounds (CUDA reference baseline)
 *   scalar<7> : 1 element/thread, 7 rounds (reduced compute)
 *   vec4<10>  : float4 + sincosf + 4x normals, 10 rounds
 *   vec4<7>   : float4 + sincosf + 4x normals, 7 rounds (most optimized)
 *
 * Key insight: Each Philox-4x32 call produces 4 uint32 outputs.
 * Dual Box-Muller with sincosf converts these to 4 independent N(0,1)
 * normals, mapping perfectly to float4 vectorized load/store.
 *
 * Register pressure comparison (expected):
 *   Existing DiZO Triton Philox: 55 regs/thread (FP64 from uint64 cast)
 *   Existing ZO2 CUDA (__umulhi): 18 regs/thread
 *   This scalar kernel:           ~16-20 regs/thread (stateless, no struct)
 *   This vec4 kernel:             ~20-24 regs/thread (4x normals, float4)
 *
 * Compile: python setup_v2_cuda.py install
 *   or JIT: torch.utils.cpp_extension.load(...)
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

static constexpr float TWO_PI_F = 6.283185307179586f;


// ============================================================
// Stateless Philox 4x32-N using __umulhi (pure uint32, no FP64)
// ============================================================
//
// Unlike the struct-based Philox4x32 in existing code, this is a
// stateless function — no counter-increment logic, no struct overhead.
// Each call maps (seed, counter_val) → 4 independent uint32 outputs.

template <int ROUNDS>
__device__ __forceinline__ void philox_4x32(
    unsigned int seed, unsigned int counter_val,
    unsigned int& out0, unsigned int& out1,
    unsigned int& out2, unsigned int& out3)
{
    unsigned int c0 = counter_val;
    unsigned int c1 = 0u;
    unsigned int c2 = 0u;
    unsigned int c3 = 0u;
    unsigned int k0 = seed;
    unsigned int k1 = 0u;

    #pragma unroll
    for (int i = 0; i < ROUNDS; i++) {
        unsigned int hi0 = __umulhi(c0, 0xD2511F53u);
        unsigned int lo0 = c0 * 0xD2511F53u;
        unsigned int hi1 = __umulhi(c2, 0xCD9E8D57u);
        unsigned int lo1 = c2 * 0xCD9E8D57u;
        c0 = hi1 ^ c1 ^ k0;
        c1 = lo1;
        c2 = hi0 ^ c3 ^ k1;
        c3 = lo0;
        k0 += 0x9E3779B9u;
        k1 += 0xBB67AE85u;
    }

    out0 = c0; out1 = c1; out2 = c2; out3 = c3;
}


// Convert uint32 → uniform (0, 1) — avoids exact 0 for log safety
__device__ __forceinline__ float uint32_to_uniform(unsigned int x) {
    return (static_cast<float>(x) + 0.5f) * (1.0f / 4294967296.0f);
}


// ============================================================
// Kernel: Scalar (1 element per thread)
//   - Each thread: 1 Philox call → 2 uint32 used → 1 Box-Muller → 1 normal
//   - Simple, good baseline for register pressure comparison
// ============================================================

template <int ROUNDS>
__global__ void __launch_bounds__(256)
philox_v2_scalar_kernel(
    float* __restrict__ params,
    unsigned int seed,
    float alpha,
    int64_t n_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    unsigned int c0, c1, c2, c3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), c0, c1, c2, c3);

    float u1 = fmaxf(uint32_to_uniform(c0), 1e-7f);
    float u2 = uint32_to_uniform(c1);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI_F * u2);

    params[idx] = fmaf(alpha, z, params[idx]);
}


// ============================================================
// Kernel: Vec4 + sincosf (4 elements per thread)
//   - Each thread: 1 Philox call → 4 uint32 → 2 Box-Muller pairs
//   - sincosf() gives both sin and cos per pair → 4 normals total
//   - float4 vectorized load/store for 128-bit coalesced access
//   - This is the most compute-efficient variant
// ============================================================

template <int ROUNDS>
__global__ void __launch_bounds__(256)
philox_v2_vec4_kernel(
    float* __restrict__ params,
    unsigned int seed,
    float alpha,
    int64_t n_vec4)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) return;

    // One Philox call per thread → 4 uint32 outputs
    unsigned int r0, r1, r2, r3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), r0, r1, r2, r3);

    // Convert to 4 uniforms
    float u1 = fmaxf(uint32_to_uniform(r0), 1e-7f);
    float u2 = uint32_to_uniform(r1);
    float u3 = fmaxf(uint32_to_uniform(r2), 1e-7f);
    float u4 = uint32_to_uniform(r3);

    // Dual Box-Muller with sincosf → 4 independent normals
    float r_bm1 = sqrtf(-2.0f * logf(u1));
    float sin1, cos1;
    sincosf(TWO_PI_F * u2, &sin1, &cos1);

    float r_bm2 = sqrtf(-2.0f * logf(u3));
    float sin2, cos2;
    sincosf(TWO_PI_F * u4, &sin2, &cos2);

    float z0 = r_bm1 * cos1;
    float z1 = r_bm1 * sin1;
    float z2 = r_bm2 * cos2;
    float z3 = r_bm2 * sin2;

    // Vectorized load → perturb → store
    float4* p4 = reinterpret_cast<float4*>(params);
    float4 p = p4[idx];
    p.x = fmaf(alpha, z0, p.x);
    p.y = fmaf(alpha, z1, p.y);
    p.z = fmaf(alpha, z2, p.z);
    p.w = fmaf(alpha, z3, p.w);
    p4[idx] = p;
}


// Remainder kernel for n_elements not divisible by 4
template <int ROUNDS>
__global__ void philox_v2_remainder_kernel(
    float* __restrict__ params,
    unsigned int seed,
    float alpha,
    int64_t start_idx,
    int64_t n_elements)
{
    const int64_t idx = start_idx + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    // Use element index as counter (different range from vec4 thread indices)
    unsigned int c0, c1, c2, c3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), c0, c1, c2, c3);

    float u1 = fmaxf(uint32_to_uniform(c0), 1e-7f);
    float u2 = uint32_to_uniform(c1);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI_F * u2);

    params[idx] = fmaf(alpha, z, params[idx]);
}


// ============================================================
// Fused dual-perturb kernels:
//   params_plus  = anchor + eps * z
//   params_minus = anchor - eps * z
// in one launch.
// ============================================================

template <int ROUNDS>
__global__ void __launch_bounds__(256)
philox_v2_dual_vec4_kernel(
    float* __restrict__ params_plus,
    float* __restrict__ params_minus,
    const float* __restrict__ anchor,
    unsigned int seed,
    float eps,
    int64_t n_vec4)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) return;

    unsigned int r0, r1, r2, r3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), r0, r1, r2, r3);

    float u1 = fmaxf(uint32_to_uniform(r0), 1e-7f);
    float u2 = uint32_to_uniform(r1);
    float u3 = fmaxf(uint32_to_uniform(r2), 1e-7f);
    float u4 = uint32_to_uniform(r3);

    float r_bm1 = sqrtf(-2.0f * logf(u1));
    float sin1, cos1;
    sincosf(TWO_PI_F * u2, &sin1, &cos1);

    float r_bm2 = sqrtf(-2.0f * logf(u3));
    float sin2, cos2;
    sincosf(TWO_PI_F * u4, &sin2, &cos2);

    float z0 = r_bm1 * cos1;
    float z1 = r_bm1 * sin1;
    float z2 = r_bm2 * cos2;
    float z3 = r_bm2 * sin2;

    const float4* a4 = reinterpret_cast<const float4*>(anchor);
    float4* p4 = reinterpret_cast<float4*>(params_plus);
    float4* m4 = reinterpret_cast<float4*>(params_minus);
    float4 a = a4[idx];
    float4 p, m;
    p.x = fmaf(eps, z0, a.x);
    p.y = fmaf(eps, z1, a.y);
    p.z = fmaf(eps, z2, a.z);
    p.w = fmaf(eps, z3, a.w);
    m.x = fmaf(-eps, z0, a.x);
    m.y = fmaf(-eps, z1, a.y);
    m.z = fmaf(-eps, z2, a.z);
    m.w = fmaf(-eps, z3, a.w);
    p4[idx] = p;
    m4[idx] = m;
}

template <int ROUNDS>
__global__ void philox_v2_dual_remainder_kernel(
    float* __restrict__ params_plus,
    float* __restrict__ params_minus,
    const float* __restrict__ anchor,
    unsigned int seed,
    float eps,
    int64_t start_idx,
    int64_t n_elements)
{
    const int64_t idx = start_idx + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    unsigned int c0, c1, c2, c3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), c0, c1, c2, c3);

    float u1 = fmaxf(uint32_to_uniform(c0), 1e-7f);
    float u2 = uint32_to_uniform(c1);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI_F * u2);

    const float a = anchor[idx];
    params_plus[idx] = fmaf(eps, z, a);
    params_minus[idx] = fmaf(-eps, z, a);
}


// ============================================================
// Update kernels: params = params - lr * grad * z
// grad is read from a GPU scalar tensor (no host sync needed)
// ============================================================

template <int ROUNDS>
__global__ void __launch_bounds__(256)
philox_v2_scalar_update_kernel(
    float* __restrict__ params,
    const float* __restrict__ grad_ptr,
    unsigned int seed,
    float lr,
    int64_t n_elements)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const float grad = grad_ptr[0];

    unsigned int c0, c1, c2, c3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), c0, c1, c2, c3);

    float u1 = fmaxf(uint32_to_uniform(c0), 1e-7f);
    float u2 = uint32_to_uniform(c1);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI_F * u2);

    params[idx] = fmaf(-lr * grad, z, params[idx]);
}

template <int ROUNDS>
__global__ void __launch_bounds__(256)
philox_v2_vec4_update_kernel(
    float* __restrict__ params,
    const float* __restrict__ grad_ptr,
    unsigned int seed,
    float lr,
    int64_t n_vec4)
{
    const int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_vec4) return;

    const float grad = grad_ptr[0];
    const float scale = -lr * grad;

    unsigned int r0, r1, r2, r3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), r0, r1, r2, r3);

    float u1 = fmaxf(uint32_to_uniform(r0), 1e-7f);
    float u2 = uint32_to_uniform(r1);
    float u3 = fmaxf(uint32_to_uniform(r2), 1e-7f);
    float u4 = uint32_to_uniform(r3);

    float r_bm1 = sqrtf(-2.0f * logf(u1));
    float sin1, cos1;
    sincosf(TWO_PI_F * u2, &sin1, &cos1);

    float r_bm2 = sqrtf(-2.0f * logf(u3));
    float sin2, cos2;
    sincosf(TWO_PI_F * u4, &sin2, &cos2);

    float z0 = r_bm1 * cos1;
    float z1 = r_bm1 * sin1;
    float z2 = r_bm2 * cos2;
    float z3 = r_bm2 * sin2;

    float4* p4 = reinterpret_cast<float4*>(params);
    float4 p = p4[idx];
    p.x = fmaf(scale, z0, p.x);
    p.y = fmaf(scale, z1, p.y);
    p.z = fmaf(scale, z2, p.z);
    p.w = fmaf(scale, z3, p.w);
    p4[idx] = p;
}

template <int ROUNDS>
__global__ void philox_v2_remainder_update_kernel(
    float* __restrict__ params,
    const float* __restrict__ grad_ptr,
    unsigned int seed,
    float lr,
    int64_t start_idx,
    int64_t n_elements)
{
    const int64_t idx = start_idx + static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= n_elements) return;

    const float grad = grad_ptr[0];

    unsigned int c0, c1, c2, c3;
    philox_4x32<ROUNDS>(seed, static_cast<unsigned int>(idx), c0, c1, c2, c3);

    float u1 = fmaxf(uint32_to_uniform(c0), 1e-7f);
    float u2 = uint32_to_uniform(c1);
    float z = sqrtf(-2.0f * logf(u1)) * cosf(TWO_PI_F * u2);

    params[idx] = fmaf(-lr * grad, z, params[idx]);
}


// ============================================================
// Host-side launch wrappers
// ============================================================

static constexpr int THREADS = 256;

template <int ROUNDS>
void launch_scalar(
    float* params,
    unsigned int seed,
    float alpha,
    int64_t n_elements,
    cudaStream_t stream)
{
    const int blocks = static_cast<int>((n_elements + THREADS - 1) / THREADS);
    philox_v2_scalar_kernel<ROUNDS><<<blocks, THREADS, 0, stream>>>(
        params, seed, alpha, n_elements);
}

template <int ROUNDS>
void launch_vec4(
    float* params,
    unsigned int seed,
    float alpha,
    int64_t n_elements,
    cudaStream_t stream)
{
    const int64_t n_vec4 = n_elements / 4;
    const int64_t remainder = n_elements % 4;

    if (n_vec4 > 0) {
        const int blocks = static_cast<int>((n_vec4 + THREADS - 1) / THREADS);
        philox_v2_vec4_kernel<ROUNDS><<<blocks, THREADS, 0, stream>>>(
            params, seed, alpha, n_vec4);
    }

    if (remainder > 0) {
        philox_v2_remainder_kernel<ROUNDS><<<1, THREADS, 0, stream>>>(
            params, seed, alpha, n_vec4 * 4, n_elements);
    }
}

template <int ROUNDS>
void launch_dual_vec4(
    float* params_plus,
    float* params_minus,
    const float* anchor,
    unsigned int seed,
    float eps,
    int64_t n_elements,
    cudaStream_t stream)
{
    const int64_t n_vec4 = n_elements / 4;
    const int64_t remainder = n_elements % 4;

    if (n_vec4 > 0) {
        const int blocks = static_cast<int>((n_vec4 + THREADS - 1) / THREADS);
        philox_v2_dual_vec4_kernel<ROUNDS><<<blocks, THREADS, 0, stream>>>(
            params_plus, params_minus, anchor, seed, eps, n_vec4);
    }

    if (remainder > 0) {
        philox_v2_dual_remainder_kernel<ROUNDS><<<1, THREADS, 0, stream>>>(
            params_plus, params_minus, anchor, seed, eps, n_vec4 * 4, n_elements);
    }
}

template <int ROUNDS>
void launch_scalar_update(
    float* params,
    const float* grad_ptr,
    unsigned int seed,
    float lr,
    int64_t n_elements,
    cudaStream_t stream)
{
    const int blocks = static_cast<int>((n_elements + THREADS - 1) / THREADS);
    philox_v2_scalar_update_kernel<ROUNDS><<<blocks, THREADS, 0, stream>>>(
        params, grad_ptr, seed, lr, n_elements);
}

template <int ROUNDS>
void launch_vec4_update(
    float* params,
    const float* grad_ptr,
    unsigned int seed,
    float lr,
    int64_t n_elements,
    cudaStream_t stream)
{
    const int64_t n_vec4 = n_elements / 4;
    const int64_t remainder = n_elements % 4;

    if (n_vec4 > 0) {
        const int blocks = static_cast<int>((n_vec4 + THREADS - 1) / THREADS);
        philox_v2_vec4_update_kernel<ROUNDS><<<blocks, THREADS, 0, stream>>>(
            params, grad_ptr, seed, lr, n_vec4);
    }

    if (remainder > 0) {
        philox_v2_remainder_update_kernel<ROUNDS><<<1, THREADS, 0, stream>>>(
            params, grad_ptr, seed, lr, n_vec4 * 4, n_elements);
    }
}


// ============================================================
// PyTorch C++ Extension Bindings
// ============================================================

torch::Tensor perturb_scalar_10r(torch::Tensor params, int64_t seed, double alpha) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    launch_scalar<10>(
        params.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(alpha),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor perturb_scalar_7r(torch::Tensor params, int64_t seed, double alpha) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    launch_scalar<7>(
        params.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(alpha),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor perturb_vec4_10r(torch::Tensor params, int64_t seed, double alpha) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    launch_vec4<10>(
        params.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(alpha),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor perturb_vec4_7r(torch::Tensor params, int64_t seed, double alpha) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    launch_vec4<7>(
        params.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(alpha),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor dual_perturb_vec4_10r(
    torch::Tensor params_plus,
    torch::Tensor params_minus,
    torch::Tensor anchor,
    int64_t seed,
    double eps)
{
    TORCH_CHECK(params_plus.is_cuda(), "params_plus must be a CUDA tensor");
    TORCH_CHECK(params_minus.is_cuda(), "params_minus must be a CUDA tensor");
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(params_plus.is_contiguous(), "params_plus must be contiguous");
    TORCH_CHECK(params_minus.is_contiguous(), "params_minus must be contiguous");
    TORCH_CHECK(anchor.is_contiguous(), "anchor must be contiguous");
    TORCH_CHECK(params_plus.scalar_type() == torch::kFloat32, "params_plus must be float32");
    TORCH_CHECK(params_minus.scalar_type() == torch::kFloat32, "params_minus must be float32");
    TORCH_CHECK(anchor.scalar_type() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(params_plus.numel() == anchor.numel(), "params_plus and anchor must have same numel");
    TORCH_CHECK(params_minus.numel() == anchor.numel(), "params_minus and anchor must have same numel");
    launch_dual_vec4<10>(
        params_plus.data_ptr<float>(),
        params_minus.data_ptr<float>(),
        anchor.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(eps),
        anchor.numel(),
        at::cuda::getCurrentCUDAStream());
    return params_plus;
}

torch::Tensor dual_perturb_vec4_7r(
    torch::Tensor params_plus,
    torch::Tensor params_minus,
    torch::Tensor anchor,
    int64_t seed,
    double eps)
{
    TORCH_CHECK(params_plus.is_cuda(), "params_plus must be a CUDA tensor");
    TORCH_CHECK(params_minus.is_cuda(), "params_minus must be a CUDA tensor");
    TORCH_CHECK(anchor.is_cuda(), "anchor must be a CUDA tensor");
    TORCH_CHECK(params_plus.is_contiguous(), "params_plus must be contiguous");
    TORCH_CHECK(params_minus.is_contiguous(), "params_minus must be contiguous");
    TORCH_CHECK(anchor.is_contiguous(), "anchor must be contiguous");
    TORCH_CHECK(params_plus.scalar_type() == torch::kFloat32, "params_plus must be float32");
    TORCH_CHECK(params_minus.scalar_type() == torch::kFloat32, "params_minus must be float32");
    TORCH_CHECK(anchor.scalar_type() == torch::kFloat32, "anchor must be float32");
    TORCH_CHECK(params_plus.numel() == anchor.numel(), "params_plus and anchor must have same numel");
    TORCH_CHECK(params_minus.numel() == anchor.numel(), "params_minus and anchor must have same numel");
    launch_dual_vec4<7>(
        params_plus.data_ptr<float>(),
        params_minus.data_ptr<float>(),
        anchor.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(eps),
        anchor.numel(),
        at::cuda::getCurrentCUDAStream());
    return params_plus;
}

torch::Tensor update_scalar_10r_grad(torch::Tensor params, torch::Tensor grad, int64_t seed, double lr) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(grad.numel() == 1, "grad must contain exactly one element");
    launch_scalar_update<10>(
        params.data_ptr<float>(),
        grad.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(lr),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor update_scalar_7r_grad(torch::Tensor params, torch::Tensor grad, int64_t seed, double lr) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(grad.numel() == 1, "grad must contain exactly one element");
    launch_scalar_update<7>(
        params.data_ptr<float>(),
        grad.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(lr),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor update_vec4_10r_grad(torch::Tensor params, torch::Tensor grad, int64_t seed, double lr) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(grad.numel() == 1, "grad must contain exactly one element");
    launch_vec4_update<10>(
        params.data_ptr<float>(),
        grad.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(lr),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}

torch::Tensor update_vec4_7r_grad(torch::Tensor params, torch::Tensor grad, int64_t seed, double lr) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(grad.is_cuda(), "grad must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(grad.is_contiguous(), "grad must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    TORCH_CHECK(grad.scalar_type() == torch::kFloat32, "grad must be float32");
    TORCH_CHECK(grad.numel() == 1, "grad must contain exactly one element");
    launch_vec4_update<7>(
        params.data_ptr<float>(),
        grad.data_ptr<float>(),
        static_cast<unsigned int>(seed),
        static_cast<float>(lr),
        params.numel(),
        at::cuda::getCurrentCUDAStream());
    return params;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("perturb_scalar_10r", &perturb_scalar_10r,
          "Scalar Philox-10r perturbation (__umulhi, 1 elem/thread)",
          py::arg("params"), py::arg("seed"), py::arg("alpha"));
    m.def("perturb_scalar_7r", &perturb_scalar_7r,
          "Scalar Philox-7r perturbation (__umulhi, 1 elem/thread)",
          py::arg("params"), py::arg("seed"), py::arg("alpha"));
    m.def("perturb_vec4_10r", &perturb_vec4_10r,
          "Vec4 Philox-10r perturbation (sincosf, 4 normals/Philox call)",
          py::arg("params"), py::arg("seed"), py::arg("alpha"));
    m.def("perturb_vec4_7r", &perturb_vec4_7r,
          "Vec4 Philox-7r perturbation (sincosf, 4 normals/Philox call, fastest)",
          py::arg("params"), py::arg("seed"), py::arg("alpha"));
    m.def("dual_perturb_vec4_10r", &dual_perturb_vec4_10r,
          "Vec4 Philox-10r fused dual perturbation: plus=anchor+eps*z, minus=anchor-eps*z",
          py::arg("params_plus"), py::arg("params_minus"), py::arg("anchor"), py::arg("seed"), py::arg("eps"));
    m.def("dual_perturb_vec4_7r", &dual_perturb_vec4_7r,
          "Vec4 Philox-7r fused dual perturbation: plus=anchor+eps*z, minus=anchor-eps*z",
          py::arg("params_plus"), py::arg("params_minus"), py::arg("anchor"), py::arg("seed"), py::arg("eps"));
    m.def("update_scalar_10r_grad", &update_scalar_10r_grad,
          "Scalar Philox-10r update with grad tensor: params -= lr * grad * z",
          py::arg("params"), py::arg("grad"), py::arg("seed"), py::arg("lr"));
    m.def("update_scalar_7r_grad", &update_scalar_7r_grad,
          "Scalar Philox-7r update with grad tensor: params -= lr * grad * z",
          py::arg("params"), py::arg("grad"), py::arg("seed"), py::arg("lr"));
    m.def("update_vec4_10r_grad", &update_vec4_10r_grad,
          "Vec4 Philox-10r update with grad tensor: params -= lr * grad * z",
          py::arg("params"), py::arg("grad"), py::arg("seed"), py::arg("lr"));
    m.def("update_vec4_7r_grad", &update_vec4_7r_grad,
          "Vec4 Philox-7r update with grad tensor: params -= lr * grad * z",
          py::arg("params"), py::arg("grad"), py::arg("seed"), py::arg("lr"));
}
