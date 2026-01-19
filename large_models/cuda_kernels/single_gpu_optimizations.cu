/*
 * CUDA Kernels for Single-GPU ZO Optimization
 *
 * This file contains optimized CUDA kernels for single-GPU parallelism in
 * Zeroth-Order optimization:
 *
 * 1. fused_dual_perturb: Generate z once, write both +eps*z and -eps*z buffers
 * 2. compute_projected_grad: Compute (loss1-loss2)/(2*eps) on GPU
 * 3. fused_update_runtime_grad: Update params using GPU-side gradient
 *
 * Key Optimizations:
 * - Single RNG generation for dual perturbation (2x RNG efficiency)
 * - Vectorized float4 loads/stores for memory bandwidth
 * - Async gradient computation (no .item() CPU sync)
 * - FMA instructions for fused multiply-add
 *
 * Compile with:
 *   python setup_single_gpu_opt.py install
 *
 * Author: DiZO Team
 * Date: 2025-01
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// =============================================================================
// Philox RNG Implementation (matches PyTorch/cuRAND)
// =============================================================================

#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define PHILOX_W32_0 0x9E3779B9u
#define PHILOX_W32_1 0xBB67AE85u

struct Philox4x32 {
    unsigned int counter[4];
    unsigned int key[2];
    unsigned int output[4];
    int idx;

    // Default constructor for static factory method
    __device__ __forceinline__ Philox4x32() : idx(4) {}

    // Standard constructor (cuRAND style: seed, subsequence, offset)
    __device__ __forceinline__
    Philox4x32(unsigned long long seed, unsigned long long subsequence, unsigned long long offset) {
        key[0] = static_cast<unsigned int>(seed);
        key[1] = static_cast<unsigned int>(seed >> 32);
        counter[0] = static_cast<unsigned int>(offset);
        counter[1] = static_cast<unsigned int>(offset >> 32);
        counter[2] = static_cast<unsigned int>(subsequence);
        counter[3] = static_cast<unsigned int>(subsequence >> 32);
        idx = 4;  // Force generation on first call
    }

    // Triton-compatible constructor: uses element index as counter[0]
    // This matches the Triton kernel initialization:
    //   c0 = offsets (element index)
    //   c1, c2, c3 = 0
    //   k0 = seed, k1 = 0
    __device__ __forceinline__
    static Philox4x32 triton_style(unsigned long long seed, unsigned long long element_idx) {
        Philox4x32 rng;
        rng.key[0] = static_cast<unsigned int>(seed);
        rng.key[1] = 0;
        rng.counter[0] = static_cast<unsigned int>(element_idx);
        rng.counter[1] = 0;
        rng.counter[2] = 0;
        rng.counter[3] = 0;
        rng.idx = 4;
        return rng;
    }

    __device__ __forceinline__
    void round(unsigned int* ctr, unsigned int* key) {
        unsigned int hi0, lo0, hi1, lo1;
        lo0 = ctr[0] * PHILOX_M4x32_0;
        hi0 = __umulhi(ctr[0], PHILOX_M4x32_0);
        lo1 = ctr[2] * PHILOX_M4x32_1;
        hi1 = __umulhi(ctr[2], PHILOX_M4x32_1);

        ctr[0] = hi1 ^ ctr[1] ^ key[0];
        ctr[1] = lo1;
        ctr[2] = hi0 ^ ctr[3] ^ key[1];
        ctr[3] = lo0;
    }

    __device__ __forceinline__
    void generate() {
        unsigned int ctr[4] = {counter[0], counter[1], counter[2], counter[3]};
        unsigned int k[2] = {key[0], key[1]};

        // 10 rounds of Philox
        #pragma unroll
        for (int i = 0; i < 10; i++) {
            round(ctr, k);
            k[0] += PHILOX_W32_0;
            k[1] += PHILOX_W32_1;
        }

        output[0] = ctr[0];
        output[1] = ctr[1];
        output[2] = ctr[2];
        output[3] = ctr[3];

        // Increment counter
        counter[0]++;
        if (counter[0] == 0) {
            counter[1]++;
            if (counter[1] == 0) {
                counter[2]++;
                if (counter[2] == 0) {
                    counter[3]++;
                }
            }
        }
        idx = 0;
    }

    __device__ __forceinline__
    unsigned int next_uint32() {
        if (idx >= 4) generate();
        return output[idx++];
    }

    // Convert to uniform [0, 1) using upper 24 bits
    __device__ __forceinline__
    float next_uniform() {
        return (next_uint32() >> 8) * (1.0f / 16777216.0f);
    }

    // Box-Muller transform for normal distribution
    __device__ __forceinline__
    float next_normal() {
        float u1 = next_uniform();
        float u2 = next_uniform();
        // Avoid log(0)
        u1 = fmaxf(u1, 1e-7f);
        return sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }

    // Generate two normals efficiently (full use of Box-Muller)
    __device__ __forceinline__
    void next_normal_pair(float& z0, float& z1) {
        float u1 = next_uniform();
        float u2 = next_uniform();
        u1 = fmaxf(u1, 1e-7f);
        float r = sqrtf(-2.0f * logf(u1));
        float theta = 2.0f * M_PI * u2;
        z0 = r * cosf(theta);
        z1 = r * sinf(theta);
    }
};


// =============================================================================
// Kernel 1: Fused Dual Perturbation (Basic - 1 element per thread)
// =============================================================================
// Key insight: Generate z once, write to TWO output buffers
// Saves 50% RNG computation vs two separate perturb calls

__global__ void fused_dual_perturb_kernel_v1(
    float* __restrict__ params_plus,    // Output: θ + εz
    float* __restrict__ params_minus,   // Output: θ - εz
    const float* __restrict__ anchor,   // Input: θ₀
    const long long seed,
    const float eps,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_elements) {
        // Initialize RNG for this element (Triton-compatible)
        Philox4x32 rng = Philox4x32::triton_style(seed, idx);

        // Generate single random normal
        float z = rng.next_normal();

        // Load anchor (single read)
        float theta = anchor[idx];

        // Compute eps * z once
        float eps_z = eps * z;

        // Write both perturbations
        params_plus[idx] = theta + eps_z;
        params_minus[idx] = theta - eps_z;
    }
}


// =============================================================================
// Kernel 2: Fused Dual Perturbation (Vectorized float4)
// =============================================================================
// Processes 4 elements per thread for better memory bandwidth utilization

__global__ void fused_dual_perturb_kernel_v2(
    float4* __restrict__ params_plus,
    float4* __restrict__ params_minus,
    const float4* __restrict__ anchor,
    const long long seed,
    const float eps,
    const long long n_float4
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n_float4) {
        const long long base_idx = idx * 4;

        // Load anchor as float4 (coalesced 128-bit load)
        float4 theta = __ldg(&anchor[idx]);

        // Initialize RNG (Triton-compatible) - one per element
        Philox4x32 rng0 = Philox4x32::triton_style(seed, base_idx);
        Philox4x32 rng1 = Philox4x32::triton_style(seed, base_idx + 1);
        Philox4x32 rng2 = Philox4x32::triton_style(seed, base_idx + 2);
        Philox4x32 rng3 = Philox4x32::triton_style(seed, base_idx + 3);

        // Generate 4 random normals (each from its own RNG state)
        float z0 = rng0.next_normal();
        float z1 = rng1.next_normal();
        float z2 = rng2.next_normal();
        float z3 = rng3.next_normal();

        // Compute perturbations using FMA
        float4 plus, minus;
        plus.x = __fmaf_rn(eps, z0, theta.x);
        plus.y = __fmaf_rn(eps, z1, theta.y);
        plus.z = __fmaf_rn(eps, z2, theta.z);
        plus.w = __fmaf_rn(eps, z3, theta.w);

        minus.x = __fmaf_rn(-eps, z0, theta.x);
        minus.y = __fmaf_rn(-eps, z1, theta.y);
        minus.z = __fmaf_rn(-eps, z2, theta.z);
        minus.w = __fmaf_rn(-eps, z3, theta.w);

        // Store both (coalesced 128-bit stores)
        params_plus[idx] = plus;
        params_minus[idx] = minus;
    }
}

// Handle remainder elements (when n_elements % 4 != 0)
__global__ void fused_dual_perturb_kernel_remainder(
    float* __restrict__ params_plus,
    float* __restrict__ params_minus,
    const float* __restrict__ anchor,
    const long long seed,
    const float eps,
    const long long start_idx,
    const long long n_elements
) {
    const long long local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long idx = start_idx + local_idx;

    if (idx < n_elements) {
        Philox4x32 rng = Philox4x32::triton_style(seed, idx);
        float z = rng.next_normal();
        float theta = anchor[idx];
        float eps_z = eps * z;

        params_plus[idx] = theta + eps_z;
        params_minus[idx] = theta - eps_z;
    }
}


// =============================================================================
// Kernel 3: Fused Dual Perturbation (Optimized with 2 normals per Box-Muller)
// =============================================================================
// Uses both outputs of Box-Muller transform for maximum efficiency

__global__ void fused_dual_perturb_kernel_v3(
    float* __restrict__ params_plus,
    float* __restrict__ params_minus,
    const float* __restrict__ anchor,
    const long long seed,
    const float eps,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long stride = gridDim.x * blockDim.x;

    // Each thread processes 2 elements using both Box-Muller outputs
    for (long long i = idx * 2; i < n_elements; i += stride * 2) {
        Philox4x32 rng = Philox4x32::triton_style(seed, i);

        // Generate pair of normals
        float z0, z1;
        rng.next_normal_pair(z0, z1);

        // Process first element
        if (i < n_elements) {
            float theta0 = __ldg(&anchor[i]);
            float eps_z0 = eps * z0;
            params_plus[i] = theta0 + eps_z0;
            params_minus[i] = theta0 - eps_z0;
        }

        // Process second element
        if (i + 1 < n_elements) {
            float theta1 = __ldg(&anchor[i + 1]);
            float eps_z1 = eps * z1;
            params_plus[i + 1] = theta1 + eps_z1;
            params_minus[i + 1] = theta1 - eps_z1;
        }
    }
}


// =============================================================================
// Kernel 4: Compute Projected Gradient (Async - no CPU sync)
// =============================================================================
// Computes grad = (loss1 - loss2) / (2 * eps) entirely on GPU

__global__ void compute_projected_grad_kernel(
    const float* __restrict__ loss1_ptr,   // Scalar loss from +eps
    const float* __restrict__ loss2_ptr,   // Scalar loss from -eps
    const float eps,
    float* __restrict__ grad_ptr           // Output: projected gradient
) {
    // Single thread kernel
    if (threadIdx.x == 0 && blockIdx.x == 0) {
        float loss1 = *loss1_ptr;
        float loss2 = *loss2_ptr;
        float grad = (loss1 - loss2) / (2.0f * eps);
        *grad_ptr = grad;
    }
}


// =============================================================================
// Kernel 5: Fused Update with Runtime Gradient (Basic)
// =============================================================================
// Updates params using gradient stored on GPU (avoids .item() sync)
// Formula: params = params - lr * grad * z

__global__ void fused_update_runtime_grad_kernel_v1(
    float* __restrict__ params,
    const float* __restrict__ grad_ptr,    // Gradient on GPU
    const long long seed,
    const float lr,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load gradient (broadcast to all threads via L1 cache)
    float grad = __ldg(grad_ptr);
    float scale = -lr * grad;

    if (idx < n_elements) {
        Philox4x32 rng = Philox4x32::triton_style(seed, idx);
        float z = rng.next_normal();

        params[idx] = __fmaf_rn(scale, z, params[idx]);
    }
}


// =============================================================================
// Kernel 6: Fused Update with Runtime Gradient (Vectorized float4)
// =============================================================================

__global__ void fused_update_runtime_grad_kernel_v2(
    float4* __restrict__ params,
    const float* __restrict__ grad_ptr,
    const long long seed,
    const float lr,
    const long long n_float4
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load gradient once (cached)
    float grad = __ldg(grad_ptr);
    float scale = -lr * grad;

    if (idx < n_float4) {
        const long long base_idx = idx * 4;

        // Load params
        float4 p = params[idx];

        // Generate RNG
        Philox4x32 rng = Philox4x32::triton_style(seed, base_idx);
        float z0 = rng.next_normal();
        float z1 = rng.next_normal();
        float z2 = rng.next_normal();
        float z3 = rng.next_normal();

        // Update with FMA
        p.x = __fmaf_rn(scale, z0, p.x);
        p.y = __fmaf_rn(scale, z1, p.y);
        p.z = __fmaf_rn(scale, z2, p.z);
        p.w = __fmaf_rn(scale, z3, p.w);

        params[idx] = p;
    }
}

// Handle remainder
__global__ void fused_update_runtime_grad_kernel_remainder(
    float* __restrict__ params,
    const float* __restrict__ grad_ptr,
    const long long seed,
    const float lr,
    const long long start_idx,
    const long long n_elements
) {
    const long long local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long idx = start_idx + local_idx;

    float grad = __ldg(grad_ptr);
    float scale = -lr * grad;

    if (idx < n_elements) {
        Philox4x32 rng = Philox4x32::triton_style(seed, idx);
        float z = rng.next_normal();
        params[idx] = __fmaf_rn(scale, z, params[idx]);
    }
}


// =============================================================================
// Kernel 7: Combined Dual Perturb + Grad Compute (Future optimization)
// =============================================================================
// This kernel could fuse perturbation with partial loss computation
// Left as placeholder for future implementation


// =============================================================================
// PyTorch C++ Extension Bindings
// =============================================================================

#ifdef USE_TORCH

void fused_dual_perturb_cuda(
    torch::Tensor params_plus,
    torch::Tensor params_minus,
    torch::Tensor anchor,
    long long seed,
    float eps
) {
    const long long n_elements = anchor.numel();

    // Use vectorized kernel for aligned data
    const long long n_float4 = n_elements / 4;
    const long long remainder_start = n_float4 * 4;
    const long long remainder_count = n_elements - remainder_start;

    const int threads = 256;

    // Launch vectorized kernel
    if (n_float4 > 0) {
        const int blocks = (n_float4 + threads - 1) / threads;
        fused_dual_perturb_kernel_v2<<<blocks, threads>>>(
            reinterpret_cast<float4*>(params_plus.data_ptr<float>()),
            reinterpret_cast<float4*>(params_minus.data_ptr<float>()),
            reinterpret_cast<const float4*>(anchor.data_ptr<float>()),
            seed, eps, n_float4
        );
    }

    // Handle remainder
    if (remainder_count > 0) {
        const int blocks_rem = (remainder_count + threads - 1) / threads;
        fused_dual_perturb_kernel_remainder<<<blocks_rem, threads>>>(
            params_plus.data_ptr<float>(),
            params_minus.data_ptr<float>(),
            anchor.data_ptr<float>(),
            seed, eps, remainder_start, n_elements
        );
    }
}

void compute_projected_grad_cuda(
    torch::Tensor loss1,
    torch::Tensor loss2,
    float eps,
    torch::Tensor grad_out
) {
    compute_projected_grad_kernel<<<1, 1>>>(
        loss1.data_ptr<float>(),
        loss2.data_ptr<float>(),
        eps,
        grad_out.data_ptr<float>()
    );
}

void fused_update_runtime_grad_cuda(
    torch::Tensor params,
    torch::Tensor grad,
    long long seed,
    float lr
) {
    const long long n_elements = params.numel();

    const long long n_float4 = n_elements / 4;
    const long long remainder_start = n_float4 * 4;
    const long long remainder_count = n_elements - remainder_start;

    const int threads = 256;

    // Vectorized kernel
    if (n_float4 > 0) {
        const int blocks = (n_float4 + threads - 1) / threads;
        fused_update_runtime_grad_kernel_v2<<<blocks, threads>>>(
            reinterpret_cast<float4*>(params.data_ptr<float>()),
            grad.data_ptr<float>(),
            seed, lr, n_float4
        );
    }

    // Remainder
    if (remainder_count > 0) {
        const int blocks_rem = (remainder_count + threads - 1) / threads;
        fused_update_runtime_grad_kernel_remainder<<<blocks_rem, threads>>>(
            params.data_ptr<float>(),
            grad.data_ptr<float>(),
            seed, lr, remainder_start, n_elements
        );
    }
}

// Alternative: Basic (non-vectorized) for debugging/comparison
void fused_dual_perturb_cuda_basic(
    torch::Tensor params_plus,
    torch::Tensor params_minus,
    torch::Tensor anchor,
    long long seed,
    float eps
) {
    const long long n_elements = anchor.numel();
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;

    fused_dual_perturb_kernel_v1<<<blocks, threads>>>(
        params_plus.data_ptr<float>(),
        params_minus.data_ptr<float>(),
        anchor.data_ptr<float>(),
        seed, eps, n_elements
    );
}

void fused_update_runtime_grad_cuda_basic(
    torch::Tensor params,
    torch::Tensor grad,
    long long seed,
    float lr
) {
    const long long n_elements = params.numel();
    const int threads = 256;
    const int blocks = (n_elements + threads - 1) / threads;

    fused_update_runtime_grad_kernel_v1<<<blocks, threads>>>(
        params.data_ptr<float>(),
        grad.data_ptr<float>(),
        seed, lr, n_elements
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_dual_perturb", &fused_dual_perturb_cuda,
          "Fused dual perturbation (vectorized)",
          py::arg("params_plus"), py::arg("params_minus"), py::arg("anchor"),
          py::arg("seed"), py::arg("eps"));

    m.def("fused_dual_perturb_basic", &fused_dual_perturb_cuda_basic,
          "Fused dual perturbation (basic)",
          py::arg("params_plus"), py::arg("params_minus"), py::arg("anchor"),
          py::arg("seed"), py::arg("eps"));

    m.def("compute_projected_grad", &compute_projected_grad_cuda,
          "Compute projected gradient on GPU",
          py::arg("loss1"), py::arg("loss2"), py::arg("eps"), py::arg("grad_out"));

    m.def("fused_update_runtime_grad", &fused_update_runtime_grad_cuda,
          "Fused update with runtime gradient (vectorized)",
          py::arg("params"), py::arg("grad"), py::arg("seed"), py::arg("lr"));

    m.def("fused_update_runtime_grad_basic", &fused_update_runtime_grad_cuda_basic,
          "Fused update with runtime gradient (basic)",
          py::arg("params"), py::arg("grad"), py::arg("seed"), py::arg("lr"));
}

#endif  // USE_TORCH
