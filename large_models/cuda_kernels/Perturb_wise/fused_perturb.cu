/*
 * Fused RNG + Perturbation CUDA Kernel for MeZO
 * 
 * This kernel eliminates the need for storing z by generating random numbers
 * and applying perturbation in a single fused pass.
 * 
 * Algorithm:
 *   param[i] = param[i] + alpha * randn(seed, i)
 * 
 * Key optimizations:
 * 1. Uses Philox RNG (same as PyTorch) for reproducibility
 * 2. Box-Muller transform for Gaussian distribution
 * 3. Coalesced memory access patterns
 * 4. No intermediate storage needed (ZERO extra memory!)
 * 
 * Compile with:
 *   nvcc -O3 -arch=sm_80 -Xcompiler -fPIC -shared fused_perturb.cu -o libfused_perturb.so
 * 
 * Or for PyTorch extension:
 *   python setup.py install
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cmath>

// For PyTorch C++ extension
#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// =============================================================================
// Philox RNG Implementation (matches PyTorch/cuRAND)
// =============================================================================

// Philox constants - use static const instead of __constant__
#define PHILOX_M4x32_0 0xD2511F53u
#define PHILOX_M4x32_1 0xCD9E8D57u
#define PHILOX_W32_0 0x9E3779B9u
#define PHILOX_W32_1 0xBB67AE85u

struct Philox4x32 {
    unsigned int counter[4];
    unsigned int key[2];
    unsigned int output[4];
    int idx;
    
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
    
    // Convert to uniform [0, 1)
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
};


// =============================================================================
// Kernel 1: Fused RNG + Perturbation (Basic)
// =============================================================================

__global__ void fused_perturb_kernel_v1(
    float* __restrict__ params,
    const long long seed,
    const float alpha,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // Initialize RNG for this element
        Philox4x32 rng(seed, idx, 0);
        
        // Generate random normal value
        float z = rng.next_normal();
        
        // Apply perturbation in-place
        params[idx] += alpha * z;
    }
}


// =============================================================================
// Kernel 2: Fused RNG + Perturbation (Vectorized - 4 elements per thread)
// =============================================================================

__global__ void fused_perturb_kernel_v2(
    float4* __restrict__ params,
    const long long seed,
    const float alpha,
    const long long n_elements_div4,
    const long long n_elements_total
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements_div4) {
        const long long base_idx = idx * 4;
        
        // Load 4 parameters at once (coalesced)
        float4 p = params[idx];
        
        // Generate 4 random normals
        Philox4x32 rng(seed, base_idx, 0);
        float z0 = rng.next_normal();
        float z1 = rng.next_normal();
        float z2 = rng.next_normal();
        float z3 = rng.next_normal();
        
        // Apply perturbation
        p.x += alpha * z0;
        p.y += alpha * z1;
        p.z += alpha * z2;
        p.w += alpha * z3;
        
        // Store back (coalesced)
        params[idx] = p;
    }
}

// Handle remainder elements
__global__ void fused_perturb_kernel_remainder(
    float* __restrict__ params,
    const long long seed,
    const float alpha,
    const long long start_idx,
    const long long n_elements
) {
    const long long idx = start_idx + blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        Philox4x32 rng(seed, idx, 0);
        float z = rng.next_normal();
        params[idx] += alpha * z;
    }
}


// =============================================================================
// Kernel 3: Fused Restore + Update (combines two operations)
// =============================================================================

// Does: params = params + eps*z - lr*projected_grad*z
//     = params + (eps - lr*projected_grad) * z

__global__ void fused_restore_update_kernel(
    float* __restrict__ params,
    const long long seed,
    const float eps,
    const float projected_grad,
    const float lr,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        Philox4x32 rng(seed, idx, 0);
        float z = rng.next_normal();
        
        // Combined: restore from -eps*z position, then update
        float combined_alpha = eps - lr * projected_grad;
        params[idx] += combined_alpha * z;
    }
}


// =============================================================================
// Kernel 3b: Fused Update (MeZO update step from original position)
// =============================================================================

// Does: params = params - lr * projected_grad * z
// This is the actual MeZO update step (from trainer.py: zo_update)

__global__ void fused_update_kernel(
    float* __restrict__ params,
    const long long seed,
    const float projected_grad,
    const float lr,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        Philox4x32 rng(seed, idx, 0);
        float z = rng.next_normal();
        
        // MeZO update: params = params - lr * projected_grad * z
        params[idx] -= lr * projected_grad * z;
    }
}


// =============================================================================
// Kernel 4: Fused Perturb with cuRAND States (Alternative - pre-initialized)
// =============================================================================

// This version uses cuRAND states that can be seeded once and reused
// Faster for multiple calls but requires storing states

__global__ void init_curand_states(
    curandState* states,
    const long long seed,
    const long long n_states
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n_states) {
        curand_init(seed, idx, 0, &states[idx]);
    }
}

__global__ void fused_perturb_curand(
    float* __restrict__ params,
    curandState* states,
    const float alpha,
    const long long n_elements
) {
    const long long idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // Use pre-initialized state
        float z = curand_normal(&states[idx]);
        params[idx] += alpha * z;
    }
}


// =============================================================================
// Kernel 5: Chunked Fused Kernel (for very large models)
// =============================================================================

// This processes parameters in chunks but with fused RNG
// Each chunk uses a different subsequence to maintain reproducibility

__global__ void fused_perturb_chunked_kernel(
    float* __restrict__ params,
    const long long seed,
    const float alpha,
    const long long chunk_offset,  // Starting index for this chunk
    const long long chunk_size,    // Number of elements in this chunk
    const long long total_elements // Total elements (for bounds check)
) {
    const long long local_idx = blockIdx.x * blockDim.x + threadIdx.x;
    const long long global_idx = chunk_offset + local_idx;
    
    if (local_idx < chunk_size && global_idx < total_elements) {
        // Use global index as subsequence for reproducibility
        Philox4x32 rng(seed, global_idx, 0);
        float z = rng.next_normal();
        params[global_idx] += alpha * z;
    }
}


// =============================================================================
// Host-side Wrapper Functions
// =============================================================================

extern "C" {

void launch_fused_perturb_v1(
    float* params,
    long long seed,
    float alpha,
    long long n_elements,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n_elements + block_size - 1) / block_size;
    
    fused_perturb_kernel_v1<<<grid_size, block_size, 0, stream>>>(
        params, seed, alpha, n_elements
    );
}

void launch_fused_perturb_v2(
    float* params,
    long long seed,
    float alpha,
    long long n_elements,
    cudaStream_t stream
) {
    const int block_size = 256;
    const long long n_elements_div4 = n_elements / 4;
    const long long remainder = n_elements % 4;
    
    if (n_elements_div4 > 0) {
        const int grid_size = (n_elements_div4 + block_size - 1) / block_size;
        fused_perturb_kernel_v2<<<grid_size, block_size, 0, stream>>>(
            reinterpret_cast<float4*>(params), seed, alpha, n_elements_div4, n_elements
        );
    }
    
    if (remainder > 0) {
        const int grid_size_rem = 1;
        fused_perturb_kernel_remainder<<<grid_size_rem, block_size, 0, stream>>>(
            params, seed, alpha, n_elements_div4 * 4, n_elements
        );
    }
}

void launch_fused_restore_update(
    float* params,
    long long seed,
    float eps,
    float projected_grad,
    float lr,
    long long n_elements,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n_elements + block_size - 1) / block_size;
    
    fused_restore_update_kernel<<<grid_size, block_size, 0, stream>>>(
        params, seed, eps, projected_grad, lr, n_elements
    );
}

void launch_fused_update(
    float* params,
    long long seed,
    float projected_grad,
    float lr,
    long long n_elements,
    cudaStream_t stream
) {
    const int block_size = 256;
    const int grid_size = (n_elements + block_size - 1) / block_size;
    
    fused_update_kernel<<<grid_size, block_size, 0, stream>>>(
        params, seed, projected_grad, lr, n_elements
    );
}

}  // extern "C"


// =============================================================================
// PyTorch C++ Extension Bindings
// =============================================================================

#ifdef USE_TORCH

torch::Tensor fused_perturb_cuda(
    torch::Tensor params,
    int64_t seed,
    double alpha
) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    
    const int64_t n_elements = params.numel();
    float* params_ptr = params.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_fused_perturb_v2(params_ptr, seed, static_cast<float>(alpha), n_elements, stream);
    
    return params;
}

torch::Tensor fused_restore_update_cuda(
    torch::Tensor params,
    int64_t seed,
    double eps,
    double projected_grad,
    double lr
) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    
    const int64_t n_elements = params.numel();
    float* params_ptr = params.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_fused_restore_update(
        params_ptr, seed, 
        static_cast<float>(eps), 
        static_cast<float>(projected_grad),
        static_cast<float>(lr),
        n_elements, stream
    );
    
    return params;
}

torch::Tensor fused_update_cuda(
    torch::Tensor params,
    int64_t seed,
    double projected_grad,
    double lr
) {
    TORCH_CHECK(params.is_cuda(), "params must be a CUDA tensor");
    TORCH_CHECK(params.is_contiguous(), "params must be contiguous");
    TORCH_CHECK(params.scalar_type() == torch::kFloat32, "params must be float32");
    
    const int64_t n_elements = params.numel();
    float* params_ptr = params.data_ptr<float>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    
    launch_fused_update(
        params_ptr, seed,
        static_cast<float>(projected_grad),
        static_cast<float>(lr),
        n_elements, stream
    );
    
    return params;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_perturb", &fused_perturb_cuda, 
          "Fused RNG + Perturbation CUDA kernel",
          py::arg("params"), py::arg("seed"), py::arg("alpha"));
    m.def("fused_restore_update", &fused_restore_update_cuda,
          "Fused Restore + Update CUDA kernel",
          py::arg("params"), py::arg("seed"), py::arg("eps"), 
          py::arg("projected_grad"), py::arg("lr"));
    m.def("fused_update", &fused_update_cuda,
          "Fused Update CUDA kernel (MeZO update step)",
          py::arg("params"), py::arg("seed"), 
          py::arg("projected_grad"), py::arg("lr"));
}

#endif  // USE_TORCH
