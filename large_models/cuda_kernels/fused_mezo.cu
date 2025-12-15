/*
 * Fused MeZO CUDA Kernels
 * 
 * These kernels operate directly on model parameters stored in a flat contiguous buffer,
 * enabling single-kernel operations for perturbation and update.
 * 
 * Key optimizations:
 * 1. Single kernel launch for all parameters
 * 2. Inline RNG using cuRAND device API
 * 3. Fused multiply-add operations
 * 4. Coalesced memory access patterns
 */

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <cuda_fp16.h>

// Block size for optimal occupancy
#define BLOCK_SIZE 256
#define WARP_SIZE 32

// =============================================================================
// Kernel 1: Fused Perturbation with Inline RNG
// =============================================================================

__global__ void fused_perturb_with_rng_kernel(
    float* __restrict__ params,      // Model parameters (flat buffer)
    const int64_t n_elements,        // Total number of elements
    const float eps,                 // Perturbation scale
    const uint64_t seed              // Random seed
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // Initialize cuRAND state for this thread
        curandState_t state;
        curand_init(seed, idx, 0, &state);
        
        // Generate random normal value
        float z = curand_normal(&state);
        
        // Fused multiply-add: params[idx] += eps * z
        params[idx] = __fmaf_rn(eps, z, params[idx]);
    }
}


// =============================================================================
// Kernel 2: Perturbation with Pre-generated Z
// =============================================================================

__global__ void perturb_add_kernel(
    float* __restrict__ params,
    const float* __restrict__ z,
    const int64_t n_elements,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        // Fused multiply-add
        params[idx] = __fmaf_rn(alpha, z[idx], params[idx]);
    }
}


// =============================================================================
// Kernel 3: Vectorized Perturbation (4x throughput)
// =============================================================================

__global__ void perturb_add_vec4_kernel(
    float4* __restrict__ params,
    const float4* __restrict__ z,
    const int64_t n_vec4,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_vec4) {
        float4 p = params[idx];
        float4 z_val = z[idx];
        
        p.x = __fmaf_rn(alpha, z_val.x, p.x);
        p.y = __fmaf_rn(alpha, z_val.y, p.y);
        p.z = __fmaf_rn(alpha, z_val.z, p.z);
        p.w = __fmaf_rn(alpha, z_val.w, p.w);
        
        params[idx] = p;
    }
}


// =============================================================================
// Kernel 4: Fused Perturbation + Copy to Model (avoids separate sync)
// =============================================================================

// This kernel copies from flat buffer to multiple destination tensors
// in a single kernel launch, avoiding CPU-side loop overhead

struct TensorInfo {
    float* data;
    int64_t numel;
    int64_t offset;  // Offset in flat buffer
};

__global__ void fused_perturb_and_scatter_kernel(
    float* __restrict__ flat_params,     // Source: flat parameter buffer
    const float* __restrict__ z,          // Perturbation values
    float** __restrict__ model_params,    // Destination: model parameter pointers
    const int64_t* __restrict__ offsets,  // Offsets for each parameter
    const int64_t* __restrict__ sizes,    // Sizes for each parameter
    const int n_params,                   // Number of parameters
    const int64_t total_elements,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < total_elements) {
        // Apply perturbation to flat buffer
        float new_val = __fmaf_rn(alpha, z[idx], flat_params[idx]);
        flat_params[idx] = new_val;
        
        // Find which parameter this element belongs to (binary search)
        int param_idx = 0;
        int64_t cumsum = 0;
        for (int i = 0; i < n_params; i++) {
            if (idx < cumsum + sizes[i]) {
                param_idx = i;
                break;
            }
            cumsum += sizes[i];
        }
        
        // Copy to model parameter
        int64_t local_offset = idx - offsets[param_idx];
        model_params[param_idx][local_offset] = new_val;
    }
}


// =============================================================================
// Kernel 5: Half Precision Support (for larger models)
// =============================================================================

__global__ void perturb_add_half_kernel(
    __half* __restrict__ params,
    const __half* __restrict__ z,
    const int64_t n_elements,
    const __half alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_elements) {
        __half p = params[idx];
        __half z_val = z[idx];
        params[idx] = __hfma(alpha, z_val, p);
    }
}


// Vectorized half precision (8 elements at once)
__global__ void perturb_add_half8_kernel(
    float4* __restrict__ params,  // Actually 8 halfs packed as float4
    const float4* __restrict__ z,
    const int64_t n_float4,
    const float alpha_f
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n_float4) {
        float4 p = params[idx];
        float4 z_val = z[idx];
        
        // Unpack, compute, repack
        __half2* p_h2 = reinterpret_cast<__half2*>(&p);
        const __half2* z_h2 = reinterpret_cast<const __half2*>(&z_val);
        __half2 alpha_h2 = __float2half2_rn(alpha_f);
        
        #pragma unroll
        for (int i = 0; i < 4; i++) {
            p_h2[i] = __hfma2(alpha_h2, z_h2[i], p_h2[i]);
        }
        
        params[idx] = p;
    }
}


// =============================================================================
// Kernel 6: Async-friendly kernel with minimal sync points
// =============================================================================

// This kernel is designed to work with CUDA streams for async execution
// It processes a chunk of parameters, allowing overlap with CPU work

__global__ void perturb_chunk_kernel(
    float* __restrict__ params,
    const float* __restrict__ z,
    const int64_t start_idx,
    const int64_t end_idx,
    const float alpha
) {
    const int64_t idx = blockIdx.x * blockDim.x + threadIdx.x + start_idx;
    
    if (idx < end_idx) {
        params[idx] = __fmaf_rn(alpha, z[idx], params[idx]);
    }
}


// =============================================================================
// Host-side wrapper functions (extern "C" for Python ctypes/pybind)
// =============================================================================

extern "C" {

void launch_perturb_add(
    float* params,
    const float* z,
    int64_t n_elements,
    float alpha,
    cudaStream_t stream
) {
    const int grid_size = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    perturb_add_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        params, z, n_elements, alpha
    );
}


void launch_perturb_add_vec4(
    float* params,
    const float* z,
    int64_t n_elements,
    float alpha,
    cudaStream_t stream
) {
    const int64_t n_vec4 = n_elements / 4;
    const int grid_size = (n_vec4 + BLOCK_SIZE - 1) / BLOCK_SIZE;
    
    perturb_add_vec4_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        reinterpret_cast<float4*>(params),
        reinterpret_cast<const float4*>(z),
        n_vec4,
        alpha
    );
    
    // Handle remainder
    const int64_t remainder_start = n_vec4 * 4;
    if (remainder_start < n_elements) {
        const int remainder = n_elements - remainder_start;
        perturb_add_kernel<<<1, remainder, 0, stream>>>(
            params + remainder_start,
            z + remainder_start,
            remainder,
            alpha
        );
    }
}


void launch_perturb_with_rng(
    float* params,
    int64_t n_elements,
    float eps,
    uint64_t seed,
    cudaStream_t stream
) {
    const int grid_size = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    fused_perturb_with_rng_kernel<<<grid_size, BLOCK_SIZE, 0, stream>>>(
        params, n_elements, eps, seed
    );
}


// Async chunked execution for CPU overlap
void launch_perturb_async_chunks(
    float* params,
    const float* z,
    int64_t n_elements,
    float alpha,
    int n_chunks,
    cudaStream_t* streams
) {
    const int64_t chunk_size = (n_elements + n_chunks - 1) / n_chunks;
    
    for (int i = 0; i < n_chunks; i++) {
        const int64_t start = i * chunk_size;
        const int64_t end = min(start + chunk_size, n_elements);
        const int64_t chunk_elements = end - start;
        
        if (chunk_elements > 0) {
            const int grid_size = (chunk_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
            perturb_chunk_kernel<<<grid_size, BLOCK_SIZE, 0, streams[i]>>>(
                params, z, start, end, alpha
            );
        }
    }
}

}  // extern "C"
