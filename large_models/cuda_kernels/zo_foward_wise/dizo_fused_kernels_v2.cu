/*
 * Optimized CUDA Kernels for DiZO zo_forward - V2
 * 
 * Key optimizations over V1:
 * 1. Float4 vectorized memory access (4x bandwidth)
 * 2. Multi-block parallelism per parameter group
 * 3. Warp-level reductions (faster on H100/H200)
 * 4. Grid-stride loops for better occupancy
 * 
 * Performance targets:
 * - Norm computation: 3-4x faster than per-group PyTorch
 * - Constraint apply: 2-3x faster
 * 
 * Compile: TORCH_CUDA_ARCH_LIST="9.0" python setup_v2.py install
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// Warp size
constexpr int WARP_SIZE = 32;

// =============================================================================
// Helper: Warp-level reduction (faster than shared memory on modern GPUs)
// =============================================================================

template<typename T>
__device__ __forceinline__ T warp_reduce_sum(T val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

template<typename T>
__device__ __forceinline__ T block_reduce_sum(T val) {
    __shared__ T shared[32];  // Max 32 warps per block
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write warp results to shared memory
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    // First warp reduces across warps
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}


// =============================================================================
// Kernel 1: Vectorized Norm Computation with Float4
// =============================================================================

__global__ void fused_norm_kernel_v2(
    const float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    float* __restrict__ norms_out,
    float* __restrict__ partial_sums,  // For multi-block reduction
    int num_params,
    int blocks_per_param
) {
    // Which parameter group and which block within that group
    int param_idx = blockIdx.x / blocks_per_param;
    int block_within_param = blockIdx.x % blocks_per_param;
    
    if (param_idx >= num_params) return;
    
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    // Each block handles a portion of the parameter
    int64_t elements_per_block = (size + blocks_per_param - 1) / blocks_per_param;
    int64_t block_start = block_within_param * elements_per_block;
    int64_t block_end = min(block_start + elements_per_block, size);
    
    // Grid-stride loop with float4 vectorization
    float acc = 0.0f;
    
    // Handle float4-aligned portion
    int64_t aligned_start = ((offset + block_start + 3) / 4) * 4 - offset;
    int64_t aligned_end = ((offset + block_end) / 4) * 4 - offset;
    
    // Process aligned float4 elements
    const float4* param_vec = reinterpret_cast<const float4*>(param_flat + offset);
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor_flat + offset);
    
    int64_t vec_start = aligned_start / 4;
    int64_t vec_end = aligned_end / 4;
    
    for (int64_t i = vec_start + threadIdx.x; i < vec_end; i += blockDim.x) {
        float4 p = param_vec[i];
        float4 a = anchor_vec[i];
        
        float d0 = p.x - a.x;
        float d1 = p.y - a.y;
        float d2 = p.z - a.z;
        float d3 = p.w - a.w;
        
        acc += d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3;
    }
    
    // Handle unaligned head
    for (int64_t i = block_start + threadIdx.x; i < aligned_start && i < block_end; i += blockDim.x) {
        float diff = param_flat[offset + i] - anchor_flat[offset + i];
        acc += diff * diff;
    }
    
    // Handle unaligned tail
    for (int64_t i = aligned_end + threadIdx.x; i < block_end; i += blockDim.x) {
        float diff = param_flat[offset + i] - anchor_flat[offset + i];
        acc += diff * diff;
    }
    
    // Block-level reduction
    acc = block_reduce_sum(acc);
    
    // Store partial sum
    if (threadIdx.x == 0) {
        partial_sums[blockIdx.x] = acc;
    }
}


// Final reduction kernel (run after all blocks complete)
__global__ void norm_final_reduce_kernel(
    float* __restrict__ partial_sums,
    float* __restrict__ norms_out,
    int num_params,
    int blocks_per_param
) {
    int param_idx = blockIdx.x;
    if (param_idx >= num_params) return;
    
    // Sum partial results for this parameter
    float sum = 0.0f;
    int base = param_idx * blocks_per_param;
    
    for (int i = threadIdx.x; i < blocks_per_param; i += blockDim.x) {
        sum += partial_sums[base + i];
    }
    
    sum = block_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        norms_out[param_idx] = sqrtf(sum + 1e-8f);
    }
}


// =============================================================================
// Kernel 2: Vectorized Constraint Application
// =============================================================================

__global__ void fused_apply_constraints_kernel_v2(
    float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const float* __restrict__ constraints,
    const float* __restrict__ norms,
    int num_params,
    int blocks_per_param,
    float eps
) {
    int param_idx = blockIdx.x / blocks_per_param;
    int block_within_param = blockIdx.x % blocks_per_param;
    
    if (param_idx >= num_params) return;
    
    float constraint = constraints[param_idx];
    float norm = norms[param_idx];
    float alpha = constraint / (norm + eps);
    
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    int64_t elements_per_block = (size + blocks_per_param - 1) / blocks_per_param;
    int64_t block_start = block_within_param * elements_per_block;
    int64_t block_end = min(block_start + elements_per_block, size);
    
    // Vectorized processing
    int64_t aligned_start = ((offset + block_start + 3) / 4) * 4 - offset;
    int64_t aligned_end = ((offset + block_end) / 4) * 4 - offset;
    
    float4* param_vec = reinterpret_cast<float4*>(param_flat + offset);
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor_flat + offset);
    
    int64_t vec_start = aligned_start / 4;
    int64_t vec_end = aligned_end / 4;
    
    for (int64_t i = vec_start + threadIdx.x; i < vec_end; i += blockDim.x) {
        float4 p = param_vec[i];
        float4 a = anchor_vec[i];
        
        float4 result;
        result.x = a.x + (p.x - a.x) * alpha;
        result.y = a.y + (p.y - a.y) * alpha;
        result.z = a.z + (p.z - a.z) * alpha;
        result.w = a.w + (p.w - a.w) * alpha;
        
        param_vec[i] = result;
    }
    
    // Handle unaligned portions
    for (int64_t i = block_start + threadIdx.x; i < aligned_start && i < block_end; i += blockDim.x) {
        int64_t idx = offset + i;
        float p = param_flat[idx];
        float a = anchor_flat[idx];
        param_flat[idx] = a + (p - a) * alpha;
    }
    
    for (int64_t i = aligned_end + threadIdx.x; i < block_end; i += blockDim.x) {
        int64_t idx = offset + i;
        float p = param_flat[idx];
        float a = anchor_flat[idx];
        param_flat[idx] = a + (p - a) * alpha;
    }
}


// =============================================================================
// Kernel 3: Vectorized Constraint Reversal
// =============================================================================

__global__ void fused_reverse_constraints_kernel_v2(
    float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const float* __restrict__ alphas,
    int num_params,
    int blocks_per_param
) {
    int param_idx = blockIdx.x / blocks_per_param;
    int block_within_param = blockIdx.x % blocks_per_param;
    
    if (param_idx >= num_params) return;
    
    float alpha = alphas[param_idx];
    float inv_alpha = 1.0f / alpha;
    
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    int64_t elements_per_block = (size + blocks_per_param - 1) / blocks_per_param;
    int64_t block_start = block_within_param * elements_per_block;
    int64_t block_end = min(block_start + elements_per_block, size);
    
    // Vectorized processing
    int64_t aligned_start = ((offset + block_start + 3) / 4) * 4 - offset;
    int64_t aligned_end = ((offset + block_end) / 4) * 4 - offset;
    
    float4* param_vec = reinterpret_cast<float4*>(param_flat + offset);
    const float4* anchor_vec = reinterpret_cast<const float4*>(anchor_flat + offset);
    
    int64_t vec_start = aligned_start / 4;
    int64_t vec_end = aligned_end / 4;
    
    for (int64_t i = vec_start + threadIdx.x; i < vec_end; i += blockDim.x) {
        float4 p = param_vec[i];
        float4 a = anchor_vec[i];
        
        float4 result;
        result.x = a.x + (p.x - a.x) * inv_alpha;
        result.y = a.y + (p.y - a.y) * inv_alpha;
        result.z = a.z + (p.z - a.z) * inv_alpha;
        result.w = a.w + (p.w - a.w) * inv_alpha;
        
        param_vec[i] = result;
    }
    
    // Handle unaligned portions
    for (int64_t i = block_start + threadIdx.x; i < aligned_start && i < block_end; i += blockDim.x) {
        int64_t idx = offset + i;
        float p = param_flat[idx];
        float a = anchor_flat[idx];
        param_flat[idx] = a + (p - a) * inv_alpha;
    }
    
    for (int64_t i = aligned_end + threadIdx.x; i < block_end; i += blockDim.x) {
        int64_t idx = offset + i;
        float p = param_flat[idx];
        float a = anchor_flat[idx];
        param_flat[idx] = a + (p - a) * inv_alpha;
    }
}


// =============================================================================
// Kernel 4: Gamma Update (unchanged - already efficient for small tensors)
// =============================================================================

__global__ void fused_update_gamma_kernel_v2(
    float* __restrict__ gamma,
    const float* __restrict__ ts,
    const float* __restrict__ zs,
    float grad,
    float step_size,
    float tau,
    int num_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;
    
    float gamma_val = gamma[idx];
    float t = ts[idx];
    float z = zs[idx];
    
    float gamma_new = gamma_val - step_size * t * grad * z;
    
    float gamma_min = (1.0f - tau) * t;
    float gamma_max = (1.0f + tau) * t;
    
    gamma[idx] = fminf(fmaxf(gamma_new, gamma_min), gamma_max);
}


// =============================================================================
// PyTorch Extension Interface
// =============================================================================

#ifdef USE_TORCH

// Determine optimal blocks per parameter based on size
int get_blocks_per_param(int64_t max_size, int target_threads_per_block = 256) {
    // Target: each thread handles ~1024 elements
    int64_t target_elements_per_thread = 1024;
    int64_t total_threads_needed = max_size / target_elements_per_thread;
    int blocks = (total_threads_needed + target_threads_per_block - 1) / target_threads_per_block;
    // Clamp to reasonable range
    return std::max(1, std::min(blocks, 32));
}


torch::Tensor fused_compute_norms_v2(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    // Find max size to determine parallelism
    int64_t max_size = sizes.max().item<int64_t>();
    int blocks_per_param = get_blocks_per_param(max_size);
    
    // Allocate outputs
    auto norms = torch::empty({num_params}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto partial_sums = torch::empty({num_params * blocks_per_param}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Launch norm kernel
    int total_blocks = num_params * blocks_per_param;
    int threads = 256;
    
    fused_norm_kernel_v2<<<total_blocks, threads>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        norms.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        num_params,
        blocks_per_param
    );
    
    // Launch final reduction
    norm_final_reduce_kernel<<<num_params, 256>>>(
        partial_sums.data_ptr<float>(),
        norms.data_ptr<float>(),
        num_params,
        blocks_per_param
    );
    
    return norms;
}


void fused_apply_constraints_v2(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor constraints,
    torch::Tensor norms,
    float eps
) {
    int num_params = offsets.size(0);
    
    int64_t max_size = sizes.max().item<int64_t>();
    int blocks_per_param = get_blocks_per_param(max_size);
    
    int total_blocks = num_params * blocks_per_param;
    int threads = 256;
    
    fused_apply_constraints_kernel_v2<<<total_blocks, threads>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        constraints.data_ptr<float>(),
        norms.data_ptr<float>(),
        num_params,
        blocks_per_param,
        eps
    );
}


void fused_reverse_constraints_v2(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor alphas
) {
    int num_params = offsets.size(0);
    
    int64_t max_size = sizes.max().item<int64_t>();
    int blocks_per_param = get_blocks_per_param(max_size);
    
    int total_blocks = num_params * blocks_per_param;
    int threads = 256;
    
    fused_reverse_constraints_kernel_v2<<<total_blocks, threads>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        alphas.data_ptr<float>(),
        num_params,
        blocks_per_param
    );
}


void fused_update_gamma_v2(
    torch::Tensor gamma,
    torch::Tensor ts,
    torch::Tensor zs,
    float grad,
    float step_size,
    float tau
) {
    int num_params = gamma.size(0);
    int threads = 256;
    int blocks = (num_params + threads - 1) / threads;
    
    fused_update_gamma_kernel_v2<<<blocks, threads>>>(
        gamma.data_ptr<float>(),
        ts.data_ptr<float>(),
        zs.data_ptr<float>(),
        grad,
        step_size,
        tau,
        num_params
    );
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compute_norms", &fused_compute_norms_v2, "Fused norm computation V2 (CUDA)");
    m.def("fused_apply_constraints", &fused_apply_constraints_v2, "Fused constraint application V2 (CUDA)");
    m.def("fused_reverse_constraints", &fused_reverse_constraints_v2, "Fused constraint reversal V2 (CUDA)");
    m.def("fused_update_gamma", &fused_update_gamma_v2, "Fused gamma update V2 (CUDA)");
}

#endif  // USE_TORCH
