/*
 * Optimized CUDA Kernels for DiZO zo_forward - V3
 * 
 * Key optimizations over V2 (matching Triton multi-block performance):
 * 1. Pre-computed block mapping (one block per BLOCK_SIZE elements)
 * 2. Atomic reductions (single kernel instead of compute + reduce)
 * 3. Higher block count for better SM utilization
 * 4. Simpler indexing without float4 complexity (let GPU handle coalescing)
 * 
 * Based on Triton multi-block approach that achieved 25x speedup.
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// Configuration - match Triton's approach
constexpr int BLOCK_SIZE = 2048;  // Elements per block (same as Triton)
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = BLOCK_SIZE / THREADS_PER_BLOCK;  // 8
constexpr int WARP_SIZE = 32;


// =============================================================================
// Warp-level reduction
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
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : T(0);
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}


// =============================================================================
// Kernel 1: Multi-block Norm with Atomic Reduction (single kernel!)
// =============================================================================

__global__ void fused_norm_atomic_kernel(
    const float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    float* __restrict__ partial_sums,     // [num_params] - atomic accumulation target
    const int* __restrict__ block_to_param,  // Which param group this block belongs to
    const int* __restrict__ block_start,     // Start index within param group
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    // Get which parameter group and starting position for this block
    int param_idx = block_to_param[block_id];
    int my_block_start = block_start[block_id];
    
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    // Calculate end position for this block
    int block_end = min(my_block_start + BLOCK_SIZE, (int)size);
    
    // Accumulate squared differences
    float acc = 0.0f;
    
    #pragma unroll 4
    for (int i = my_block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        float p = param_flat[offset + i];
        float a = anchor_flat[offset + i];
        float diff = p - a;
        acc += diff * diff;
    }
    
    // Block-level reduction
    acc = block_reduce_sum(acc);
    
    // Atomic add to parameter's accumulator
    if (threadIdx.x == 0) {
        atomicAdd(&partial_sums[param_idx], acc);
    }
}


// Finalize norms: sqrt of accumulated sums
__global__ void norm_sqrt_kernel(
    float* __restrict__ partial_sums,
    float* __restrict__ norms_out,
    int num_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;
    
    norms_out[idx] = sqrtf(partial_sums[idx] + 1e-8f);
}


// =============================================================================
// Kernel 2: Multi-block Constraint Application
// =============================================================================

__global__ void fused_apply_multiblock_kernel(
    float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const float* __restrict__ alphas,
    const int* __restrict__ block_to_param,
    const int* __restrict__ block_start,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    int param_idx = block_to_param[block_id];
    int my_block_start = block_start[block_id];
    
    float alpha = alphas[param_idx];
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    int block_end = min(my_block_start + BLOCK_SIZE, (int)size);
    
    #pragma unroll 4
    for (int i = my_block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        int64_t idx = offset + i;
        float p = param_flat[idx];
        float a = anchor_flat[idx];
        param_flat[idx] = a + (p - a) * alpha;
    }
}


// =============================================================================
// Kernel 3: Multi-block Constraint Reversal
// =============================================================================

__global__ void fused_reverse_multiblock_kernel(
    float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const float* __restrict__ alphas,
    const int* __restrict__ block_to_param,
    const int* __restrict__ block_start,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    int param_idx = block_to_param[block_id];
    int my_block_start = block_start[block_id];
    
    float alpha = alphas[param_idx];
    float inv_alpha = 1.0f / (alpha + 1e-8f);
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    int block_end = min(my_block_start + BLOCK_SIZE, (int)size);
    
    #pragma unroll 4
    for (int i = my_block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        int64_t idx = offset + i;
        float p = param_flat[idx];
        float a = anchor_flat[idx];
        param_flat[idx] = a + (p - a) * inv_alpha;
    }
}


// =============================================================================
// Kernel 4: Gamma Update
// =============================================================================

__global__ void fused_update_gamma_kernel(
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

// Cache for block mapping (to avoid recomputation)
struct BlockMappingCache {
    torch::Tensor block_to_param;
    torch::Tensor block_start;
    int num_blocks;
    int num_params;
    bool initialized;
    
    BlockMappingCache() : num_blocks(0), num_params(0), initialized(false) {}
};

// Global cache (will be set per-model)
static BlockMappingCache g_cache;


// Build block mapping for given sizes
void build_block_mapping(
    torch::Tensor sizes,
    torch::Device device
) {
    int num_params = sizes.size(0);
    auto sizes_cpu = sizes.to(torch::kCPU);
    auto sizes_acc = sizes_cpu.accessor<int64_t, 1>();
    
    std::vector<int> block_to_param_vec;
    std::vector<int> block_start_vec;
    
    for (int i = 0; i < num_params; i++) {
        int64_t size = sizes_acc[i];
        int num_blocks_for_param = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        for (int b = 0; b < num_blocks_for_param; b++) {
            block_to_param_vec.push_back(i);
            block_start_vec.push_back(b * BLOCK_SIZE);
        }
    }
    
    int total_blocks = block_to_param_vec.size();
    
    g_cache.block_to_param = torch::from_blob(
        block_to_param_vec.data(), {total_blocks}, torch::kInt32
    ).clone().to(device);
    
    g_cache.block_start = torch::from_blob(
        block_start_vec.data(), {total_blocks}, torch::kInt32
    ).clone().to(device);
    
    g_cache.num_blocks = total_blocks;
    g_cache.num_params = num_params;
    g_cache.initialized = true;
}


torch::Tensor fused_compute_norms_v3(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    // Build block mapping if not cached or if num_params changed
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping(sizes, device);
    }
    
    // Allocate outputs - partial_sums needs to be zeroed!
    auto partial_sums = torch::zeros({num_params}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    auto norms = torch::empty({num_params}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Launch multi-block norm kernel
    fused_norm_atomic_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        partial_sums.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
    
    // Launch sqrt kernel
    int sqrt_blocks = (num_params + 255) / 256;
    norm_sqrt_kernel<<<sqrt_blocks, 256>>>(
        partial_sums.data_ptr<float>(),
        norms.data_ptr<float>(),
        num_params
    );
    
    return norms;
}


void fused_apply_constraints_v3(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor constraints,
    torch::Tensor norms,
    float eps
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    // Build block mapping if needed
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping(sizes, device);
    }
    
    // Compute alphas
    auto alphas = constraints / (norms + eps);
    
    // Launch multi-block apply kernel
    fused_apply_multiblock_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        alphas.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
}


void fused_reverse_constraints_v3(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor alphas
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    // Build block mapping if needed
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping(sizes, device);
    }
    
    // Launch multi-block reverse kernel
    fused_reverse_multiblock_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        alphas.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
}


void fused_update_gamma_v3(
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
    
    fused_update_gamma_kernel<<<blocks, threads>>>(
        gamma.data_ptr<float>(),
        ts.data_ptr<float>(),
        zs.data_ptr<float>(),
        grad,
        step_size,
        tau,
        num_params
    );
}


// Function to pre-initialize block mapping
void init_block_mapping(torch::Tensor sizes) {
    auto device = sizes.device();
    build_block_mapping(sizes, device);
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compute_norms", &fused_compute_norms_v3, "Fused norm computation V3 (CUDA)");
    m.def("fused_apply_constraints", &fused_apply_constraints_v3, "Fused constraint application V3 (CUDA)");
    m.def("fused_reverse_constraints", &fused_reverse_constraints_v3, "Fused constraint reversal V3 (CUDA)");
    m.def("fused_update_gamma", &fused_update_gamma_v3, "Fused gamma update V3 (CUDA)");
    m.def("init_block_mapping", &init_block_mapping, "Initialize block mapping cache");
}

#endif  // USE_TORCH
