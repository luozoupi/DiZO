/*
 * Optimized CUDA Kernels for DiZO zo_forward - V4
 * 
 * Key insight from Triton analysis:
 * - Triton uses 2048 elements per block, one block per chunk
 * - CUDA V2 was capping at 32 blocks per param (3200 elements per thread!)
 * - V3's atomics add overhead that negates parallelism gains
 * 
 * V4 Strategy:
 * 1. Use same block mapping as Triton (2048 elements per block)
 * 2. Two-phase reduction WITHOUT atomics (compute -> reduce)
 * 3. Store partial sums to global memory, then single reduction kernel
 * 4. Pre-computed block mapping cached on GPU
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// Configuration - exactly match Triton
constexpr int BLOCK_SIZE = 2048;  // Elements per block
constexpr int THREADS_PER_BLOCK = 256;
constexpr int WARP_SIZE = 32;


// =============================================================================
// Warp-level reduction (no shared memory needed for small reductions)
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[32];  // One per warp
    
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}


// =============================================================================
// Kernel 1a: Multi-block Norm Computation (partial sums)
// One block per 2048 elements, stores partial sum
// =============================================================================

__global__ void fused_norm_partial_kernel(
    const float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    float* __restrict__ partial_sums,        // [num_blocks] - one per block
    const int* __restrict__ block_to_param,  // Which param group this block belongs to
    const int* __restrict__ block_start,     // Start index within param group
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    int num_blocks
) {
    int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    int param_idx = block_to_param[block_id];
    int my_block_start = block_start[block_id];
    
    int64_t offset = offsets[param_idx];
    int64_t size = sizes[param_idx];
    
    int block_end = min(my_block_start + BLOCK_SIZE, (int)size);
    
    // Each thread accumulates multiple elements
    float acc = 0.0f;
    
    #pragma unroll 8
    for (int i = my_block_start + threadIdx.x; i < block_end; i += blockDim.x) {
        float p = param_flat[offset + i];
        float a = anchor_flat[offset + i];
        float diff = p - a;
        acc += diff * diff;
    }
    
    // Block-level reduction
    acc = block_reduce_sum(acc);
    
    // Store this block's partial sum
    if (threadIdx.x == 0) {
        partial_sums[block_id] = acc;
    }
}


// =============================================================================
// Kernel 1b: Final Norm Reduction
// Sum partial sums for each parameter group, then sqrt
// =============================================================================

__global__ void norm_reduce_final_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ norms_out,
    const int* __restrict__ param_block_starts,  // Where each param's blocks start
    const int* __restrict__ param_block_counts,  // How many blocks per param
    int num_params
) {
    int param_idx = blockIdx.x;
    if (param_idx >= num_params) return;
    
    int start = param_block_starts[param_idx];
    int count = param_block_counts[param_idx];
    
    // Sum partial results (one thread block per param)
    float sum = 0.0f;
    for (int i = threadIdx.x; i < count; i += blockDim.x) {
        sum += partial_sums[start + i];
    }
    
    sum = block_reduce_sum(sum);
    
    if (threadIdx.x == 0) {
        norms_out[param_idx] = sqrtf(sum + 1e-8f);
    }
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
    
    #pragma unroll 8
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
    
    #pragma unroll 8
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

// Cache for block mapping
struct BlockMappingCache {
    torch::Tensor block_to_param;
    torch::Tensor block_start;
    torch::Tensor param_block_starts;  // Where each param's blocks begin in the list
    torch::Tensor param_block_counts;  // How many blocks per param
    torch::Tensor partial_sums;        // Pre-allocated buffer for partial sums
    int num_blocks;
    int num_params;
    bool initialized;
    
    BlockMappingCache() : num_blocks(0), num_params(0), initialized(false) {}
};

static BlockMappingCache g_cache;


void build_block_mapping(torch::Tensor sizes, torch::Device device) {
    int num_params = sizes.size(0);
    auto sizes_cpu = sizes.to(torch::kCPU);
    auto sizes_acc = sizes_cpu.accessor<int64_t, 1>();
    
    std::vector<int> block_to_param_vec;
    std::vector<int> block_start_vec;
    std::vector<int> param_block_starts_vec;
    std::vector<int> param_block_counts_vec;
    
    int current_block = 0;
    for (int i = 0; i < num_params; i++) {
        int64_t size = sizes_acc[i];
        int num_blocks_for_param = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
        
        param_block_starts_vec.push_back(current_block);
        param_block_counts_vec.push_back(num_blocks_for_param);
        
        for (int b = 0; b < num_blocks_for_param; b++) {
            block_to_param_vec.push_back(i);
            block_start_vec.push_back(b * BLOCK_SIZE);
            current_block++;
        }
    }
    
    int total_blocks = block_to_param_vec.size();
    
    g_cache.block_to_param = torch::from_blob(
        block_to_param_vec.data(), {total_blocks}, torch::kInt32
    ).clone().to(device);
    
    g_cache.block_start = torch::from_blob(
        block_start_vec.data(), {total_blocks}, torch::kInt32
    ).clone().to(device);
    
    g_cache.param_block_starts = torch::from_blob(
        param_block_starts_vec.data(), {num_params}, torch::kInt32
    ).clone().to(device);
    
    g_cache.param_block_counts = torch::from_blob(
        param_block_counts_vec.data(), {num_params}, torch::kInt32
    ).clone().to(device);
    
    // Pre-allocate partial sums buffer
    g_cache.partial_sums = torch::empty({total_blocks}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    g_cache.num_blocks = total_blocks;
    g_cache.num_params = num_params;
    g_cache.initialized = true;
}


torch::Tensor fused_compute_norms_v4(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping(sizes, device);
    }
    
    auto norms = torch::empty({num_params}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Phase 1: Compute partial sums (one block per 2048 elements)
    fused_norm_partial_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        g_cache.partial_sums.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
    
    // Phase 2: Reduce partial sums per parameter and sqrt
    norm_reduce_final_kernel<<<num_params, THREADS_PER_BLOCK>>>(
        g_cache.partial_sums.data_ptr<float>(),
        norms.data_ptr<float>(),
        g_cache.param_block_starts.data_ptr<int>(),
        g_cache.param_block_counts.data_ptr<int>(),
        num_params
    );
    
    return norms;
}


void fused_apply_constraints_v4(
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
    
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping(sizes, device);
    }
    
    auto alphas = constraints / (norms + eps);
    
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


void fused_reverse_constraints_v4(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor alphas
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping(sizes, device);
    }
    
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


void fused_update_gamma_v4(
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


void init_block_mapping(torch::Tensor sizes) {
    build_block_mapping(sizes, sizes.device());
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compute_norms", &fused_compute_norms_v4, "Fused norm computation V4 (CUDA)");
    m.def("fused_apply_constraints", &fused_apply_constraints_v4, "Fused constraint application V4 (CUDA)");
    m.def("fused_reverse_constraints", &fused_reverse_constraints_v4, "Fused constraint reversal V4 (CUDA)");
    m.def("fused_update_gamma", &fused_update_gamma_v4, "Fused gamma update V4 (CUDA)");
    m.def("init_block_mapping", &init_block_mapping, "Initialize block mapping cache");
}

#endif  // USE_TORCH
