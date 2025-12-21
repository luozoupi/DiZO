/*
 * Optimized CUDA Kernels for DiZO zo_forward - V5
 * 
 * Key optimizations matching Triton multi-block performance:
 * 1. Block mapping: 2048 elements per block (matching Triton wrapper's BLOCK_SIZE)
 * 2. Pre-computed block mapping cached on GPU (avoids runtime computation)
 * 3. Coalesced memory access with vectorized float4 loads and __ldg intrinsics
 * 4. Efficient two-phase reduction without atomics
 * 5. Persistent partial_sums buffer (pre-allocated)
 * 6. Unrolled loops with #pragma unroll and FMA intrinsics
 * 
 * The critical fix: V2 used max 32 blocks/param (3200 elements/thread)
 * V5 uses ~6400 blocks/param for large params (8 elements/thread) like Triton
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>
#include <vector>

#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// Match Triton wrapper's block mapping (2048 elements per block)
constexpr int ELEMENTS_PER_BLOCK = 2048;
constexpr int THREADS_PER_BLOCK = 256;
constexpr int ELEMENTS_PER_THREAD = ELEMENTS_PER_BLOCK / THREADS_PER_BLOCK;  // 8
constexpr int WARP_SIZE = 32;


// =============================================================================
// Fast warp/block reductions
// =============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float shared[WARP_SIZE];
    
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int num_warps = THREADS_PER_BLOCK / WARP_SIZE;
    
    val = warp_reduce_sum(val);
    
    if (lane == 0) {
        shared[warp_id] = val;
    }
    __syncthreads();
    
    val = (threadIdx.x < num_warps) ? shared[threadIdx.x] : 0.0f;
    
    if (warp_id == 0) {
        val = warp_reduce_sum(val);
    }
    
    return val;
}


// =============================================================================
// Kernel 1: Multi-block Norm with Triton-style parallelism + float4 vectorization
// =============================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_norm_multiblock_kernel(
    const float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    float* __restrict__ partial_sums,
    const int* __restrict__ block_to_param,
    const int* __restrict__ block_start_idx,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const int num_blocks
) {
    const int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    const int param_idx = __ldg(&block_to_param[block_id]);
    const int block_start = __ldg(&block_start_idx[block_id]);
    const int64_t offset = __ldg(&offsets[param_idx]);
    const int64_t size = __ldg(&sizes[param_idx]);
    const int block_end = min(block_start + ELEMENTS_PER_BLOCK, (int)size);
    const int block_len = block_end - block_start;
    
    const float* p_ptr = param_flat + offset + block_start;
    const float* a_ptr = anchor_flat + offset + block_start;
    
    float acc = 0.0f;
    
    // Process float4 aligned portion (4 elements at a time) with ldg
    const int vec_len = block_len / 4;
    const float4* p_vec = reinterpret_cast<const float4*>(p_ptr);
    const float4* a_vec = reinterpret_cast<const float4*>(a_ptr);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_len; i += THREADS_PER_BLOCK) {
        float4 p4 = __ldg(&p_vec[i]);
        float4 a4 = __ldg(&a_vec[i]);
        
        float d0 = p4.x - a4.x;
        float d1 = p4.y - a4.y;
        float d2 = p4.z - a4.z;
        float d3 = p4.w - a4.w;
        
        acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    
    // Handle remaining elements
    const int remainder_start = vec_len * 4;
    for (int i = remainder_start + threadIdx.x; i < block_len; i += THREADS_PER_BLOCK) {
        float diff = __ldg(&p_ptr[i]) - __ldg(&a_ptr[i]);
        acc += diff * diff;
    }
    
    // Block reduction
    acc = block_reduce_sum(acc);
    
    if (threadIdx.x == 0) {
        partial_sums[block_id] = acc;
    }
}


// =============================================================================
// Kernel 1b: Direct atomic norm kernel - avoids two-phase reduction
// Uses atomicAdd on norms_out (only num_params atomics, not millions)
// =============================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_norm_atomic_kernel(
    const float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    float* __restrict__ norms_out,  // Direct atomic accumulation
    const int* __restrict__ block_to_param,
    const int* __restrict__ block_start_idx,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const int num_blocks
) {
    const int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    const int param_idx = __ldg(&block_to_param[block_id]);
    const int block_start = __ldg(&block_start_idx[block_id]);
    const int64_t offset = __ldg(&offsets[param_idx]);
    const int64_t size = __ldg(&sizes[param_idx]);
    const int block_end = min(block_start + ELEMENTS_PER_BLOCK, (int)size);
    const int block_len = block_end - block_start;
    
    const float* p_ptr = param_flat + offset + block_start;
    const float* a_ptr = anchor_flat + offset + block_start;
    
    float acc = 0.0f;
    
    // Process float4 aligned portion
    const int vec_len = block_len / 4;
    const float4* p_vec = reinterpret_cast<const float4*>(p_ptr);
    const float4* a_vec = reinterpret_cast<const float4*>(a_ptr);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_len; i += THREADS_PER_BLOCK) {
        float4 p4 = __ldg(&p_vec[i]);
        float4 a4 = __ldg(&a_vec[i]);
        
        float d0 = p4.x - a4.x;
        float d1 = p4.y - a4.y;
        float d2 = p4.z - a4.z;
        float d3 = p4.w - a4.w;
        
        acc += d0*d0 + d1*d1 + d2*d2 + d3*d3;
    }
    
    // Handle remaining elements
    const int remainder_start = vec_len * 4;
    for (int i = remainder_start + threadIdx.x; i < block_len; i += THREADS_PER_BLOCK) {
        float diff = __ldg(&p_ptr[i]) - __ldg(&a_ptr[i]);
        acc += diff * diff;
    }
    
    // Block reduction
    acc = block_reduce_sum(acc);
    
    // Atomic add to norms_out (cheap - only ~500 blocks per param contending)
    if (threadIdx.x == 0 && acc > 0.0f) {
        atomicAdd(&norms_out[param_idx], acc);
    }
}


// Kernel to compute sqrt of accumulated norms
__global__ void norm_sqrt_kernel(float* norms, int num_params) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < num_params) {
        norms[idx] = sqrtf(norms[idx] + 1e-8f);
    }
}


// =============================================================================
// Kernel 2: Final norm reduction - sum partial_sums per param, then sqrt
// =============================================================================

__global__ void norm_reduce_sqrt_kernel(
    const float* __restrict__ partial_sums,
    float* __restrict__ norms_out,
    const int* __restrict__ param_block_offset,  // Where each param's blocks start
    const int* __restrict__ param_block_count,   // How many blocks per param
    const int num_params
) {
    const int param_idx = blockIdx.x;
    if (param_idx >= num_params) return;
    
    const int start = param_block_offset[param_idx];
    const int count = param_block_count[param_idx];
    
    // Each thread sums a portion of the partial sums
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
// Kernel 3: Multi-block constraint application with float4 vectorization
// =============================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_apply_multiblock_kernel(
    float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const float* __restrict__ alphas,
    const int* __restrict__ block_to_param,
    const int* __restrict__ block_start_idx,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const int num_blocks
) {
    const int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    // Load block metadata once and share via registers (L1 cached anyway)
    const int param_idx = block_to_param[block_id];
    const int block_start = block_start_idx[block_id];
    const int64_t offset = offsets[param_idx];
    const int64_t size = sizes[param_idx];
    const int block_end = min(block_start + ELEMENTS_PER_BLOCK, (int)size);
    const int block_len = block_end - block_start;
    
    // Use ldg for read-only alpha value
    const float alpha = __ldg(&alphas[param_idx]);
    
    float* p_ptr = param_flat + offset + block_start;
    const float* a_ptr = anchor_flat + offset + block_start;
    
    // Process float4 aligned portion with unrolled loop
    const int vec_len = block_len / 4;
    float4* p_vec = reinterpret_cast<float4*>(p_ptr);
    const float4* a_vec = reinterpret_cast<const float4*>(a_ptr);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_len; i += THREADS_PER_BLOCK) {
        // Use ldg for read-only anchor
        float4 p4 = p_vec[i];
        float4 a4 = __ldg(&a_vec[i]);
        
        // Fused multiply-add operations
        float4 result;
        result.x = __fmaf_rn(p4.x - a4.x, alpha, a4.x);
        result.y = __fmaf_rn(p4.y - a4.y, alpha, a4.y);
        result.z = __fmaf_rn(p4.z - a4.z, alpha, a4.z);
        result.w = __fmaf_rn(p4.w - a4.w, alpha, a4.w);
        
        p_vec[i] = result;
    }
    
    // Handle remaining elements
    const int remainder_start = vec_len * 4;
    for (int i = remainder_start + threadIdx.x; i < block_len; i += THREADS_PER_BLOCK) {
        float p = p_ptr[i];
        float a = __ldg(&a_ptr[i]);
        p_ptr[i] = __fmaf_rn(p - a, alpha, a);
    }
}


// =============================================================================
// Kernel 4: Multi-block constraint reversal with float4 vectorization
// =============================================================================

__global__ void __launch_bounds__(THREADS_PER_BLOCK)
fused_reverse_multiblock_kernel(
    float* __restrict__ param_flat,
    const float* __restrict__ anchor_flat,
    const float* __restrict__ alphas,
    const int* __restrict__ block_to_param,
    const int* __restrict__ block_start_idx,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const int num_blocks
) {
    const int block_id = blockIdx.x;
    if (block_id >= num_blocks) return;
    
    const int param_idx = block_to_param[block_id];
    const int block_start = block_start_idx[block_id];
    const float alpha = __ldg(&alphas[param_idx]);
    const float inv_alpha = __frcp_rn(alpha + 1e-8f);  // Fast reciprocal
    const int64_t offset = offsets[param_idx];
    const int64_t size = sizes[param_idx];
    const int block_end = min(block_start + ELEMENTS_PER_BLOCK, (int)size);
    const int block_len = block_end - block_start;
    
    float* p_ptr = param_flat + offset + block_start;
    const float* a_ptr = anchor_flat + offset + block_start;
    
    // Process float4 aligned portion with unrolled loop
    const int vec_len = block_len / 4;
    float4* p_vec = reinterpret_cast<float4*>(p_ptr);
    const float4* a_vec = reinterpret_cast<const float4*>(a_ptr);
    
    #pragma unroll 4
    for (int i = threadIdx.x; i < vec_len; i += THREADS_PER_BLOCK) {
        float4 p4 = p_vec[i];
        float4 a4 = __ldg(&a_vec[i]);
        
        float4 result;
        result.x = __fmaf_rn(p4.x - a4.x, inv_alpha, a4.x);
        result.y = __fmaf_rn(p4.y - a4.y, inv_alpha, a4.y);
        result.z = __fmaf_rn(p4.z - a4.z, inv_alpha, a4.z);
        result.w = __fmaf_rn(p4.w - a4.w, inv_alpha, a4.w);
        
        p_vec[i] = result;
    }
    
    // Handle remaining elements
    const int remainder_start = vec_len * 4;
    for (int i = remainder_start + threadIdx.x; i < block_len; i += THREADS_PER_BLOCK) {
        float p = p_ptr[i];
        float a = __ldg(&a_ptr[i]);
        p_ptr[i] = __fmaf_rn(p - a, inv_alpha, a);
    }
}


// =============================================================================
// Kernel 5: Gamma update (same as before, already optimized)
// =============================================================================

__global__ void fused_update_gamma_kernel(
    float* __restrict__ gamma,
    const float* __restrict__ ts,
    const float* __restrict__ zs,
    const float grad,
    const float step_size,
    const float tau,
    const int num_params
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_params) return;
    
    const float g = gamma[idx];
    const float t = ts[idx];
    const float z = zs[idx];
    
    float g_new = g - step_size * t * grad * z;
    
    const float g_min = (1.0f - tau) * t;
    const float g_max = (1.0f + tau) * t;
    
    gamma[idx] = fminf(fmaxf(g_new, g_min), g_max);
}


// =============================================================================
// PyTorch Extension Interface with Cached Block Mapping
// =============================================================================

#ifdef USE_TORCH

// Persistent cache for block mapping (computed once per model)
struct BlockMappingCache {
    torch::Tensor block_to_param;      // [num_blocks] - which param each block belongs to
    torch::Tensor block_start_idx;     // [num_blocks] - start index within param
    torch::Tensor param_block_offset;  // [num_params] - where each param's blocks start
    torch::Tensor param_block_count;   // [num_params] - how many blocks per param
    torch::Tensor partial_sums;        // [num_blocks] - pre-allocated buffer
    int num_blocks = 0;
    int num_params = 0;
    bool initialized = false;
};

static BlockMappingCache g_cache;


void build_block_mapping_v5(const torch::Tensor& sizes, torch::Device device) {
    const int num_params = sizes.size(0);
    auto sizes_cpu = sizes.to(torch::kCPU);
    auto sizes_acc = sizes_cpu.accessor<int64_t, 1>();
    
    std::vector<int> block_to_param_vec;
    std::vector<int> block_start_vec;
    std::vector<int> param_block_offset_vec;
    std::vector<int> param_block_count_vec;
    
    block_to_param_vec.reserve(num_params * 100);  // Estimated
    block_start_vec.reserve(num_params * 100);
    
    int global_block_idx = 0;
    for (int i = 0; i < num_params; i++) {
        const int64_t size = sizes_acc[i];
        const int num_blocks_for_param = (size + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
        
        param_block_offset_vec.push_back(global_block_idx);
        param_block_count_vec.push_back(num_blocks_for_param);
        
        for (int b = 0; b < num_blocks_for_param; b++) {
            block_to_param_vec.push_back(i);
            block_start_vec.push_back(b * ELEMENTS_PER_BLOCK);
        }
        global_block_idx += num_blocks_for_param;
    }
    
    const int total_blocks = global_block_idx;
    
    // Copy to GPU tensors
    g_cache.block_to_param = torch::from_blob(
        block_to_param_vec.data(), {total_blocks}, torch::kInt32
    ).clone().to(device);
    
    g_cache.block_start_idx = torch::from_blob(
        block_start_vec.data(), {total_blocks}, torch::kInt32
    ).clone().to(device);
    
    g_cache.param_block_offset = torch::from_blob(
        param_block_offset_vec.data(), {num_params}, torch::kInt32
    ).clone().to(device);
    
    g_cache.param_block_count = torch::from_blob(
        param_block_count_vec.data(), {num_params}, torch::kInt32
    ).clone().to(device);
    
    // Pre-allocate partial sums buffer
    g_cache.partial_sums = torch::empty(
        {total_blocks}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device)
    );
    
    g_cache.num_blocks = total_blocks;
    g_cache.num_params = num_params;
    g_cache.initialized = true;
}


torch::Tensor fused_compute_norms_v5(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes
) {
    const int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    // Build/rebuild cache if needed
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping_v5(sizes, device);
    }
    
    auto norms = torch::empty({num_params}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Phase 1: Compute partial sums (many blocks, each handles 2048 elements)
    fused_norm_multiblock_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        g_cache.partial_sums.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start_idx.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
    
    // Phase 2: Reduce partial sums per param and sqrt
    norm_reduce_sqrt_kernel<<<num_params, THREADS_PER_BLOCK>>>(
        g_cache.partial_sums.data_ptr<float>(),
        norms.data_ptr<float>(),
        g_cache.param_block_offset.data_ptr<int>(),
        g_cache.param_block_count.data_ptr<int>(),
        num_params
    );
    
    return norms;
}


// Alternative: Compute norms using atomic reduction (like Triton)
torch::Tensor fused_compute_norms_atomic(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes
) {
    const int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    // Build/rebuild cache if needed
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping_v5(sizes, device);
    }
    
    // Zero-initialize norms for atomic accumulation
    auto norms = torch::zeros({num_params}, 
        torch::TensorOptions().dtype(torch::kFloat32).device(device));
    
    // Single phase: Compute partial sums and atomic add directly to norms
    fused_norm_atomic_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        norms.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start_idx.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
    
    // Phase 2: sqrt only
    const int threads = 256;
    const int blocks = (num_params + threads - 1) / threads;
    norm_sqrt_kernel<<<blocks, threads>>>(norms.data_ptr<float>(), num_params);
    
    return norms;
}


void fused_apply_constraints_v5(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor constraints,
    torch::Tensor norms,
    float eps
) {
    const int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping_v5(sizes, device);
    }
    
    // Compute alphas on GPU
    auto alphas = constraints / (norms + eps);
    
    fused_apply_multiblock_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        alphas.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start_idx.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
}


void fused_reverse_constraints_v5(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor alphas
) {
    const int num_params = offsets.size(0);
    auto device = param_flat.device();
    
    if (!g_cache.initialized || g_cache.num_params != num_params) {
        build_block_mapping_v5(sizes, device);
    }
    
    fused_reverse_multiblock_kernel<<<g_cache.num_blocks, THREADS_PER_BLOCK>>>(
        param_flat.data_ptr<float>(),
        anchor_flat.data_ptr<float>(),
        alphas.data_ptr<float>(),
        g_cache.block_to_param.data_ptr<int>(),
        g_cache.block_start_idx.data_ptr<int>(),
        offsets.data_ptr<int64_t>(),
        sizes.data_ptr<int64_t>(),
        g_cache.num_blocks
    );
}


void fused_update_gamma_v5(
    torch::Tensor gamma,
    torch::Tensor ts,
    torch::Tensor zs,
    float grad,
    float step_size,
    float tau
) {
    const int num_params = gamma.size(0);
    const int threads = 256;
    const int blocks = (num_params + threads - 1) / threads;
    
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


void init_block_mapping_v5(torch::Tensor sizes) {
    build_block_mapping_v5(sizes, sizes.device());
}


int get_num_blocks() {
    return g_cache.num_blocks;
}


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compute_norms", &fused_compute_norms_v5, "Fused norm computation V5");
    m.def("fused_compute_norms_atomic", &fused_compute_norms_atomic, "Fused norm computation with atomics");
    m.def("fused_apply_constraints", &fused_apply_constraints_v5, "Fused constraint application V5");
    m.def("fused_reverse_constraints", &fused_reverse_constraints_v5, "Fused constraint reversal V5");
    m.def("fused_update_gamma", &fused_update_gamma_v5, "Fused gamma update V5");
    m.def("init_block_mapping", &init_block_mapping_v5, "Initialize block mapping cache");
    m.def("get_num_blocks", &get_num_blocks, "Get total number of blocks");
}

#endif  // USE_TORCH
