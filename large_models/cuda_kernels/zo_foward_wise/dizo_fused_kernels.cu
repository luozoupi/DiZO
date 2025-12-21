/*
 * Fused CUDA Kernels for DiZO zo_forward Optimization
 * 
 * This file provides optimized CUDA kernels for:
 * 1. Batch L2 norm computation across parameter groups
 * 2. Fused constraint application (compute alpha + apply projection)
 * 3. Fused constraint reversal
 * 4. Fused gamma update with clipping
 * 
 * Key optimizations:
 * 1. Coalesced memory access patterns
 * 2. Efficient reduction operations
 * 3. Minimal kernel launches (batch operations)
 * 4. No intermediate storage needed
 * 
 * Compile with:
 *   python setup.py install
 */

#include <cuda.h>
#include <cuda_runtime.h>
#include <cmath>

// For PyTorch C++ extension
#ifdef USE_TORCH
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#endif

// =============================================================================
// Kernel 1: Fused Norm Computation
// =============================================================================

template<typename scalar_t>
__global__ void fused_norm_kernel(
    const scalar_t* __restrict__ param_flat,
    const scalar_t* __restrict__ anchor_flat,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    scalar_t* __restrict__ norms_out,
    int num_params
) {
    int pid = blockIdx.x;
    
    if (pid >= num_params) {
        return;
    }
    
    // Load offset and size for this parameter group
    int64_t offset = offsets[pid];
    int64_t size = sizes[pid];
    
    // Compute L2 norm using block-wise reduction
    scalar_t acc = scalar_t(0.0);
    
    // Each thread processes multiple elements
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    for (int64_t i = tid; i < size; i += block_size) {
        int64_t idx = offset + i;
        scalar_t diff = param_flat[idx] - anchor_flat[idx];
        acc += diff * diff;
    }
    
    // Block-wise reduction using shared memory
    __shared__ scalar_t sdata[256];
    sdata[tid] = acc;
    __syncthreads();
    
    // Reduction in shared memory
    for (int s = block_size / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    // Write result (first thread writes)
    if (tid == 0) {
        scalar_t norm = sqrt(sdata[0] + scalar_t(1e-8));  // Add epsilon for numerical stability
        norms_out[pid] = norm;
    }
}


// =============================================================================
// Kernel 2: Fused Constraint Application
// =============================================================================

template<typename scalar_t>
__global__ void fused_apply_constraints_kernel(
    scalar_t* __restrict__ param_flat,
    const scalar_t* __restrict__ anchor_flat,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const scalar_t* __restrict__ constraints,  // gamma values
    const scalar_t* __restrict__ norms,  // pre-computed norms
    int num_params,
    scalar_t eps
) {
    int pid = blockIdx.x;
    
    if (pid >= num_params) {
        return;
    }
    
    // Load constraint and norm for this parameter group
    scalar_t constraint = constraints[pid];
    scalar_t norm = norms[pid];
    
    // Compute alpha
    scalar_t alpha = constraint / (norm + eps);
    
    // Load offset and size
    int64_t offset = offsets[pid];
    int64_t size = sizes[pid];
    
    // Apply constraint: param = anchor + (param - anchor) * alpha
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    for (int64_t i = tid; i < size; i += block_size) {
        int64_t idx = offset + i;
        scalar_t param_val = param_flat[idx];
        scalar_t anchor_val = anchor_flat[idx];
        
        // Project: param = anchor + (param - anchor) * alpha
        scalar_t diff = param_val - anchor_val;
        param_flat[idx] = anchor_val + diff * alpha;
    }
}


// =============================================================================
// Kernel 3: Fused Constraint Reversal
// =============================================================================

template<typename scalar_t>
__global__ void fused_reverse_constraints_kernel(
    scalar_t* __restrict__ param_flat,
    const scalar_t* __restrict__ anchor_flat,
    const int64_t* __restrict__ offsets,
    const int64_t* __restrict__ sizes,
    const scalar_t* __restrict__ alphas,  // alpha values from apply_constraints
    int num_params
) {
    int pid = blockIdx.x;
    
    if (pid >= num_params) {
        return;
    }
    
    // Load alpha
    scalar_t alpha = alphas[pid];
    
    // Load offset and size
    int64_t offset = offsets[pid];
    int64_t size = sizes[pid];
    
    // Reverse constraint: param = anchor + (param - anchor) / alpha
    const int tid = threadIdx.x;
    const int block_size = blockDim.x;
    
    for (int64_t i = tid; i < size; i += block_size) {
        int64_t idx = offset + i;
        scalar_t param_val = param_flat[idx];
        scalar_t anchor_val = anchor_flat[idx];
        
        // Reverse: param = anchor + (param - anchor) / alpha
        scalar_t diff = param_val - anchor_val;
        param_flat[idx] = anchor_val + diff / alpha;
    }
}


// =============================================================================
// Kernel 4: Fused Gamma Update
// =============================================================================

template<typename scalar_t>
__global__ void fused_update_gamma_kernel(
    scalar_t* __restrict__ gamma,
    const scalar_t* __restrict__ ts,
    const scalar_t* __restrict__ zs,
    scalar_t grad,
    scalar_t step_size,
    scalar_t tau,
    int num_params
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_params) {
        return;
    }
    
    scalar_t gamma_val = gamma[idx];
    scalar_t t = ts[idx];
    scalar_t z = zs[idx];
    
    // Compute update: gamma = gamma - step_size * ts * grad * z
    scalar_t gamma_new = gamma_val - step_size * t * grad * z;
    
    // Clip: gamma = clip(gamma_new, (1-tau)*ts, (1+tau)*ts)
    scalar_t gamma_min = (scalar_t(1.0) - tau) * t;
    scalar_t gamma_max = (scalar_t(1.0) + tau) * t;
    
    // Use template-friendly min/max
    gamma[idx] = (gamma_new < gamma_min) ? gamma_min : ((gamma_new > gamma_max) ? gamma_max : gamma_new);
}


// =============================================================================
// PyTorch C++ Extension Interface
// =============================================================================

#ifdef USE_TORCH

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Wrapper for fused norm computation
torch::Tensor fused_compute_norms_cuda(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes
) {
    int num_params = offsets.size(0);
    auto device = param_flat.device();
    auto dtype = param_flat.dtype();
    
    auto norms = torch::empty({num_params}, torch::TensorOptions().dtype(dtype).device(device));
    
    // Launch configuration
    dim3 grid(num_params);
    dim3 block(256);  // Optimal block size for reduction
    
    AT_DISPATCH_FLOATING_TYPES(param_flat.scalar_type(), "fused_norm_kernel", [&] {
        fused_norm_kernel<scalar_t><<<grid, block>>>(
            param_flat.data_ptr<scalar_t>(),
            anchor_flat.data_ptr<scalar_t>(),
            offsets.data_ptr<int64_t>(),
            sizes.data_ptr<int64_t>(),
            norms.data_ptr<scalar_t>(),
            num_params
        );
    });
    
    return norms;
}


// Wrapper for fused constraint application
void fused_apply_constraints_cuda(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor constraints,
    torch::Tensor norms,
    float eps
) {
    int num_params = offsets.size(0);
    
    // Launch configuration
    dim3 grid(num_params);
    dim3 block(256);
    
    AT_DISPATCH_FLOATING_TYPES(param_flat.scalar_type(), "fused_apply_constraints_kernel", [&] {
        fused_apply_constraints_kernel<scalar_t><<<grid, block>>>(
            param_flat.data_ptr<scalar_t>(),
            anchor_flat.data_ptr<scalar_t>(),
            offsets.data_ptr<int64_t>(),
            sizes.data_ptr<int64_t>(),
            constraints.data_ptr<scalar_t>(),
            norms.data_ptr<scalar_t>(),
            num_params,
            static_cast<scalar_t>(eps)
        );
    });
}


// Wrapper for fused constraint reversal
void fused_reverse_constraints_cuda(
    torch::Tensor param_flat,
    torch::Tensor anchor_flat,
    torch::Tensor offsets,
    torch::Tensor sizes,
    torch::Tensor alphas
) {
    int num_params = offsets.size(0);
    
    // Launch configuration
    dim3 grid(num_params);
    dim3 block(256);
    
    AT_DISPATCH_FLOATING_TYPES(param_flat.scalar_type(), "fused_reverse_constraints_kernel", [&] {
        fused_reverse_constraints_kernel<scalar_t><<<grid, block>>>(
            param_flat.data_ptr<scalar_t>(),
            anchor_flat.data_ptr<scalar_t>(),
            offsets.data_ptr<int64_t>(),
            sizes.data_ptr<int64_t>(),
            alphas.data_ptr<scalar_t>(),
            num_params
        );
    });
}


// Wrapper for fused gamma update
void fused_update_gamma_cuda(
    torch::Tensor gamma,
    torch::Tensor ts,
    torch::Tensor zs,
    float grad,
    float step_size,
    float tau
) {
    int num_params = gamma.size(0);
    
    // Launch configuration
    int threads = 256;
    int blocks = (num_params + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(gamma.scalar_type(), "fused_update_gamma_kernel", [&] {
        fused_update_gamma_kernel<scalar_t><<<blocks, threads>>>(
            gamma.data_ptr<scalar_t>(),
            ts.data_ptr<scalar_t>(),
            zs.data_ptr<scalar_t>(),
            static_cast<scalar_t>(grad),
            static_cast<scalar_t>(step_size),
            static_cast<scalar_t>(tau),
            num_params
        );
    });
}


// Python bindings
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_compute_norms", &fused_compute_norms_cuda, "Fused norm computation (CUDA)");
    m.def("fused_apply_constraints", &fused_apply_constraints_cuda, "Fused constraint application (CUDA)");
    m.def("fused_reverse_constraints", &fused_reverse_constraints_cuda, "Fused constraint reversal (CUDA)");
    m.def("fused_update_gamma", &fused_update_gamma_cuda, "Fused gamma update (CUDA)");
}

#endif  // USE_TORCH
