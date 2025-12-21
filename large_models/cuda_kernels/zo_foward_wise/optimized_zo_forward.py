"""
Optimized zo_forward Implementation using Fused Kernels

This module provides an optimized version of DiZO.zo_forward that uses
fused CUDA/Triton kernels to minimize kernel launches and CPU-GPU synchronization.
"""

import torch
import sys
import os

# Add paths
sys.path.append(os.path.dirname(__file__))
from param_utils import ParameterFlattener
from dizo_fused_kernels import (
    fused_compute_norms,
    fused_apply_constraints,
    fused_reverse_constraints,
    fused_perturb_gamma,
    fused_update_gamma,
)


class OptimizedDiZO:
    """
    Optimized version of DiZO.zo_forward using fused kernels.
    """
    
    def __init__(self, dizo_instance, model, anchor_model):
        """
        Initialize optimized DiZO.
        
        Args:
            dizo_instance: Original DiZO instance
            model: Model to optimize
            anchor_model: Anchor/pre-trained model
        """
        self.dizo = dizo_instance
        self.model = model
        self.anchor_model = anchor_model
        
        # Create parameter flattener
        self.flattener = ParameterFlattener(model, exclude_list=dizo_instance.exclude_list)
        
        # Pre-allocate flattened buffers
        self.param_flat = None
        self.anchor_flat = None
        self.offsets = None
        self.sizes = None
        
        # Cache for alpha values (needed for reverse_constraints)
        self.alpha_cache = {}
    
    def _ensure_buffers(self):
        """Ensure flattened buffers are allocated."""
        if self.param_flat is None:
            self.param_flat, self.offsets, self.sizes = self.flattener.flatten(self.model)
            self.anchor_flat = self.flattener.get_anchor_flat(self.anchor_model)
    
    def zo_forward_optimized(self, new=None, pre_trained=None, x=None, apply=False, args=None):
        """
        Optimized version of zo_forward using fused kernels.
        
        This method replaces the original zo_forward with batched operations
        that minimize kernel launches and synchronization overhead.
        """
        tau = args.clip_range
        zo_eps = args.zo_eps_projection
        step_size = args.step_size_projection
        
        if apply:
            # Apply constraints (final step)
            self._ensure_buffers()
            self.param_flat, _, _ = self.flattener.flatten(new)
            
            # Compute norms
            norms = fused_compute_norms(
                self.param_flat,
                self.anchor_flat,
                self.offsets,
                self.sizes,
            )
            
            # Get constraints
            constraints = torch.stack([p.data for p in self.dizo.constraints])
            
            # Apply constraints
            fused_apply_constraints(
                self.param_flat,
                self.anchor_flat,
                self.offsets,
                self.sizes,
                constraints,
                norms,
                eps=1e-8,
            )
            
            # Unflatten back to model
            self.flattener.unflatten(new, self.param_flat)
            
        else:
            # Gradient estimation step
            self._ensure_buffers()
            
            with torch.no_grad():
                # Flatten parameters
                self.param_flat, self.offsets, self.sizes = self.flattener.flatten(new)
                
                # Compute norms for all parameters at once
                norms = fused_compute_norms(
                    self.param_flat,
                    self.anchor_flat,
                    self.offsets,
                    self.sizes,
                )
                
                # Convert to list for compatibility
                ts = norms.tolist()
                
                # Initialize constraints if needed
                if self.dizo.init:
                    constraints = torch.stack([p for p in self.dizo.constraints])
                    constraints.data = norms
                    for i, (name, gamma) in enumerate(self.dizo.constraints.named_parameters()):
                        gamma.data = norms[i]
                
                # Get constraints as tensor
                constraints = torch.stack([p.data for p in self.dizo.constraints])
                
                # Perturb gamma (+eps)
                seed = torch.randint(0, 2**31, (1,), device=constraints.device).item()
                zs = fused_perturb_gamma(
                    constraints,
                    norms,
                    seed,
                    delta=1.0,
                    tau=tau,
                    zo_eps=zo_eps,
                    zs=None,
                )
                
                # Apply constraints
                fused_apply_constraints(
                    self.param_flat,
                    self.anchor_flat,
                    self.offsets,
                    self.sizes,
                    constraints,
                    norms,
                    eps=1e-8,
                )
                
                # Unflatten and compute loss1
                self.flattener.unflatten(new, self.param_flat)
                loss1 = self.dizo.forward_wrap_with_option_len(new, **x, return_dict=True).loss
                
                # Reverse constraints
                # Compute alphas for reverse
                alphas = constraints / (norms + 1e-8)
                alphas_tensor = alphas
                
                # Re-flatten (parameters may have changed)
                self.param_flat, _, _ = self.flattener.flatten(new)
                
                fused_reverse_constraints(
                    self.param_flat,
                    self.anchor_flat,
                    self.offsets,
                    self.sizes,
                    alphas_tensor,
                )
                
                # Unflatten
                self.flattener.unflatten(new, self.param_flat)
                
                # Perturb gamma (-2eps)
                zs = fused_perturb_gamma(
                    constraints,
                    norms,
                    seed,
                    delta=-2.0,
                    tau=tau,
                    zo_eps=zo_eps,
                    zs=zs,  # Reuse zs
                )
                
                # Apply constraints again
                fused_apply_constraints(
                    self.param_flat,
                    self.anchor_flat,
                    self.offsets,
                    self.sizes,
                    constraints,
                    norms,
                    eps=1e-8,
                )
                
                # Unflatten and compute loss2
                self.flattener.unflatten(new, self.param_flat)
                loss2 = self.dizo.forward_wrap_with_option_len(new, **x, return_dict=True).loss
                
                # Reverse constraints again
                self.param_flat, _, _ = self.flattener.flatten(new)
                fused_reverse_constraints(
                    self.param_flat,
                    self.anchor_flat,
                    self.offsets,
                    self.sizes,
                    alphas_tensor,
                )
                self.flattener.unflatten(new, self.param_flat)
                
                # Perturb gamma back (+eps)
                zs = fused_perturb_gamma(
                    constraints,
                    norms,
                    seed,
                    delta=1.0,
                    tau=tau,
                    zo_eps=zo_eps,
                    zs=zs,  # Reuse zs
                )
                
                # Compute gradient
                grad = (loss1 - loss2) / (2 * zo_eps)
                
                # Update gamma with fused kernel
                fused_update_gamma(
                    constraints,
                    norms,
                    zs,
                    grad.item(),
                    step_size,
                    tau,
                )
                
                # Update constraints in DiZO instance
                for i, (name, gamma) in enumerate(self.dizo.constraints.named_parameters()):
                    gamma.data = constraints[i]
            
            self.dizo.ts = ts

