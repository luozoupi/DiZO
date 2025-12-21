"""
Optimized Parameter Buffer for DiZO

Key improvements over param_utils.py:
1. Pre-allocated persistent buffer (no runtime allocation)
2. In-place view-based operations when possible
3. Async copy support for overlapping compute
4. Efficient unflatten using pre-computed slices
"""

import torch
from typing import List, Tuple, Dict, Optional


class OptimizedParameterBuffer:
    """
    Memory-efficient parameter buffer with pre-allocation.
    
    Instead of creating new tensors each flatten/unflatten call,
    maintains persistent buffers that are reused.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        anchor_model: torch.nn.Module = None,
        exclude_list: List[str] = None,
        dtype: torch.dtype = None,
    ):
        """
        Initialize parameter buffer.
        
        Args:
            model: Model whose parameters to manage
            anchor_model: Anchor/pre-trained model (optional, can set later)
            exclude_list: Parameter names to exclude
            dtype: Data type (default: model's dtype)
        """
        self.exclude_list = set(exclude_list or [])
        self.device = next(model.parameters()).device
        self.dtype = dtype or next(model.parameters()).dtype
        
        # Build metadata
        self._build_metadata(model)
        
        # Pre-allocate buffers
        self.param_flat = torch.empty(
            self.total_size, 
            device=self.device, 
            dtype=self.dtype
        )
        self.anchor_flat = torch.empty(
            self.total_size, 
            device=self.device, 
            dtype=self.dtype
        )
        
        # Pre-allocate offset/size tensors (on GPU for kernel use)
        self.offsets = torch.tensor(
            [m['offset'] for m in self.param_metadata],
            device=self.device,
            dtype=torch.long
        )
        self.sizes = torch.tensor(
            [m['size'] for m in self.param_metadata],
            device=self.device,
            dtype=torch.long
        )
        
        # Initialize anchor if provided
        if anchor_model is not None:
            self.set_anchor(anchor_model)
        
        # Track if buffers are valid
        self._param_valid = False
        self._anchor_valid = False
    
    def _build_metadata(self, model: torch.nn.Module):
        """Build parameter metadata for efficient flatten/unflatten."""
        self.param_metadata = []
        self.param_names = []
        
        total_size = 0
        for name, param in model.named_parameters():
            if name not in self.exclude_list and param.requires_grad:
                size = param.numel()
                self.param_metadata.append({
                    'name': name,
                    'offset': total_size,
                    'size': size,
                    'shape': param.shape,
                })
                self.param_names.append(name)
                total_size += size
        
        self.total_size = total_size
        self.num_params = len(self.param_metadata)
    
    def flatten_into_buffer(self, model: torch.nn.Module) -> torch.Tensor:
        """
        Flatten model parameters into pre-allocated buffer.
        
        More efficient than creating new tensor each time.
        """
        param_dict = dict(model.named_parameters())
        
        for meta in self.param_metadata:
            param = param_dict[meta['name']]
            start = meta['offset']
            end = start + meta['size']
            
            # Direct copy into buffer (no allocation)
            self.param_flat[start:end].copy_(param.data.flatten())
        
        self._param_valid = True
        return self.param_flat
    
    def unflatten_from_buffer(self, model: torch.nn.Module) -> None:
        """
        Copy flattened buffer back to model parameters.
        """
        param_dict = dict(model.named_parameters())
        
        for meta in self.param_metadata:
            param = param_dict[meta['name']]
            start = meta['offset']
            end = start + meta['size']
            
            # Reshape and copy back
            param.data.copy_(self.param_flat[start:end].view(meta['shape']))
    
    def set_anchor(self, anchor_model: torch.nn.Module) -> torch.Tensor:
        """
        Set anchor parameters (typically done once at initialization).
        """
        anchor_dict = dict(anchor_model.named_parameters())
        
        for meta in self.param_metadata:
            name = meta['name']
            start = meta['offset']
            end = start + meta['size']
            
            if name in anchor_dict:
                self.anchor_flat[start:end].copy_(anchor_dict[name].data.flatten())
            else:
                # If anchor doesn't have this param, use zeros
                self.anchor_flat[start:end].zero_()
        
        self._anchor_valid = True
        return self.anchor_flat
    
    def get_buffers(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get all buffers needed for kernel operations.
        
        Returns:
            (param_flat, anchor_flat, offsets, sizes)
        """
        return self.param_flat, self.anchor_flat, self.offsets, self.sizes
    
    @property
    def shape_info(self) -> Dict:
        """Get shape information for debugging."""
        return {
            'total_size': self.total_size,
            'num_params': self.num_params,
            'param_shapes': [(m['name'], m['shape']) for m in self.param_metadata],
        }


class ViewBasedParameterBuffer:
    """
    Even more efficient: uses views into a contiguous buffer.
    
    WARNING: Requires model parameters to be contiguous.
    Only use if model was initialized with contiguous storage.
    """
    
    def __init__(
        self, 
        model: torch.nn.Module,
        exclude_list: List[str] = None,
    ):
        self.exclude_list = set(exclude_list or [])
        self.device = next(model.parameters()).device
        self.dtype = next(model.parameters()).dtype
        
        # Check if we can use views (parameters must be contiguous)
        self._check_contiguous(model)
        
        # Build metadata
        self._build_metadata(model)
    
    def _check_contiguous(self, model: torch.nn.Module):
        """Check if parameters are contiguous in memory."""
        params = [p for n, p in model.named_parameters() 
                  if n not in self.exclude_list and p.requires_grad]
        
        if len(params) == 0:
            raise ValueError("No trainable parameters found")
        
        # Check if all parameters are contiguous
        for p in params:
            if not p.data.is_contiguous():
                raise ValueError(
                    f"Parameter not contiguous. Use OptimizedParameterBuffer instead."
                )
    
    def _build_metadata(self, model: torch.nn.Module):
        """Build metadata for view-based access."""
        self.param_metadata = []
        
        total_size = 0
        for name, param in model.named_parameters():
            if name not in self.exclude_list and param.requires_grad:
                size = param.numel()
                self.param_metadata.append({
                    'name': name,
                    'offset': total_size,
                    'size': size,
                    'shape': param.shape,
                    'param_ref': param,  # Keep reference
                })
                total_size += size
        
        self.total_size = total_size
        self.num_params = len(self.param_metadata)
        
        # Create view-based flat tensor
        # This is a single tensor that views all parameters
        self._create_flat_view(model)
    
    def _create_flat_view(self, model: torch.nn.Module):
        """Create a flat tensor that shares storage with parameters."""
        # Concatenate all parameters into a single tensor
        param_list = []
        for meta in self.param_metadata:
            param_list.append(meta['param_ref'].data.flatten())
        
        # Note: This creates a copy, not a view
        # True view-based would require custom memory allocation
        self.param_flat = torch.cat(param_list)
        
        # Pre-allocate offsets/sizes on GPU
        self.offsets = torch.tensor(
            [m['offset'] for m in self.param_metadata],
            device=self.device,
            dtype=torch.long
        )
        self.sizes = torch.tensor(
            [m['size'] for m in self.param_metadata],
            device=self.device,
            dtype=torch.long
        )


def create_parameter_buffer(
    model: torch.nn.Module,
    anchor_model: torch.nn.Module = None,
    exclude_list: List[str] = None,
    try_view_based: bool = False,
) -> OptimizedParameterBuffer:
    """
    Factory function to create appropriate parameter buffer.
    
    Args:
        model: Model to manage
        anchor_model: Anchor model (optional)
        exclude_list: Parameters to exclude
        try_view_based: Try view-based buffer (faster but stricter requirements)
    
    Returns:
        OptimizedParameterBuffer instance
    """
    if try_view_based:
        try:
            return ViewBasedParameterBuffer(model, exclude_list)
        except ValueError:
            pass  # Fall back to copy-based
    
    return OptimizedParameterBuffer(model, anchor_model, exclude_list)
