"""
Parameter Flattening Utilities for DiZO Optimization

Provides functions to flatten/unflatten model parameters for batch operations.
"""

import torch
from typing import List, Tuple, Dict, Optional


class ParameterFlattener:
    """
    Manages flattening and unflattening of model parameters.
    """
    
    def __init__(self, model, exclude_list: List[str] = None):
        """
        Initialize flattener for a model.
        
        Args:
            model: PyTorch model
            exclude_list: List of parameter names to exclude
        """
        self.exclude_list = exclude_list or []
        self.param_metadata = []
        self.param_names = []
        
        # Build metadata
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
    
    def flatten(self, model) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Flatten all trainable parameters into a single tensor.
        
        Args:
            model: PyTorch model
        
        Returns:
            param_flat: Flattened parameters [total_size]
            offsets: Starting offsets for each param [num_params]
            sizes: Sizes of each param [num_params]
        """
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        
        param_flat = torch.empty(self.total_size, device=device, dtype=dtype)
        offsets = torch.zeros(self.num_params, device=device, dtype=torch.long)
        sizes = torch.zeros(self.num_params, device=device, dtype=torch.long)
        
        for i, meta in enumerate(self.param_metadata):
            name = meta['name']
            param = dict(model.named_parameters())[name]
            
            offsets[i] = meta['offset']
            sizes[i] = meta['size']
            
            # Copy parameter data to flat tensor
            param_flat[meta['offset']:meta['offset'] + meta['size']] = param.data.flatten()
        
        return param_flat, offsets, sizes
    
    def unflatten(self, model, param_flat: torch.Tensor) -> None:
        """
        Unflatten parameters back into model.
        
        Args:
            model: PyTorch model (modified in-place)
            param_flat: Flattened parameters [total_size]
        """
        param_dict = dict(model.named_parameters())
        
        for meta in self.param_metadata:
            name = meta['name']
            param = param_dict[name]
            
            # Extract and reshape
            flat_slice = param_flat[meta['offset']:meta['offset'] + meta['size']]
            param.data.copy_(flat_slice.view(meta['shape']))
    
    def get_anchor_flat(self, anchor_model) -> torch.Tensor:
        """
        Get flattened anchor parameters.
        
        Args:
            anchor_model: Anchor model
        
        Returns:
            anchor_flat: Flattened anchors [total_size]
        """
        device = next(anchor_model.parameters()).device
        dtype = next(anchor_model.parameters()).dtype
        
        anchor_flat = torch.empty(self.total_size, device=device, dtype=dtype)
        anchor_dict = dict(anchor_model.named_parameters())
        
        for meta in self.param_metadata:
            name = meta['name']
            if name in anchor_dict:
                anchor = anchor_dict[name]
                anchor_flat[meta['offset']:meta['offset'] + meta['size']] = anchor.data.flatten()
            else:
                # If anchor doesn't have this param, use zeros
                anchor_flat[meta['offset']:meta['offset'] + meta['size']] = 0.0
        
        return anchor_flat

