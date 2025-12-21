"""
Example integration of optimized zo_forward into DiZO trainer.

This shows how to modify trainer.py to use the optimized kernels.
"""

# Example modification to trainer.py
# 
# In dizo_trainer class, modify dizo_zo_iters method:
#
# OLD CODE:
# def dizo_zo_iters(self, model, base_model, apply=False, args=None):
#     if not apply:
#         # ... existing code ...
#         self.dizo.zo_forward(model, base_model, x=data, args=args)
#     self.dizo.zo_forward(model, self.pre_trained, apply=True, args=args)
#
# NEW CODE:
# def dizo_zo_iters(self, model, base_model, apply=False, args=None):
#     # Import optimized version
#     from cuda_kernels.zo_foward_wise import OptimizedDiZO
#     
#     # Initialize optimized DiZO (can be cached)
#     if not hasattr(self, '_optimized_dizo'):
#         self._optimized_dizo = OptimizedDiZO(
#             self.dizo,
#             model,
#             base_model
#         )
#     
#     if not apply:
#         # ... existing data loading code ...
#         # Use optimized version
#         self._optimized_dizo.zo_forward_optimized(
#             new=model,
#             pre_trained=base_model,
#             x=data,
#             apply=False,
#             args=args
#         )
#     else:
#         # Final apply step
#         self._optimized_dizo.zo_forward_optimized(
#             new=model,
#             pre_trained=self.pre_trained,
#             x=None,
#             apply=True,
#             args=args
#         )


def patch_trainer_zo_forward(trainer_instance):
    """
    Patch a trainer instance to use optimized zo_forward.
    
    Usage:
        trainer = dizo_trainer(...)
        patch_trainer_zo_forward(trainer)
        # Now trainer.dizo_zo_iters will use optimized kernels
    """
    from cuda_kernels.zo_foward_wise import OptimizedDiZO
    
    original_dizo_zo_iters = trainer_instance.dizo_zo_iters
    
    def optimized_dizo_zo_iters(model, base_model, apply=False, args=None):
        # Initialize optimized DiZO if needed
        if not hasattr(trainer_instance, '_optimized_dizo'):
            trainer_instance._optimized_dizo = OptimizedDiZO(
                trainer_instance.dizo,
                model,
                base_model
            )
        
        if not apply:
            trainer_instance.count = 0
            trainer_instance.dizo = trainer_instance.dizo.to(trainer_instance.device)
            trainer_instance.dizo.init = True
            
            while trainer_instance.count < trainer_instance.max_iters:
                try:
                    data = next(trainer_instance.dataset_iterator)
                except StopIteration:
                    trainer_instance.dataset_iterator = iter(trainer_instance.pgmloader)
                    data = next(trainer_instance.dataset_iterator)
                
                for each in data:
                    data[each] = data[each].to(trainer_instance.device)
                
                # Use optimized zo_forward
                trainer_instance._optimized_dizo.zo_forward_optimized(
                    new=model,
                    pre_trained=base_model,
                    x=data,
                    apply=False,
                    args=args
                )
                trainer_instance.dizo.init = False
                trainer_instance.count += 1
        
        # Final apply step
        trainer_instance._optimized_dizo.zo_forward_optimized(
            new=model,
            pre_trained=trainer_instance.pre_trained,
            x=None,
            apply=True,
            args=args
        )
        trainer_instance.i += 1
    
    # Replace method
    trainer_instance.dizo_zo_iters = optimized_dizo_zo_iters
    
    return trainer_instance


if __name__ == "__main__":
    print("This is an example integration file.")
    print("See comments for usage instructions.")

