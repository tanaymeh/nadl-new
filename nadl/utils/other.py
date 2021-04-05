import numpy as np
from typing import List

class validateMatrixOp:
    """
    Bunch of functions to validation matrix operations
    """
    
    def matmulSizeCheck(tensor1: 'Tensor', tensor2: 'Tensor'):
        """Checks if given 2 tensors can be multiplied or not.

        Args:
            tensor1 (Tensor): First Tensor
            tensor2 (Tensor): Second Tensor
        """
        if tensor1.shape[1] != tensor2.shape[0]:
            return False
        else:
            return True 
    
    def variableMatmulSizeCheck(*args):
        """
        Checks if the given sequence of tensors are multiplications compatible
        
        Given list of Tensors will be checked in accordance. Pass them in right order!
        """
        # Loop through all tensors by making a window of 2 tensors (current, next)
        # Check if the tensors in current window can be multiplied or not.
        tensors = args
        for _idx_current in range(len(tensors)):
            # If we reach last element then return True result
            if _idx_current == len(tensors) - 1:
                return (True, None)
            
            _idx_next = _idx_current + 1
            _current_tensor, _next_tensor = tensors[_idx_current], tensors[_idx_next]
            mml_check = validateMatrixOp.matmulSizeCheck(
                _current_tensor,
                _next_tensor
            )
            if not mml_check:
                return (False, _idx_current)