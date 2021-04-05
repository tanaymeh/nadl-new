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
    
    def variableMatmulSizeCheck(*args: List['Tensor']):
        """
        Checks if the given sequence of tensors are multiplications compatible
        
        Given list of Tensors will be checked in accordance. Pass them in right order!
        """
        
        raise NotImplementedError("Function: <variableMatmulSizeCheck> is not yet implemented.")