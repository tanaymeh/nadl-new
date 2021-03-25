import numpy as np
import numba
from typing import List

from ..utils.checks import gradDepCheck
from ..utils.other import validateMatrixOp
from ..core.dep import Dependency

class Ops:
    def add(*args):
        """
        This is a generic implementations of Addition Operation.
        This will be called both via overloaded op and via math module.

        Args:
            tensor1 (Tensor): First Tensor ('self' in overloaded op)
            tensor2 (Tensor): Second Tensor ('tensor' in overloaded op)
            TensorDataTypeWrapper (Tensor): Used for wrapping the Output
        """
        # Extract inputs from passed arguments list
        tensor1: 'Tensor' = args[0]
        tensor2: 'Tensor' = args[1]
        TensorDataTypeWrapper: 'Tensor' = args[2]

        # Get the output and do gradient dependency checks
        output = tensor1.data + tensor2.data
        t1_rg, t2_rg = gradDepCheck(tensor1), gradDepCheck(tensor2)
        requires_grad = t1_rg or t2_rg

        parent: List[Dependency] = []

        # Start Operations for Backward
        if t1_rg:
            def _t1_grad_fn(grad: np.ndarray):
                ndims_add = grad.ndims - tensor1.grad.ndim
                for _ in range(ndims_add):
                    grad = grad.sum(axis=0)

                for i, dim in enumerate(tensor1.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                
                return grad
            parent.append(Dependency(tensor1, _t1_grad_fn))
        
        if t2_rg:
            def _t2_grad_fn(grad: np.ndarray):
                ndims_add = grad.ndims - tensor2.grad.ndim
                for _ in range(ndims_add):
                    grad = grad.sum(axis=0)

                for i, dim in enumerate(tensor2.shape):
                    if dim == 1:
                        grad = grad.sum(axis=i, keepdims=True)
                
                return grad
            parent.append(Dependency(tensor1, _t2_grad_fn))
            
        return TensorDataTypeWrapper(
            data=output,
            requires_grad=requires_grad,
            parents=parent
        )
        
    def matmul(*args):
        """
        This is a generic implementation of Matrix Multiplication Operation.
        This will be called both via overloaded op and via math module.

        Args:
            tensor1 (Tensor): First Tensor ('self' in overloaded op)
            tensor2 (Tensor): Second Tensor ('tensor' in overloaded op)
        """
        # Extract inputs from passed arguments list
        tensor1: 'Tensor' = args[0]
        tensor2: 'Tensor' = args[1]
        TensorDataTypeWrapper: 'Tensor' = args[2]
        
        # Validate the sizes of the matrices for compatibility
        isCompatible = validateMatrixOp.matmulSizeCheck(tensor1, tensor2)
        if not isCompatible:
            raise ValueError(
                f"Matrices of dims: {tensor1.shape} and {tensor2.shape} are not compatible"
            )
            
        # Now get the output (may take some time.)
        # TODO: Implement a numba matmul operation here for faster speed
        # ? The Numpy matmul operation isn't supposed to throw an error
        output = np.matmul(tensor1.data, tensor2.data)
        
        # See if both do require gradients
        t1_rg, t2_rg = gradDepCheck(tensor1), gradDepCheck(tensor2)
        requires_grad = t1_rg or t2_rg
        
        parent: List[Dependency] = []
        
        # Start operation for matmul AD
        if t1_rg:
            def _t1_grad_fn(grad: np.ndarray):
                return np.matmul(grad, tensor2.data.T)
            parent.append(Dependency(tensor1, _t1_grad_fn))
            
        if t2_rg:
            def _t2_grad_fn(grad: np.ndarray):
                return np.matmul(tensor1.data.T, grad)
            parent.append(Dependency(tensor2, _t2_grad_fn))
        
        return TensorDataTypeWrapper(
            data=output,
            requires_grad=requires_grad,
            parents=parent
        )       
        
    def sub(*args):
        """
        This is a generic implementations of Subtraction Operation.
        This will be called both via overloaded op and via math module.

        Args:
            tensor1 (Tensor): First Tensor ('self' in overloaded op)
            tensor2 (Tensor): Second Tensor ('tensor' in overloaded op)
        """
        # Extract inputs from passed arguments list
        tensor1: 'Tensor' = args[0]
        tensor2: 'Tensor' = args[1]

        # Subtraction is basically addition but in negatives.
        output = tensor1 + -tensor2
        return output

    def negative(*args):
        """
        This is a generic implementations of Negative Operation.
        This will be called via overloaded op only (in most cases).

        Args:
            tensor (Tensor): First Tensor ('self' in overloaded op)
            TensorDataTypeWrapper (Tensor): Used for wrapping the Output
        """
        # Extract inputs from args
        tensor: 'Tensor' = args[0]
        TensorDataTypeWrapper: 'Tensor' = args[1]

        # Operations
        data =  - tensor.data
        requires_grad = tensor.requires_grad

        if requires_grad:
            parent = [Dependency(tensor, lambda x: -x)]
        else:
            parent = []
        
        return TensorDataTypeWrapper(
            data=data,
            requires_grad=requires_grad,
            parents=parent
        )

    def power(*args):
        raise NotImplemented("Power function is not yet implemented.")