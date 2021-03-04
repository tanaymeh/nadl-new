import numpy as np
import numba
from typing import List

from ..utils.checks import gradDepCheck
from ..core.dep import Dependency

class Ops:
    def add(*args):
        """
        This is a generic implementations of Addition Operation.
        This will be called both via overloaded op and via math module.

        Args:
            tensor1 ([Tensor]): First Tensor ('self' in overloaded op)
            tensor2 ([Tensor]): Second Tensor ('tensor' in overloaded op)
            TensorDataTypeWrapper ([class Tensor]): Used for wrapping the Output
        """
        # Extract inputs from passed arguments list
        tensor1: 'Tensor' = args[0]
        tensor2: 'Tensor' = args[1]
        TensorDataTypeWrapper: 'Tensor' = args[2]

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

    def sub(*args):
        pass

    def matmul(*args):
        pass

    def pow(*args):
        pass