import numpy as np
import warnings
from typing import Union, NamedTuple, Callable, List, Optional

from ..utils.checks import dataTypeCheck, gradDepCheck
from ..core.dep import Dependency
from ..core.cpu_ops import Ops as c_ops
# from gpu_ops import Ops as g_ops

###############  Define Type Checks  ################
AllowedDataType = Union[np.ndarray, list, float] 
CanBeTensorfied = Union['Tensor', np.ndarray, float]
AllowedDataTypeTuple = (np.ndarray, list, float)
CanBeTensorfiedTuple = ('Tensor', np.ndarray, float)

class Tensor:
    def __init__(
        self, 
        data: AllowedDataType,
        requires_grad: bool = True,
        parents: List[Dependency] = None
        ) -> None:
        """Constructor Method that will initialize the starting properties of a Tensor

        Args:
            data (AllowedDataType): [Data for the Tensor]
            requires_grad (bool, optional): [Should the Tensor's gradients be calculated],
                                            Defaults to True.
            parents (List[Dependency], optional): [List of all Dependencies],
                                            Defaults to None.
        """

        # Sanity Check data type
        self.__data = dataTypeCheck(data, AllowedDataTypeTuple)

        # Initialize other properties
        self.requires_grad = requires_grad
        self.parents = parents or []
        self.shape = self.__data.shape
        self.grad: Optional['Tensor'] = None

        # Initialize gradients as Tensor itself
        # This way we can calculate n-order derivatives
        if self.requires_grad:
            self.zero_grad()
    
    def experimental_set_data(self, new_data: np.ndarray) -> None:
        """
        Explicitly change the data of a Tensor.
        THIS WILL ERASE ALL GRADIENTS
        Not Recommended.    

        Args:
            new_data (np.ndarray): [New Data to be set]
        """
        warnings.warn(
            "Changing Data Explicitly is not Recommended, this will erase all the gradients!",
            RuntimeWarning
        )

        self.__data = new_data
        self.grad = None

    def __repr__(self):
        return f"Tensor<shape={self.__data.shape}, requires_grad={self.requires_grad}>"

    def zero_grad(self) -> None:
        """
            Set the gradient back to a Zero Array
        """
        self.grad = Tensor(np.zeros_like(self.data, dtype=np.float64))
    
    @property
    def data(self) -> np.ndarray:
        return self.__data
    
    def __add__(self, tensor):
        """
        Addition operation that add external tensor to out tensor

        Args:
            tensor (Tensor): Must be of a Type Tensor
        """
        output = c_ops.add(self, tensor, Tensor)
        return output
    
    def __neg__(self):
        """
        Negative operation that does what it says it does.
        """
        output = c_ops.negative(self, Tensor)
        return output

    def __sub__(self, tensor):
        """
        Subtraction operation that Subtracts 'tensor' from 'self' tensor

        Args:
            tensor (Tensor): Another Tensor
        """
        output = c_ops.sub(self, tensor, Tensor)
        return output
    
    def __pow__(self, power: Union[int, float]):
        """
        Power operation that raises the current tensor to the power of "power"
        
        Args:
            power (int | float): Power to be raised to. Can only be integer and float
        """
        output = c_ops.pow(self, power, Tensor)
        return output

    def __div__(self, tensor):
        """
        Divides a Tensor by another Tensor
        
        Args: tensor (Tensor): Tensor that comes as the denominator
        """
        raise NotImplementedError("Division Operation is not yet implemented.")
        output = c_ops.div(self, tensor, Tensor)
        return output
    
    def backward(self, grad: 'Tensor'=None) -> None:
        """
        Performs gradient calculation using automatic differentiation
        """
        # First check if gradient calculation is allowed or not
        if not self.requires_grad:
            raise ValueError("Cannot calculate gradient on a constant tensor.")
        
        if grad is None:
            if self.shape == ():
                grad = Tensor(1.0)
            else:
                raise RuntimeError("Gradient must be satisfied for a Non-Zero tensor")
        
        self.grad.data = self.grad.data + grad.data
        
        for parent in self.parents:
            backward_grad = parent.backward_fn(grad.data)
            parent.tensor.backward(Tensor(backward_grad))