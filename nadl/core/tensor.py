import numpy as np
import warnings
from typing import Union, NamedTuple, Callable, List

from utils.typechecks import dataTypeCheck

###############  Define Type Checks  ############### 
AllowedDataType = Union[np.ndarray, list, float]
CanBeTensorfied = Union['Tensor', np.ndarray, float]

class Dependency(NamedTuple):
    """
    All Tensors (realistically) have a dependency.
    They will be passed around when doing operations.

    Members:
        tensor: Tensor Instance
        _backward: Gradient Calculation function for necessary operation
    """
    tensor: 'Tensor'
    _backward: Callable[[np.ndarray], np.ndarray]

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
        self.__data = dataTypeCheck(data, AllowedDataType)

        # Initialize other properties
        self.requires_grad = requires_grad
        self.parents = parents or []
        self.shape = self.__data.shape
        self.grad = None

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
        return f"Tensor<shape:{self.__data.shape}, requires_grad:{self.requires_grad}>"

    @property
    def data(self) -> np.ndarray:
        return self.__data

    def zero_grad(self) -> None:
        """
            Set the gradient back to a Zero Tensor
        """
        self.grad = Tensor(data=np.zeros_like(self.data, dtype=np.float64))
        