import numpy as np
from typing import Callable, NamedTuple

class Dependency(NamedTuple):
    """
    All Tensors (realistically) have a dependency.
    They will be passed around when doing operations.

    Members:
        tensor: Tensor Instance
        _backward: Gradient Calculation function for necessary operation
    """
    tensor: 'Tensor'
    backward_fn: Callable[[np.ndarray], np.ndarray]