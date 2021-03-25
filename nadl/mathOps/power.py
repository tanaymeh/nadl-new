import numpy as np
from numba import cuda, jit
from typing import List, Union

from ..core.cpu_ops import Ops
from ..core.tensor import Tensor

class CPUOps:
    # Operation implemented to run by default on CPU
    def exec(power: Union[int, float]):
        output: Tensor = Ops.power(power, Tensor)
        return output

class GPUOps:
    # Operation implemented to run faster on GPU (using Numba jit)
    pass