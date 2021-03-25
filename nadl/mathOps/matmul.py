import numpy as np
from numba import cuda, jit
from typing import List

from ..core.cpu_ops import Ops
from ..core.tensor import Tensor

class CPUOps:
    # Operation implemented to run by default on CPU
    def exec(tensor1: Tensor, tensor2: Tensor):
        output: Tensor = Ops.matmul(tensor1, tensor2, Tensor)
        return output

class GPUOps:
    # Operation implemented to run faster on GPU (using Numba jit)
    pass