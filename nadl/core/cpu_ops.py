import numpy as np
import numba

class Ops:
    def add(*args):
        """
        This is a generic implementations of Addition Operations.
        This will be called both via overloaded op and via math module.

        Args:
            tensor1 ([Tensor]): First Tensor ('self' in overloaded op)
            tensor2 ([Tensor]): Second Tensor ('tensor' in overloaded op)
            TensorDataTypeWrapper ([class Tensor]): Used for wrapping the Output
        """
        return

    def sub(*args):
        pass

    def matmul(*args):
        pass

    def pow(*args):
        pass