"""
    This file will have upper-level abstraction of all the math functions
"""
import numpy as np

from .core.tensor import Tensor
from .mathOps.add import CPUOps as Add_C_Ops
from .mathOps.subtract import CPUOps as Sub_C_Ops
from .mathOps.power import CPUOps as Pwr_C_Ops
from .mathOps.matmul import CPUOps as Mml_C_Ops

def add(x, y): 
    """
    Adds 2 Tensors and returns the resulting Tensor
    
    If shape of the first Tensor is (m × n) then the shape of second Tensor 
    must also be (m × n)
    
    Args:
        x (Tensor): First Tensor
        y (Tensor): Second Tensor
    """
    # TODO: Add GPU Dispatch and Id system to see what operations works best.
    # ? Dispatch may need the entire code tree to be restructured
     
    out = Add_C_Ops.add()
    return out
    
def matmul(): raise NotImplementedError()
def power(): raise NotImplementedError()
def subtract(): raise NotImplementedError()