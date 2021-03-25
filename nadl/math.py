"""
    This file will have upper-level abstraction of all the math functions
"""
import numpy as np
from typing import Union

from .core.tensor import Tensor
from .mathOps.add import CPUOps as Add_C_Ops
from .mathOps.subtract import CPUOps as Sub_C_Ops
from .mathOps.power import CPUOps as Pwr_C_Ops
from .mathOps.matmul import CPUOps as Mml_C_Ops

def add(x: Tensor, y: Tensor): 
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
     
    out = Add_C_Ops.exec(x, y)
    return out
    
def matmul(x: Tensor, y: Tensor): 
    """
    Performs Matrix Multiplication on 2 Tensors and returns the product Tensor
    
    If shape of the first Tensor is (m × n) then the shape of second Tensor 
    must be (n × k).
    
    Args:
        x (Tensor): First Tensor
        y (Tensor): Second Tensor
    """
    # TODO: Add GPU Dispatch and Id system to see what operations works best.
    # ? Dispatch may need the entire code tree to be restructured
     
    out = Mml_C_Ops.exec(x, y)
    return out

def power(power: Union[int, float]):
    """
    Raises the current Tensor to the power of "power"
    
    Only supporting Integer and Floating powers.
    
    Args:
        power (int | float): Exponent
    """
    # TODO: Add GPU Dispatch and Id system to see what operations works best.
    # ? Dispatch may need the entire code tree to be restructured
     
    out = Pwr_C_Ops.exec(power)
    return out

def subtract(x: Tensor, y: Tensor):
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
     
    out = Sub_C_Ops.exec(x, y)
    return out