import numpy as np

def dataTypeCheck(data, types):
    """Checks if the given data is of the given type.

    Args:
        data (any): Given data
        types (Tensor | list | float): Given data type we need to check for

    Raises:
        ValueError: If data is not of the given data type.

    Returns:
        np.ndarray : Returns data in numpy array-ed 
    """
    if not isinstance(data, types):
        raise ValueError(f"Wrong Data Type Provided: {type(data)}")
    else:
        if isinstance(data, types[1:]):
            return np.array(data)
        else:
            return data
        
def gradDepCheck(tensor):
    """
    Checks if the given tensor requires gradients
    """
    if tensor.requires_grad:
        return True
    return False

def numeralGradientDependencyCheck(*args):
    """
    Just like gradDepCheck() except it works on any number of Tensors as argument
    
    Returns a list of True or False corresponding to that gradient's status
    """
    gradientStatus = []
    for tensor in args:
        gradientStatus.append(gradDepCheck(tensor))
    return gradientStatus