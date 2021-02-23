import numpy as np
from typing import Union

def dataTypeCheck(data, union):
    if not isinstance(data, union):
        raise ValueError(f"Given data is of wrong datatype. Given datatype: {type(data)}")
    else:
        if isinstance(data, union):
            return data
        else:
            return np.array(data)