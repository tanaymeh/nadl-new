import numpy as np

def dataTypeCheck(data, types):
    if not isinstance(data, types):
        raise ValueError(f"Wrong Data Type Provided: {type(data)}")
    else:
        if isinstance(data, types[1:]):
            return np.array(data)
        else:
            return data