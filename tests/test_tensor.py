import sys
sys.path.append(".")
from nadl.core.tensor import Tensor

a = Tensor(data=[1, 2, 3])

print(a)
print(a.grad)
a.zero_grad()
print(a.grad)