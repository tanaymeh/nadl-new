import sys
sys.path.append(".")
from nadl.core.tensor import Tensor

a = Tensor(data=[[10, 9, 8], [7, 6, 5]])
b = Tensor(data=[[1, 2, 3], [5, 6, 7]])

print(a)
print(b)

c = a + b

print(a.data)
print(b.data)
print(c)
print(c.data)