import sys
sys.path.append(".")
from nadl.core.tensor import Tensor

a = Tensor(data=[[10, 9, 8], [7, 6, 5]], requires_grad=True)
b = Tensor(data=[[1, 2, 3], [5, 6, 7]], requires_grad=True)

print(f"Tensor a: {a}")
print(f"Tensor b: {b}")

c = a + b
c.backward()

print(a,b,c)

print(a.grad)
print(b.grad)