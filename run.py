from teenygrad.tensor import Tensor

# a = Tensor([1.5, 2])
# b = Tensor([2.3, 1])

a = Tensor([2,4])
b = Tensor([2,5])
print((a @ b).numpy())
