from mygrad.tensor import Tensor

a = Tensor([1.5, 2])
b = Tensor([2.3, 1])

# a = Tensor(2)
# b = Tensor(3)
print((a + b).numpy())
