from mygrad.tensor import Tensor



a = Tensor([3,4], requires_grad=True)
b = Tensor([5,6], requires_grad=True)


r = a*b
print(r.numpy())



square = r.square()
print(f"{square=}")
mean = square.mean()
print(f"{mean=}")
print(f"{square.numpy()=}")
print(f"{mean.numpy()=}")
print(mean.backward().numpy())

