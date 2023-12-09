from mygrad.tensor import Tensor

# a = Tensor([1.5, 2])
# b = Tensor([2.3, 1])

class Linear:
    def __init__(self, in_features, out_features, bias=False, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)



a = Tensor.rand((128, 784))


net = Linear(784, 128, bias=True)

print(net.weight)
print(net.bias)

print(net(a))
