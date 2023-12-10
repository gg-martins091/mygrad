from mygrad.tensor import Tensor

# a = Tensor([1.5, 2])
# b = Tensor([2.3, 1])

class Linear:
    def __init__(self, in_features, out_features, bias=False, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        print(self.weight)
        self.bias = Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)



class TinyNet:
    def __init__(self):
        self.l1 = Linear(784, 128, bias=False)
        self.l2 = Linear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x


net = TinyNet()

a = Tensor.rand((128, 784))

print(net(a).numpy())
