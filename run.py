from mygrad.tensor import Tensor

# a = Tensor([1.5, 2])
# b = Tensor([2.3, 1])

class Linear:
    def __init__(self, in_features, out_features, bias=True, initialization: str='kaiming_uniform'):
        self.weight = getattr(Tensor, initialization)(out_features, in_features)
        # self.bias = Tensor.zeros(out_features) if bias else None



a = Tensor([[2,3],[4,5]])
b = Tensor([[2,6],[8,9]])
# print((a.linear(b)).numpy())

# Net = Linear(784, 128, bias=False)

print(Tensor.kaiming_uniform(784, 128))
