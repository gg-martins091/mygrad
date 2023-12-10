
import teenygrad.tensor as tg
import unittest
from mygrad.tensor import Tensor
from mygrad.helpers import DEBUG
import time
import numpy as np


weights = {}

def get_weights(x, y):
    key = (x,y)
    try:
        w = weights[key]
        return w
    except:
        ret = Tensor.kaiming_uniform(x, y).numpy()
        weights[key] = ret
        return ret


class TgLinear:
    def __init__(self, in_features, out_features, bias=False, initialization: str='kaiming_uniform'):
        self.weight = tg.Tensor(get_weights(out_features, in_features))
        self.bias = tg.Tensor.zeros(out_features) if bias else None

    def __call__(self, x):
        return x.linear(self.weight.transpose(), self.bias)

class TgTinyNet:
    def __init__(self):
        self.l1 = TgLinear(784, 128, bias=False)
        self.l2 = TgLinear(128, 10, bias=False)

    def __call__(self, x):
        x = self.l1(x)
        x = x.leakyrelu()
        x = self.l2(x)
        return x


class Linear:
    def __init__(self, in_features, out_features, bias=False, initialization: str='kaiming_uniform'):
        self.weight = Tensor(get_weights(out_features, in_features))
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



def helper_test_op(data, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, forward_only=True):
    #inputs
    teeny, my = prepare_test_op(data)
    if DEBUG >= 1:
        print(f"{teeny=}")
        print(f"{my=}")

    teenygrad_fxn = TgTinyNet()
    mygrad_fxn = TinyNet()

    st = time.monotonic()
    out = teenygrad_fxn(teeny)
    teeny_fp = time.monotonic() - st

    st = time.monotonic()
    ret = mygrad_fxn(my)
    my_fp = time.monotonic() - st


    def compare(s, x, y, atol, rtol):
        if DEBUG >= 1: print(s, x, y)
        assert x.shape == y.shape, f"shape mismatch: mygrad={x.shape} | teeny={y.shape}"
        try:
            np.testing.assert_allclose(x, y, atol=atol, rtol=rtol)
        except Exception:
            raise Exception(f"{s} failed shape {x.shape}")

    compare("forward pass", ret.numpy(), out.numpy(), atol=atol, rtol=rtol)

    # TODO: can't go backwards yet
    torch_fbp, tinygrad_fbp = np.nan, np.nan
    # if not forward_only:
    #     st = time.monotonic()
    #     (out+1).square().mean().backward()
    #     torch_fbp = time.monotonic() - st

    #     st = time.monotonic()
    #     (ret+1).square().mean().backward()
    #     for tt in tst: tt.grad.realize()
    #         tinygrad_fbp = time.monotonic() - st

    #     for i, (t, tt) in enumerate(zip(ts, tst)):
    #       compare(f"backward pass tensor {i}", tt.grad.numpy(), t.grad.detach().numpy(), atol=grad_atol, rtol=grad_rtol)

    print("\ntesting   teenygrad/mygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % (teeny_fp*1000, my_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(data):
    teeny = tg.Tensor(data)
    my = Tensor(data)


    return teeny, my

class TestNn(unittest.TestCase):
    def teste_linear(self):
        data = Tensor.rand((128,784)).numpy()
        helper_test_op(data)

    """TODO: how to test
    Tensor([2,3]) * 2?
    Tensor([2,3]) + 2?
    and so on...
    change in prepare_test_op because it transforms everything to tensor
    """

if __name__ == '__main__':
    unittest.main(verbosity=2)
