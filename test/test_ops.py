
import teenygrad.tensor as tg
import unittest
from mygrad.tensor import Tensor
import time
import numpy as np

PRINT_VALUES=False

def helper_test_op(shps, teenygrad_fxn, mygrad_fxn=None, atol=1e-6, rtol=1e-3, grad_atol=1e-4, grad_rtol=1e-3, forward_only=True):
    if mygrad_fxn is None: mygrad_fxn = teenygrad_fxn

    #inputs
    teeny, my = prepare_test_op(shps)
    if PRINT_VALUES:
        print(f"{teeny=}")
        print(f"{my=}")

    st = time.monotonic()
    out = teenygrad_fxn(*teeny)
    teeny_fp = time.monotonic() - st

    st = time.monotonic()
    ret = mygrad_fxn(*my)
    my_fp = time.monotonic() - st


    def compare(s, x, y, atol, rtol):
        if PRINT_VALUES: print(s, x, y)
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

    print("\ntesting %40r   teenygrad/mygrad fp: %.2f / %.2f ms  bp: %.2f / %.2f ms " % (shps, teeny_fp*1000, my_fp*1000, torch_fbp*1000, tinygrad_fbp*1000), end="")

def prepare_test_op(shps):
    teeny = [tg.Tensor(sh) for sh in shps]
    my = [Tensor(sh) for sh in shps]


    return teeny, my


class TestOps(unittest.TestCase):
    def test_add_number(self):
        helper_test_op([1,4], lambda x, y: x+y, Tensor.add)
    def test_add_list(self):
        helper_test_op([[2,3,4],[5,6,8]], lambda x, y: x+y, Tensor.add)
    def test_add_mat(self):
        helper_test_op([[[2,3],[4,5]],[[6,7],[8,9]]], lambda x, y: x+y, Tensor.add)



if __name__ == '__main__':
    unittest.main(verbosity=2)
