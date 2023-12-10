# para tiper o retorno do metodo com o tipo da classe
from __future__ import annotations
import numpy as np
from mygrad.ops import LoadOps, BinaryOps, UnaryOps, ReduceOps
from mygrad.helpers import dtypes, DType, STV, DEBUG 
class LazyBuffer:
    device = "CPU"

    def __init__(self, buf: np.ndarray): self._np = buf
    def __repr__(self): return f"<LB {self.shape} {self.dtype}> {self._np if STV else None}"

    # no teenygrad temos apenas um device, nao precisa fazer nada.
    def copy_to_device(self, device:str) -> LazyBuffer: return self

    @property
    def dtype(self): return dtypes.from_np(self._np.dtype)
    @property
    def shape(self): return self._np.shape

    #TODO: type theese params
    @staticmethod
    def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
        if op == LoadOps.RAND:
            np.random.default_rng(arg).random(size=shape, dtype=dtype.np)
            return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        if op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY: return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else: raise NotImplementedError(op)

    def schedule(self, seen=None): return []

    @staticmethod
    def fromCPU(data) -> LazyBuffer:
        return LazyBuffer(data)


    def const(self, v) -> LazyBuffer:
        # return LazyBuffer(np.full_like(self._np, v))

        # this one I tried implementing myself, it worked, but turns out it was closer to tinygrad's implementation than teenygrad's.
        # is that because teenygrad assumes only one device? i'll keep my shot at it, it is still different from tiny's, but lets see...
        return LazyBuffer.loadop(LoadOps.CONST, self.shape, self.dtype, self.device, v)

    def cast(self, dtype:DType, bitcast:bool=False) -> LazyBuffer: return LazyBuffer(self._np.view(dtype.np) if bitcast else self._np.astype(dtype.np))

    def e(self, op, *srcs:LazyBuffer):
        if DEBUG >= 1: print(op, self, srcs)
        if op == UnaryOps.NEG: ret = -self._np
        elif op == BinaryOps.ADD: ret = self._np + srcs[0]._np
        elif op == BinaryOps.SUB: ret = self._np - srcs[0]._np
        elif op == BinaryOps.MUL: ret = self._np * srcs[0]._np
        elif op == BinaryOps.DIV: ret = self._np / srcs[0]._np
        elif op == BinaryOps.CMPLT: ret = self._np < srcs[0]._np
        elif op == BinaryOps.MAX: ret = np.maximum(self._np, srcs[0]._np)

        # theese should not be here
        elif op == BinaryOps.POW: ret = self._np ** srcs[0]._np
        elif op == BinaryOps.MATMUL: ret = self._np @ srcs[0]._np
        else: raise NotImplementedError(op)

        return LazyBuffer(ret.astype(self.dtype.np if len(srcs) == 0 else max(self.dtype, *[s.dtype for s in srcs]).np, copy=False))

    def r(self, op, new_shape):
        if DEBUG >= 1: print(op, self, new_shape)
        assert len(self.shape) == len(new_shape), "reduce shapes must have same dimensions"
        axis = tuple(i for i,(a,b) in enumerate(zip(self.shape, new_shape)) if a != b)
        if op == ReduceOps.SUM: return LazyBuffer(self._np.sum(axis, dtype=self._np.dtype, keepdims=True))
        elif op == ReduceOps.MAX: return LazyBuffer(self._np.max(axis, keepdims=True))
        else: raise NotImplementedError(op)

    # move ops
    def reshape(self, shape) -> LazyBuffer: return LazyBuffer(self._np.reshape(shape))
    def expand(self, arg): return LazyBuffer(np.broadcast_to(self._np, arg))
    def permute(self, arg): return LazyBuffer(self._np.transpose(arg))
