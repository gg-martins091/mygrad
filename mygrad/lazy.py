# para tiper o retorno do metodo com o tipo da classe
from __future__ import annotations
import numpy as np
from mygrad.ops import LoadOps, BinaryOps
from mygrad.helpers import dtypes

class LazyBuffer:
    device = "CPU"

    def __init__(self, buf: np.ndarray): self._np = buf
    def __repr__(self): return f"<LB {self.shape} {self.dtype}>: {self._np}"

    # no teenygrad temos apenas um device, nao precisa fazer nada.
    def copy_to_device(self, device:str) -> LazyBuffer: return self

    @property
    def dtype(self): return dtypes.from_np(self._np.dtype)
    @property
    def shape(self): return self._np.shape

    #TODO: type theese params
    @staticmethod
    def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
        # if op == LoadOps.RAND: return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        if op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY: return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else: raise NotImplementedError(op)

    @staticmethod
    def fromCPU(data) -> LazyBuffer:
        return LazyBuffer(data)


    def e(self, op, *srcs:LazyBuffer):
        if op == BinaryOps.ADD: ret = self._np + srcs[0]._np
        else: raise NotImplementedError(op)

        return LazyBuffer(ret.astype(self.dtype.np if len(srcs) == 0 else max(self.dtype, *[s.dtype for s in srcs]).np, copy=False))
