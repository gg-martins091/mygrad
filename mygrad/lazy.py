# para tiper o retorno do metodo com o tipo da classe
from __future__ import annotations
import numpy as np
from .ops import LoadOps
from .helpers import dtypes

class LazyBuffer:
    device = "CPU"

    def __init__(self, buf: np.ndarray): self._np = buf
    def __repr__(self): return f"<LB {self.shape} {self.dtype}>"

    # no teenygrad temos apenas um device, nao precisa fazer nada.
    def copy_to_device(self, device:str) -> LazyBuffer: return self

    @property
    def dtype(self): return dtypes.from_np(self._np.dtype)
    @property
    def shape(self): return self._np.shape

    #TODO: tipar esses parametros
    @staticmethod
    def loadop(op, shape, dtype, device, arg=None, src=None) -> LazyBuffer:
        print(f"{op=}")
        print(f"{shape=}")
        print(f"{dtype=}")
        print(f"{device=}")
        print(f"{arg=}")
        print(f"{src=}")
        # if op == LoadOps.RAND: return LazyBuffer(np.random.default_rng(arg).random(size=shape, dtype=dtype.np))
        if op == LoadOps.CONST: return LazyBuffer(np.full(shape, arg, dtype=dtype.np))
        elif op == LoadOps.EMPTY: return LazyBuffer(np.empty(shape, dtype=dtype.np))
        else: raise NotImplementedError(op)

