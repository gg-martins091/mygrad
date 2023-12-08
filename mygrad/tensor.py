from __future__ import annotations

from typing import Union, Optional, Type, ClassVar
from mygrad.lazy import LazyBuffer
from mygrad.ops import LoadOps, Device
from mygrad.helpers import dtypes, DType
import numpy as np

class Function:
    def __init__(self, device:str, *tensors:Tensor):
        self.device = device
        self.needs_input_grad = [t.requires_grad for t in tensors]
        self.requires_grad = True if any(self.needs_input_grad) else None if None in self.needs_input_grad else False

        if self.requires_grad: self.parents = tensors

    def forward(self, *args, **kwargs): raise NotImplementedError(f"forward function not implemented for {type(self)}")
    def backward(self, *args, **kwargs): raise RuntimeError(f"forward function not implemented for {type(self)}")

    @classmethod
    def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
        ctx = fxn(x[0].device, *x)
        ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
        # if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx # used by autograd engine, we dont have it yet.
        return ret

import mygrad.mlops as mlops

class Tensor():
    __slots__= "lazydata", "grad", "requires_grad"
    default_type:ClassVar[DType] = dtypes.float32
    def __init__(self, data:Union[int, float, list, LazyBuffer], device:Optional[str]=None, requires_grad:Optional[bool]=None, dtype:Optional[DType]=None):
        device = Device.canonicalize(device)
        self.grad: Optional[Tensor] = None

        self.requires_grad = requires_grad

        if data.__class__ is list:
            assert dtype is None or dtype.np is not None, f"{dtype} does not have a numpy type"
            data = LazyBuffer.fromCPU(np.array([] if data is None else data, dtype=(dtype or Tensor.default_type).np))
            pass
        if isinstance(data, (int, float)):
            data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtypes.float32, device, data)

        if not isinstance(data, LazyBuffer): raise RuntimeError(f"cant create Tensor from {data!r} with type {type(data)}")
        self.lazydata = data

    def __repr__(self):
        return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

    def numpy(self): return self.lazydata._np

    @property
    def device(self) -> str: return self.lazydata.device

    def add(self, x):
        # TODO: should add broadcast here in case of different shapes
        return mlops.Add.apply(self, x)

    def __add__(self, x) -> Tensor: return self.add(x)

