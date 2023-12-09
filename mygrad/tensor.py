from __future__ import annotations

from typing import Union, Optional, Type, ClassVar, Tuple
from mygrad.lazy import LazyBuffer
from mygrad.ops import LoadOps, Device
from mygrad.helpers import all_int, dtypes, DType, argfix
import numpy as np
import math
import time

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

        if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype does not match, and casting is not supported"
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

    def detach(self) -> Tensor: return Tensor(self.lazydata, self.device, requires_grad=False)
    def numpy(self):
        assert all_int(self.shape), f"no numpy if shape is symbolic, whatever that means, {self.shape}"
        assert self.dtype.np is not None, f"no numpy dtype for {self.dtype}"
        # will implement this when needed
        #return self.detach().cast(dtypes.from_np(self.dtype.np)).contiguous().to('CPU').realize().lazydata.realized.toCPU().reshape(self.shape)
        return self.lazydata._np

    @property
    def device(self) -> str: return self.lazydata.device
    @property
    def shape(self) -> Tuple[int, ...]: return self.lazydata.shape
    @property
    def dtype(self) -> DType: return self.lazydata.dtype


    # teenygrads assign is basically the same as tinygrad which does a bunch of stuff
    # i think this is fine
    def assign(self, x) -> Tensor:
        self.lazydata = x.lazydata
        return self

    # *** unary mlops ***
    def neg(self): return mlops.Neg.apply(self)

    # *** binary mlops ***
    def _broadcasted(self, y: Union[Tensor, float], reverse=False) -> Tuple[Tensor,Tensor]:
        print("_broadcasted")
        # this is not yet broadcasting anything, it is simply turning y into a Tensor if it is not one
        x: Tensor = self
        if not isinstance(y, Tensor):
            y = Tensor(y, device=self.device, requires_grad=False, dtype=self.dtype)

        if reverse: x, y = y, x

        return x, y

    # broadcast is only used if x is not a Tensor and for other devices, because numpy does it for us, right?
    def add(self, x, reverse=False):
        #idk why we need this, so lets wait until something breaks
        # x = self._to_float(x)
        return mlops.Add.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else self
    def sub(self, x, reverse=False):
        # x = self._to_float(x)
        return mlops.Sub.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x else (-self if reverse else self)
    def mul(self, x, reverse=False):
        # x = self._to_float(x)
        if x.__class__ is not Tensor and x == 0.0: return mlops.Zero.apply(self)
        if x.__class__ is not Tensor and x == -1.0: return -self
        return mlops.Mul.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else self
    def div(self, x, reverse=False):
        # x = self._to_float(x)
        #the logic here just tries to avoid broadcasting, if possible simply multiply by 1/x
        return mlops.Div.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or reverse or not x or not dtypes.is_float(self.dtype) else self.mul(1/x)
    def pow(self, x, reverse=False):
        # x = self._to_float(x)
        #TODO: understand what is that wizardry on tinygrad's pow
        # this should not be a binary op
        return mlops.Pow.apply(*self._broadcasted(x, reverse)) if x.__class__ is Tensor or x != 1.0 else self
    def matmul(self, x:Tensor, reverse=False) -> Tensor: return x.dot(self) if reverse else self.dot(x)
    def dot(self, x):
        #TODO: understand what is that wizardry on tinygrad's matmul
        # this should not be a binary op
        return mlops.MatMul.apply(self, x)

    # ***** op wrappers (to make the typechecker happy) *****
    def __neg__(self) -> Tensor: return self.neg()

    def __add__(self, x) -> Tensor: return self.add(x)
    def __sub__(self, x) -> Tensor: return self.sub(x)
    def __mul__(self, x) -> Tensor: return self.mul(x)
    def __truediv__(self, x) -> Tensor: return self.div(x)
    def __pow__(self, x) -> Tensor: return self.pow(x)
    def __matmul__(self, x) -> Tensor: return self.matmul(x)

    # *** creation llop entrypoint ***
    @staticmethod
    def _loadop(op, sz, device:Optional[str]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
        assert isinstance(sz, int), f"cannot create with symbolic size {sz} (what does this mean?)"
        return Tensor(LazyBuffer.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

    _seed: int = int(time.time())
    @staticmethod
    def rand(*shape, **kwargs):
        Tensor._seed += 1
        return Tensor._loadop(LoadOps.RAND, math.prod((shape:=argfix(*shape))), arg=Tensor._seed, **kwargs)#.reshape(shape)

    # *** functional nn ops ***

    def linear(self, weights:Tensor, bias:Optional[Tensor]=None):
        ret = self.mul(weights) if len(weights.shape) == 1 else self.dot(weights)
        return ret.add(bias) if bias is not None else ret


    # *** rng hlops ***

    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        dtype = kwargs.pop('dtype', Tensor.default_type)
        return ((high-low) * Tensor.rand(*shape, **kwargs)).cast(dtype) + low

    @staticmethod
    def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a **2)) / math.sqrt(math.prod(shape[1:]))

        return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

