from __future__ import annotations

from typing import Union, Optional, Type, ClassVar, Tuple, List
from mygrad.lazy import LazyBuffer
from mygrad.ops import LoadOps, Device
from mygrad.helpers import all_int, dtypes, DType, argfix
from mygrad.realize import run_schedule
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
    def backward(self, *args, **kwargs): raise RuntimeError(f"backward function not implemented for {type(self)}")

    @classmethod
    def apply(fxn:Type[Function], *x:Tensor, **kwargs) -> Tensor:
        ctx = fxn(x[0].device, *x)
        ret = Tensor(ctx.forward(*[t.lazydata for t in x], **kwargs), device=ctx.device, requires_grad=ctx.requires_grad)
        if ctx.requires_grad and not Tensor.no_grad: ret._ctx = ctx # used by autograd engine
        return ret

import mygrad.mlops as mlops

class Tensor():
    __slots__= "lazydata", "grad", "requires_grad", "grad", "_ctx"
    __deletable__ = ('_ctx',)
    default_type:ClassVar[DType] = dtypes.float32
    no_grad: ClassVar[bool] = False
    def __init__(self, data:Union[int, float, list, LazyBuffer, np.ndarray], device:Optional[str]=None, requires_grad:Optional[bool]=None, dtype:Optional[DType]=None):
        device = Device.canonicalize(device)
        self.grad: Optional[Tensor] = None

        self.requires_grad = requires_grad
        self._ctx: Optional[Function] = None

        if isinstance(data, LazyBuffer): assert dtype is None or dtype == data.dtype, "dtype does not match, and casting is not supported"
        elif data.__class__ is list:
            assert dtype is None or dtype.np is not None, f"{dtype} does not have a numpy type"
            data = LazyBuffer.fromCPU(np.array([] if data is None else data, dtype=(dtype or Tensor.default_type).np))
            pass
        elif isinstance(data, (int, float)):
            data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtypes.float32, device, data)
        elif isinstance(data, np.ndarray):
          assert dtype is None or dtype.np is not None, f"{dtype} doesn't have a numpy dtype"
          if data.shape == ():
            data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtype or dtypes.from_np(data.dtype), device, data.item())
          else:
            data = LazyBuffer.fromCPU(data.astype(dtype.np) if dtype is not None and dtype.np is not None else data)

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


    def realize(self) -> Tensor:
        run_schedule(self.lazydata.schedule())
        return self

    # TODO: understand teenygrad's assign
    # i think for now this is fine
    def assign(self, x) -> Tensor:
        self.lazydata = x.lazydata
        return self

    # ***** toposort and backward pass *****
    def deepwalk(self):
        def _deepwalk(node, visited, nodes):
          visited.add(node)
          if getattr(node, "_ctx", None):
            for i in node._ctx.parents:
              if i not in visited: _deepwalk(i, visited, nodes)
            nodes.append(node)
          return nodes
        return _deepwalk(self, set(), [])

    def backward(self) -> Tensor:
        assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
        # this is "implicit gradient creation"
        self.grad = Tensor(1, device=self.device, requires_grad=False)

        dw = reversed(self.deepwalk())
        for t0 in dw:
          assert (t0.grad is not None)
          grads = t0._ctx.backward(t0.grad.lazydata)
          grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
            for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
          for t, g in zip(t0._ctx.parents, grads):
            if g is not None and t.requires_grad:
              assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
              t.grad = g if t.grad is None else (t.grad + g)
          del t0._ctx
        return self

    # *** unary mlops ***
    def neg(self): return mlops.Neg.apply(self)


    # *** math functions (unary) ***
    def square(self): return self*self

    # *** binary mlops ***
    def _broadcasted(self, y: Union[Tensor, float], reverse=False) -> Tuple[Tensor,Tensor]:
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

    def __radd__(self, x) -> Tensor: return self.add(x, reverse=True)
    def __rsub__(self, x) -> Tensor: return self.sub(x, reverse=True)
    def __rmul__(self, x) -> Tensor: return self.mul(x, reverse=True)
    def __rtruediv__(self, x) -> Tensor: return self.div(x, reverse=True)
    def __rpow__(self, x) -> Tensor: return self.pow(x, reverse=True)
    def __rmatmul__(self, x) -> Tensor: return self.matmul(x, reverse=True)

    def __iadd__(self, x) -> Tensor: return self.assign(self.add(x))
    def __isub__(self, x) -> Tensor: return self.assign(self.sub(x))
    def __imul__(self, x) -> Tensor: return self.assign(self.mul(x))
    def __itruediv__(self, x) -> Tensor: return self.assign(self.div(x))
    def __ipow__(self, x) -> Tensor: return self.assign(self.pow(x))
    def __imatmul__(self, x) -> Tensor: return self.assign(self.matmul(x))

    # *** creation helper functions ***
    @staticmethod
    def full(shape: Tuple[int, ...], fill_value, **kwargs) -> Tensor: return Tensor(fill_value, **kwargs).reshape([1]*len(new_shape := argfix(shape))).expand(new_shape)
    @staticmethod
    def zeros(*shape, **kwargs) -> Tensor: return Tensor.full(argfix(*shape), 0, **kwargs)


    # *** cast ops ***
    def cast(self, dtype:DType) -> Tensor: return mlops.Cast.apply(self, dtype=dtype) if self.dtype != dtype else self

    # *** reduce ops ***
    # TODO: understand this.
    def _reduce(self, fxn:Type[Function], axis:Optional[Union[int, Tuple[int, ...]]]=None, keepdim=False) -> Tensor:
        axis_: List[int] = list(range(len(self.shape))) if axis is None else ([axis] if isinstance(axis, int) else list(axis))
        axis_ = [x if x >= 0 else x+len(self.shape) for x in axis_]
        shape = tuple(s for i,s in enumerate(self.shape) if i not in axis_)
        if 0 in self.shape and 0 not in shape: return Tensor.full(tuple(1 if s == 0 else s for s in self.shape) if keepdim else shape, {mlops.Sum: 0, mlops.Max: -float("inf")}[fxn])
        ret = fxn.apply(self, new_shape=tuple([1 if i in axis_ else s for i,s in enumerate(self.shape)]))
        return ret if keepdim else ret.reshape(shape=shape)

    def sum(self, axis=None, keepdim=False): return self._reduce(mlops.Sum, axis, keepdim)
    def mean(self, axis=None, keepdim=False):
        assert all_int(self.shape), "does not support symbolic shape"
        out = self.sum(axis=axis, keepdim=keepdim)
        return out.mul(math.prod(out.shape)/math.prod(self.shape)) if 0 not in self.shape else out

    # *** movement mlops ***
    def reshape(self, shape, *args) -> Tensor:
        # new_shape = argfix(shape, *args)
        return mlops.Reshape.apply(self,shape=shape)
    def expand(self, shape, *args) -> Tensor: return mlops.Expand.apply(self, shape=tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))]))
    def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))

    # *** movement hlops ***
    def transpose(self, ax1=1, ax2=0) -> Tensor:
        order = list(range(len(self.shape)))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    # *** creation llop entrypoint ***
    @staticmethod
    def _loadop(op, sz, device:Optional[str]=None, dtype:Optional[DType]=None, arg=None, **kwargs):
        assert isinstance(sz, int), f"cannot create with symbolic size {sz} (what does this mean?)"
        return Tensor(LazyBuffer.loadop(op, (sz,), Tensor.default_type if dtype is None else dtype, Device.canonicalize(device), arg), dtype=dtype, device=device, **kwargs)

    _seed: int = int(time.time())
    @staticmethod
    def rand(*shape, **kwargs):
        Tensor._seed += 1
        return Tensor._loadop(LoadOps.RAND, math.prod((shape:=argfix(*shape))), arg=Tensor._seed, **kwargs).reshape(shape)

    # *** functional nn ops ***
    def linear(self, weights:Tensor, bias:Optional[Tensor]=None):
        ret = self.mul(weights) if len(weights.shape) == 1 else self.dot(weights)
        return ret.add(bias) if bias is not None else ret


    # *** mlops (unary) ***
    def relu(self): return mlops.Relu.apply(self)

    # *** activation functions (unary) ***
    def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()

    # *** rng hlops ***
    @staticmethod
    def uniform(*shape, low=0.0, high=1.0, **kwargs) -> Tensor:
        dtype = kwargs.pop('dtype', Tensor.default_type)
        ret = (high-low) * Tensor.rand(*shape, **kwargs)
        return ret.cast(dtype) + low

    @staticmethod
    def kaiming_uniform(*shape, a:float = 0.01, **kwargs) -> Tensor:
        bound = math.sqrt(3.0) * math.sqrt(2.0 / (1 + a **2)) / math.sqrt(math.prod(shape[1:]))

        return Tensor.uniform(*shape, low=-bound, high=bound, **kwargs)

