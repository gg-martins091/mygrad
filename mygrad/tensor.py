from __future__ import annotations

import traceback
from typing import Union, Optional, Type, ClassVar, Tuple, List, Sequence, Iterable, Set, Any
from mygrad.lazy import LazyBuffer
from mygrad.ops import LoadOps, Device
from collections import defaultdict
from mygrad.helpers import all_int, dtypes, DType, argfix, flatten, round_up, make_pair
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
    training: ClassVar[bool] = False
    class train:
        def __init__(self, val=True): self.val = val
        def __enter__(self): self.prev, Tensor.training = Tensor.training, self.val
        def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any): Tensor.training = self.prev

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

    def __hash__(self): return id(self)

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


    @staticmethod
    def corealize(lst:Iterable[Tensor]):
        seen:Set[LazyBuffer] = set()
        sched = []
        for t in lst: sched += t.lazydata.schedule(seen)
        run_schedule(sched)

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
        print("tensor.backward()")
        assert self.shape == tuple(), f"backward can only be called for scalar tensors, but it has shape {self.shape})"

        # fill in the first grad with one. don't use Tensor.ones because we don't need contiguous
        # this is "implicit gradient creation"
        self.grad = Tensor(1, device=self.device, requires_grad=False)

        dw = reversed(self.deepwalk())

        dw = reversed(self.deepwalk())
        for t0 in dw:
          print(f"(before assert) t0={t0}")
          assert (t0.grad is not None)
          print(f"{t0=}")
          print(f"{t0._ctx=}")
          print(f"{t0.grad=}")
          print(f"{t0._ctx.parents=}")
          grads = t0._ctx.backward(t0.grad.lazydata)

          print(f"{grads=}")
          grads = [Tensor(g, device=self.device, requires_grad=False) if g is not None else None
            for g in ([grads] if len(t0._ctx.parents) == 1 else grads)]
          print(f"{grads=}")
          for t, g in zip(t0._ctx.parents, grads):
            if g is not None and t.requires_grad:
              assert g.shape == t.shape, f"grad shape must match tensor shape, {g.shape!r} != {t.shape!r}"
              t.grad = g if t.grad is None else (t.grad + g)
          print(f"{t0=}")
          print()
          del t0._ctx
        return self

    # *** unary mlops ***
    def neg(self): return mlops.Neg.apply(self)
    def exp(self): return mlops.Exp.apply(self)
    def log(self): return mlops.Log.apply(self)
    def contiguous(self): return mlops.Contiguous.apply(self)


    # *** math functions (unary) ***
    def square(self): return self*self
    def sign(self): return self / (self.abs() + 1e-10)
    def abs(self): return self.relu() + (-self).relu()

    # *** binary mlops ***
    def _broadcasted(self, y: Union[Tensor, float], reverse=False) -> Tuple[Tensor,Tensor]:
        x: Tensor = self
        if not isinstance(y, Tensor):
            if 0 in x.shape: return x, x.full_like(y)
            y = Tensor(y, device=self.device, requires_grad=False, dtype=self.dtype if self.dtype != dtypes.bool else dtypes.float32)
        if reverse: x, y = y, x
        if (xshape:=x.shape) == (yshape:=y.shape): return (x, y)

        shape_delta = len(xshape) - len(yshape)
        if shape_delta > 0: y = y.reshape((1,) * shape_delta + yshape)
        elif shape_delta < 0: x = x.reshape((1,) * -shape_delta + xshape)
        if (xshape:=x.shape) == (yshape:=y.shape): return (x, y)

        shape_ret = tuple([max(x, y) for x, y in zip(xshape, yshape)])
        if xshape != shape_ret: x = x.expand(shape_ret)
        if yshape != shape_ret: y = y.expand(shape_ret)
        return (x, y)


    # *** processing ops ***
    def _pool(self, k_:Tuple[int, ...], stride:Union[Tuple[int, ...], int]=1, dilation:Union[Tuple[int, ...], int]=1) -> Tensor:
      assert len(self.shape) >= len(k_), f"can't pool {self.shape} with {k_}"
      assert all_int(self.shape) and all_int(k_), f"does not support symbolic {self.shape=}, {k_=}"
      s_, d_ = make_pair(stride, len(k_)), make_pair(dilation, len(k_))
      assert len(k_) == len(s_) and len(k_) == len(d_), f"stride/dilation mismatch kernel:{k_} stride:{s_} dilation:{d_}"
      slc_prefix, prefix, i_ = [(0,x) for x in self.shape[0:-len(k_)]], self.shape[0:-len(k_)], self.shape[-len(k_):]
      if any(k > s for k,s in zip(k_, s_)) or any(d != 1 for d in d_):
        o_ = [(i - d * (k-1) - 1)//s + 1 for i,d,k,s in zip(i_, d_, k_, s_)]
        e_ = [math.ceil(k*(i+d) / i) for k,i,d in zip(k_, i_, d_)]  # expands such that we don't need padding
        xup = self.reshape(*prefix, *flatten((1,i) for i in i_)).expand(*prefix, *flatten((e,i) for e,i in zip(e_, i_))).reshape(*prefix, *[e*i for e,i in zip(e_, i_)])
        # slide by dilation
        xup = xup.slice(slc_prefix + [(0,k*(i+d)) for k,i,d in zip(k_, i_, d_)])
        xup = xup.reshape(*prefix, *flatten((k,i+d) for k,i,d in zip(k_, i_, d_)))
        xup = xup.slice(slc_prefix + flatten(((0,k), (0,o*s)) for k,o,s in zip(k_, o_, s_)))
        # handle stride, and permute to move reduce to the end
        xup = xup.reshape(*prefix, *flatten((k,o,s) for k,o,s in zip(k_, o_, s_)))
        xup = xup.slice(slc_prefix + flatten(((0,k), (0,o), (0,1)) for k,o in zip(k_, o_)))
        xup = xup.reshape(*prefix, *flatten((k,o) for k,o in zip(k_, o_)))
        return xup.permute(*range(len(prefix)), *[len(prefix)+i*2+1 for i in range(len(k_))], *[len(prefix)+i*2 for i in range(len(k_))])
      # TODO: once the shapetracker can optimize well, remove this alternative implementation. or not if the CPU implementation doesn't use ShapeTracker
      o_ = [(i+(s-k))//s for i,s,k in zip(i_, s_, k_)]
      xup = self.slice(slc_prefix + [(0,o*s) for o,s in zip(o_, s_)])
      xup = xup.reshape(*prefix, *flatten(((o, s) for o,s in zip(o_, s_))))
      xup = xup.slice(slc_prefix + flatten(((0,o), (0,k)) for o,k in zip(o_, k_)))
      return xup.permute(*range(len(prefix)), *[len(prefix)+i*2 for i in range(len(k_))], *[len(prefix)+i*2+1 for i in range(len(k_))])

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
    def dot(self, w:Tensor):
        #TODO: understand what is that wizardry on tinygrad's matmul
        # this should not be a binary op
        n1, n2 = len(self.shape), len(w.shape)
        assert n1 != 0 and n2 != 0, f"both arguments to matmul need to be at least 1D, but they are {n1}D and {n2}D"
        assert self.shape[-1] == w.shape[-min(n2, 2)], f"Input Tensor shapes {self.shape} and {w.shape} cannot be multiplied ({self.shape[-1]} != {w.shape[-min(n2, 2)]})"
        x = self.reshape(*self.shape[0:-1], *[1]*min(n1-1, n2-1, 1), self.shape[-1])
        w = w.reshape(*w.shape[0:-2], *[1]*min(n1-1, n2-1, 1), *w.shape[-min(n2, 2):]).transpose(-1, -min(n2, 2))
        return (x*w).sum(-1)

    def where(self:Tensor, input_:Union[Tensor, float], other:Union[Tensor, float]):
        x_,y = self._broadcasted(input_)
        x,z = x_._broadcasted(other)
        return mlops.Where.apply(x, *y._broadcasted(z))

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

    def __lt__(self, x) -> Tensor: return mlops.Less.apply(*self._broadcasted(x, False))
    def __gt__(self, x) -> Tensor: return mlops.Less.apply(*self._broadcasted(x, True))
    def __ne__(self, x) -> Tensor: return (self < x) + (self > x)
    def __eq__(self, x) -> Tensor: return 1.0-(self != x)

    # *** creation helper functions ***
    @staticmethod
    def full(shape: Tuple[int, ...], fill_value, **kwargs) -> Tensor: return Tensor(fill_value, **kwargs).reshape([1]*len(new_shape := argfix(shape))).expand(new_shape)
    @staticmethod
    def zeros(*shape, **kwargs) -> Tensor: return Tensor.full(argfix(*shape), 0, **kwargs)

    @staticmethod
    def arange(start, stop=None, step=1, **kwargs):
        if stop is None: stop, start = start, 0
        return Tensor.full((math.ceil((stop-start)/step),), step, **kwargs).cumsum() + (start - step)

    @staticmethod
    def eye(dim:int, **kwargs): return Tensor.full((dim,1),1,**kwargs).pad(((0,0),(0,dim))).reshape(dim*(dim+1)).shrink(((0,dim*dim),)).reshape(dim, dim)

    def full_like(self, fill_value, **kwargs): return Tensor.full(self.shape, fill_value=fill_value, dtype=kwargs.pop("dtype", self.dtype), device=kwargs.pop("device", self.device), **kwargs)
    def zeros_like(self, **kwargs): return self.full_like(0, **kwargs)
    def ones_like(self, **kwargs): return self.full_like(1, **kwargs)


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

    def sum(self, axis=None, keepdim=False):
        return self._reduce(mlops.Sum, axis, keepdim)
    def mean(self, axis=None, keepdim=False):
        assert all_int(self.shape), "does not support symbolic shape"
        out = self.sum(axis=axis, keepdim=keepdim)
        return out.mul(math.prod(out.shape)/math.prod(self.shape)) if 0 not in self.shape else out

    def _cumsum(self, axis:int=0, _first_zero=False) -> Tensor: return self.transpose(axis,-1).pad2d((self.shape[axis]-int(not _first_zero),0))._pool((self.shape[axis],)).sum(-1).transpose(axis,-1)
    def cumsum(self, axis:int=0) -> Tensor:
        # TODO: someday the optimizer will find this on it's own
        # for now this is a two stage cumsum
        SPLIT = 256
        if self.shape[axis] <= SPLIT*2: return self._cumsum(axis)
        ret = self.transpose(axis,-1).pad2d((round_up(self.shape[axis], SPLIT)-self.shape[axis], 0))
        ret = ret.reshape(*ret.shape[0:-1], ret.shape[-1]//SPLIT, SPLIT)._cumsum(-1)
        base_add = ret[..., -1]._cumsum(-1, _first_zero=True)[..., :-1]
        base_add = base_add.unsqueeze(-1).expand(*base_add.shape, ret.shape[-1])
        def fix(x:Tensor): return x.reshape(*ret.shape[0:-2], ret.shape[-2] * ret.shape[-1])[..., -self.shape[axis]:].transpose(axis,-1)
        return fix(ret) + fix(base_add)

    def _softmax(self, axis):
        m = self - self.max(axis=axis, keepdim=True)
        e = m.exp()
        return m, e, e.sum(axis=axis, keepdim=True)

    def log_softmax(self, axis=-1):
        m, _, ss = self._softmax(axis)
        return m - ss.log()

    def argmax(self, axis=None, keepdim=False):
        if axis is None:
            idx = (self == self.max(axis)) * Tensor.arange(math.prod(self.shape)-1,-1,-1, dtype=dtypes.int32, requires_grad=False, device=self.device).reshape(self.shape)
            return math.prod(self.shape) - idx.max() - 1
        axis = axis + len(self.shape) if axis < 0 else axis
        m = self == self.max(axis=axis, keepdim=True)
        idx = m * Tensor.arange(self.shape[axis]-1,-1,-1, dtype=dtypes.int32, requires_grad=False, device=self.device).reshape(self.shape[axis], *[1]*(self.ndim-axis-1))
        return self.shape[axis]-idx.max(axis=axis, keepdim=keepdim)-1

    def argmin(self, axis=None, keepdim=False): return (-self).argmax(axis=axis, keepdim=keepdim)


    def slice(self, arg:Sequence[Optional[Tuple[int, int]]], value:float=0) -> Tensor:
        arg_ = tuple([a if a is not None else (0,s) for s,a in zip(self.shape, arg)])
        padding = tuple([(max(0, -p[0]), max(0, p[1]-self.shape[i])) for i,p in enumerate(arg_)])
        return self.pad(padding, value=value).shrink(tuple([(p[0] + padding[i][0], p[1] + padding[i][0]) for i,p in enumerate(arg_)]))

    def pad2d(self, padding:Union[List[int], Tuple[int, ...]], value:float=0) -> Tensor:
        slc = [(-p0, s+p1) for p0,p1,s in zip(padding[::2], padding[1::2], self.shape[::-1])][::-1]
        return self.slice([(0,s) for s in self.shape[:-(len(padding)//2)]] + slc, value=value)

    def unsqueeze(self, dim) -> Tensor:
        if dim < 0: dim = len(self.shape) + dim + 1
        return self.reshape(self.shape[:dim] + (1,) + self.shape[dim:])

    # *** movement mlops ***
    def reshape(self, shape, *args) -> Tensor:
        new_shape = argfix(shape, *args)
        return mlops.Reshape.apply(self,shape=new_shape)
    def expand(self, shape, *args) -> Tensor: return mlops.Expand.apply(self, shape=tuple([x if x != -1 else s for s,x in zip(self.shape, argfix(shape, *args))]))
    def permute(self, order, *args) -> Tensor: return mlops.Permute.apply(self, order=argfix(order, *args))
    def pad(self, arg:Tuple[Optional[Tuple[int, int]], ...], value:float=0.0) -> Tensor:
        if all(x is None or x == (0,0) for x in arg): return self
        ret = mlops.Pad.apply(self, arg=(narg:=tuple(x if x is not None else (0,0) for x in arg)))
        return ret if 0 == value else ret + mlops.Pad.apply(Tensor.ones_like(self), arg=narg).where(0, value)
    def shrink(self, arg:Tuple[Optional[Tuple[int, int]], ...]) -> Tensor: return mlops.Shrink.apply(self, arg=tuple(x if x is not None else (0,s) for x,s in zip(arg, self.shape))) if any(x is not None and x != (0,s) for x,s in zip(arg, self.shape)) else self
    def flip(self, axis, *args) -> Tensor: return mlops.Flip.apply(self, axis=[x if x >= 0 else x+len(self.shape) for x in argfix(axis, *args)])

    # *** movement hlops ***
    # - Negative indices are taken relative to the end of the sequence, so X[-2] returns the 2nd-to-last element
    # - A slice i:j returns the elements with indices in [i, j)
    #    - If omitted, i and j will default to 0 and N, respectively, where N is the length of the sequence
    #    - Negative values for i and j are taken relative to the end of the sequence
    #    - Both i and j will be clamped to the range (-N, N], where N in the length of the sequence
    # - Indexing with None on a given axis will add a new dimension of size one before that axis
    # - Empty slices are not allowed (tensors with 0s in shape have to be supported first, for all backends).
    # - For a slice [i:j:k] finding the correct indices is delegated to slice.indices(len).
    # - Strides > 1 and < 0 are now allowed!:
    #    - This works by applying Shrink -> [[Flip -> ] Pad -> Reshape -> Shrink] -> Reshape (ops in brackets are optional)
    #    - Idea of stride < 0 support:
    #        - Do the slice first, flip the axes were slice.step is negative, do slice.step -> -slice.step. Go to steps below.
    #    - Idea of stride `s` > 1 support (Pad -> Reshape -> Shrink):
    #        - Instead of doing [::s] on axis [dim_sz], do [:, 0] on axes [dim_sz_padded // s, s].
    #        - So pad dim_sz with as many zeros as needed (dim_sz -> dim_sz_padded) so that reshape to [dim_sz_padded // s, s]
    #          is possible.
    #        - Apply Shrink to do the slice [:, 0] on axes of shapes [dim_sz_padded // s, s].
    # - Fancy indexing and combined indexing is supported
    #    - Combined indexing works by letting regular slicing finish first -> computing the resulting dims w.r.t to Tensors passed in -> fancy indexing
    #    - Any Tensors passed in __getitem__ will perform (CMPEQ with arange -> MUL with self -> SUM_REDUCE) iteratively
    #        - The first iteration will expand the dim of self while consecutive iterations will reduce the dim
    #    - There's a special case where a permute is needed at the end:
    #        - if first Tensor passed in (expand dims) is not at dim 0
    #        - and following Tensors does not follow consecutively to the end of fancy indexing's dims
    # val: Union[int, slice, Tensor, None, Ellipsis, Tuple[Union[int, slice, Tensor, None, Ellipsis], ...]]
    def __getitem__(self, val) -> Tensor: 
        def normalize_int(e, i, dim_sz):
            if -dim_sz <= e < dim_sz: return e if e != -1 else dim_sz-1
            raise IndexError(f"index {e} is out of bounds for dimension {i} with size {self.shape[i]}")

        orig_slices = list(val) if isinstance(val, tuple) else [val]
        count = defaultdict(list)
        for i,v in enumerate(orig_slices): count[type(v)].append(i)

        if (num_slices := len(count[int]) + len(count[slice]) + len(count[Tensor])) > len(self.shape): raise IndexError(f"too many indices for tensor of dimension {len(self.shape)}")
        if len(ellipsis_found := count[type(Ellipsis)]) > 1: raise IndexError("an index can only have a single ellipsis ('...')")

        ellipsis_idx = ellipsis_found[0] if ellipsis_found else len(orig_slices)
        orig_slices[ellipsis_idx:ellipsis_idx+1] = [slice(None)] * (len(self.shape) - num_slices)

        valid_slices = [v for v in orig_slices if v is not None]
        valid_slices = [v if isinstance(v, slice) else slice(y_ := normalize_int(v, i, dim_sz), y_+1) if isinstance(v, int) else slice(None) for i, (v, dim_sz) in enumerate(zip(valid_slices, self.shape))]

        start, stop, strides = zip(*y) if (y := [s.indices(dim_sz) for s, dim_sz in zip(valid_slices, self.shape)]) else ((), (), ())
        new_slice = tuple(((0, 0) if e < s else (s, e)) if st > 0 else ((0, 0) if e > s else (e+1, s+1)) for s, e, st in zip(start, stop, strides))
        sliced_tensor = self.shrink(new_slice).flip(axis=[i for i, s in enumerate(strides) if s < 0])
        new_shape = sliced_tensor.shape
        if any(abs(s) != 1 for s in strides):
          strides = tuple(abs(s) for s in strides)
          # Pad: add pad at the end: [dim_sz] -> [dim_sz_padded]
          padded_tensor = sliced_tensor.pad(tuple((0, s-(dim_sz % s) if dim_sz % s != 0 else 0) for s, dim_sz in zip(strides, sliced_tensor.shape)))
          # Reshape: [dim_sz_padded] -> [dim_sz_padded // s, s]
          reshaped_tensor = padded_tensor.reshape(flatten([sh // s, s] for sh, s in zip(padded_tensor.shape, strides)))
          new_shape = reshaped_tensor.shape[::2]
          # Shrink: do [:, 0]
          sliced_tensor = reshaped_tensor.shrink(tuple(flatten(((0, sh), (0, 1)) for sh in new_shape)))

        final_shape, it_shape, dim, tensors, dim_collapsed = [], iter(new_shape), [], [], 0
        for i,s in enumerate(orig_slices):
            if s is None: final_shape.append(1)
            else: # s is int or slice or Tensor
                dim_shape = next(it_shape)
                if isinstance(s, int):
                    dim_collapsed += 1
                else:
                    assert isinstance(dim_shape, int), f"does not support symbolic shape {dim_shape}"
                    final_shape.append(dim_shape)
                    if isinstance(s, Tensor):
                        tensors.append(s)
                        dim.append(i-dim_collapsed)
        ret = sliced_tensor.reshape(tuple(final_shape))

        if tensors: # Fancy/tensor indexing
            # normalize idx
            # TODO: first contiguous fixes torch+cpu_only CI, but it causes llvm to fail. Second one fixes llvm
            idx = [t.sign().contiguous().__neg__().contiguous().relu() * ret.shape[d] + t for d,t in zip(dim, tensors)]
            max_dim = max(i.ndim for i in idx)
            # compute sum_dim, arange, and idx
            sum_dim = [d if n==0 else d+max_dim-n for n,d in enumerate(dim)]
            arange = [Tensor.arange(ret.shape[d], dtype=dtypes.int32, requires_grad=False, device=self.device).reshape(*[1]*sd, ret.shape[d], *[1]*(ret.ndim + max_dim - n - sd - 1)) for n,(sd,d) in enumerate(zip(sum_dim, dim))]
            first_idx = [idx[0].reshape(*[1]*dim[0], *[1]*(1 + max_dim - idx[0].ndim), *idx[0].shape, *[1]*(ret.ndim - dim[0] - 1))]
            rest_idx = [i.reshape(*[1]*dim[0], *[1]*(max_dim - i.ndim), *i.shape, *[1]*(ret.ndim - dim[0] - n)) for n,i in enumerate(idx[1:], 1)]
            idx = first_idx + rest_idx
            ret = ret.reshape(*ret.shape[:sum_dim[0]+1], *[1]*max_dim, *ret.shape[sum_dim[0]+1:])
            # iteratively fancy index
            for a,i,sd in zip(arange, idx, sum_dim): ret = (a==i).mul(ret).sum(sd)
              # special permute case
            if dim[0] != 0 and len(dim) != 1 and dim != list(range(dim[0], dim[-1]+1)):
                ret_dims = list(range(ret.ndim))
                ret = ret.permute(ret_dims[dim[0]:dim[0]+max_dim] + ret_dims[:dim[0]] + ret_dims[dim[0]+max_dim:])
        return ret

    def __setitem__(self,s,v): return self.__getitem__(s).assign(v)

    def transpose(self, ax1=1, ax2=0) -> Tensor:
        order = list(range(len(self.shape)))
        order[ax1], order[ax2] = order[ax2], order[ax1]
        return self.permute(order)

    def flatten(self, start_dim=0): return self.reshape(shape=self.shape[:start_dim] + (-1,))

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

    def binary_crossentropy(self, y:Tensor) -> Tensor:
        return (-y*self.log() - (1-y)*(1-self).log()).mean()

    def sparse_categorical_crossentropy(self, Y, ignore_index=-1) -> Tensor:
        # NOTE: self is a logits input
        loss_mask = Y != ignore_index
        print(f"{loss_mask=}")
        print(f"{self.shape=}")
        y_counter = Tensor.arange(self.shape[-1], dtype=dtypes.int32, requires_grad=False, device=self.device).unsqueeze(0).expand(Y.numel(), self.shape[-1])
        print(f"{y_counter=}")
        y = ((y_counter == Y.flatten().reshape(-1, 1)).where(-1.0, 0) * loss_mask.reshape(-1, 1)).reshape(*Y.shape, self.shape[-1])
        return self.log_softmax().mul(y).sum() / loss_mask.sum()


    # *** mlops (unary) ***
    def relu(self): return mlops.Relu.apply(self)
    def max(self, axis=None, keepdim=False): return self._reduce(mlops.Max, axis, keepdim)

    # *** activation functions (unary) ***
    def leakyrelu(self, neg_slope=0.01): return self.relu() - (-neg_slope*self).relu()
    def softplus(self, beta=1): return (1/beta) * (1 + (self*beta).exp()).log()

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


    # convenience
    @property
    def ndim(self) -> int: return len(self.shape)
    def numel(self) -> int: return math.prod(self.shape)
