from __future__ import annotations
from mygrad.lazy import LazyBuffer
from mygrad.tensor import Function
from mygrad.ops import LoadOps, BinaryOps, UnaryOps, ReduceOps
from mygrad.helpers import DType, Tuple, argsort


class Neg(Function):
  def forward(self, x: LazyBuffer) -> LazyBuffer: return x.e(UnaryOps.NEG, x)
  def backward(self, x: LazyBuffer) -> LazyBuffer: return x.e(UnaryOps.NEG, x)

class Add(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.ADD, y)

class Sub(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.SUB, y)

class Mul(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.MUL, y)

class Div(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.DIV, y)

class Pow(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.POW, y)

class MatMul(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.MATMUL, y)

class Zero(Function):
  def forward(self, x: LazyBuffer) -> LazyBuffer: return x.const(0)
  def backward(self, grad: LazyBuffer) -> LazyBuffer: return grad.const(0)

class Cast(Function):
  def forward(self, x: LazyBuffer, dtype:DType, bitcast:bool=False) -> LazyBuffer:
    self.input_dtype, self.bitcast = x.dtype, bitcast
    return x.cast(dtype, bitcast)
  def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    return grad_output.cast(self.input_dtype, self.bitcast)

class Reshape(Function):
  def forward(self, x: LazyBuffer, shape) -> LazyBuffer:
    self.input_shape = x.shape
    return x.reshape(shape)

  def backward(self, grad_output: LazyBuffer) -> LazyBuffer:
    return grad_output.reshape(self.input_shape)

class Expand(Function):
  def forward(self, x:LazyBuffer, shape:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.expand(shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.r(ReduceOps.SUM, self.input_shape)

class Permute(Function):
  def forward(self, x:LazyBuffer, order:Tuple[int, ...]) -> LazyBuffer:
    self.input_order = order
    return x.permute(order)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.permute(argsort(self.input_order))
