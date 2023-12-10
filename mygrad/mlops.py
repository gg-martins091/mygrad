from __future__ import annotations
from mygrad.lazy import LazyBuffer
from mygrad.tensor import Function
from mygrad.ops import LoadOps, BinaryOps, UnaryOps, ReduceOps
from mygrad.helpers import DType, Tuple, argsort
from typing import Optional, Tuple


class Neg(Function):
  def forward(self, x: LazyBuffer) -> LazyBuffer: return x.e(UnaryOps.NEG, x)
  def backward(self, x: LazyBuffer) -> LazyBuffer: return x.e(UnaryOps.NEG, x)

class Add(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.ADD, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return grad_output if self.needs_input_grad[0] else None, \
           grad_output if self.needs_input_grad[1] else None

class Sub(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    return x.e(BinaryOps.SUB, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
      return grad_output if self.needs_input_grad[0] else None, \
             grad_output.e(UnaryOps.NEG) if self.needs_input_grad[1] else None

class Mul(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.MUL, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None

class Div(Function):
  def forward(self, x: LazyBuffer, y: LazyBuffer) -> LazyBuffer:
    self.x, self.y = x, y
    return x.e(BinaryOps.DIV, y)

  def backward(self, grad_output:LazyBuffer) -> Tuple[Optional[LazyBuffer], Optional[LazyBuffer]]:
    return self.y.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[0] else None, \
           self.x.e(BinaryOps.MUL, grad_output) if self.needs_input_grad[1] else None


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


class Relu(Function):
  def forward(self, x:LazyBuffer) -> LazyBuffer:
    self.ret = x.e(BinaryOps.MAX, x.const(0))
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return self.ret.const(0).e(BinaryOps.CMPLT, self.ret).e(BinaryOps.MUL, grad_output)


  # *** reduce ops ***
class Sum(Function):
  def forward(self, x:LazyBuffer, new_shape:Tuple[int, ...]) -> LazyBuffer:
    self.input_shape = x.shape
    return x.r(ReduceOps.SUM, new_shape)

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    return grad_output.expand(self.input_shape)

class Max(Function):
  def forward(self, x:LazyBuffer, new_shape:Tuple[int, ...]) -> LazyBuffer:
    self.x, self.ret = x, x.r(ReduceOps.MAX, new_shape)
    return self.ret

  def backward(self, grad_output:LazyBuffer) -> LazyBuffer:
    # 1s in locations where the max was chosen (can be two locations)
    max_is_1s = self.x.const(1.0).e(BinaryOps.SUB, self.x.e(BinaryOps.CMPLT, self.ret.expand(self.x.shape)))
    div = max_is_1s.r(ReduceOps.SUM, grad_output.shape).expand(self.x.shape)
    return max_is_1s.e(BinaryOps.DIV, div).e(BinaryOps.MUL, grad_output.expand(self.x.shape))
