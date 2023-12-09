from __future__ import annotations
from mygrad.lazy import LazyBuffer
from mygrad.tensor import Function
from mygrad.ops import LoadOps, BinaryOps, UnaryOps


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
