
from enum import Enum, auto
from typing import Optional

class UnaryOps(Enum): NEG = auto()
class BinaryOps(Enum):
  ADD = auto();
  MUL = auto();
  DIV = auto();
  SUB = auto();
  MOD = auto();
  MAX = auto();
  CMPLT = auto();

  # theese should no be here
  POW = auto();
  MATMUL = auto();

class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto()

class Device:
  DEFAULT = "CPU"
  _buffers = ["CPU"]
  @staticmethod
  def canonicalize(device:Optional[str]) -> str: return "CPU"

