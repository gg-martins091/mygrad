
from enum import Enum, auto
from typing import Optional

class BinaryOps(Enum): ADD = auto(); MUL = auto()
class LoadOps(Enum): EMPTY = auto(); RAND = auto(); CONST = auto(); FROM = auto(); CONTIGUOUS = auto(); CUSTOM = auto()

class Device:
  DEFAULT = "CPU"
  _buffers = ["CPU"]
  @staticmethod
  def canonicalize(device:Optional[str]) -> str: return "CPU"

