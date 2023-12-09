from typing import Optional, Final, Tuple, Any
from dataclasses import dataclass
import numpy as np
import os
import functools

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
STV = getenv("STV") # show tensor values, is really lazydata values, but w/e

def all_int(x: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in x)
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (list, tuple) else x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))

@dataclass(frozen=True, order=True)
class DType:
  priority: int  # this determines when things get upcasted
  itemsize: int
  name: str
  np: Optional[type]  # TODO: someday this will be removed with the "remove numpy" project
  sz: int = 1
  def __repr__(self): return f"dtypes.{self.name}"


class dtypes:
  @staticmethod
  def is_float(x: DType) -> bool: return x in (dtypes.float16, dtypes.float32, dtypes.float64)
  @staticmethod
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  float16: Final[DType] = DType(9, 2, "half", np.float16)
  float32: Final[DType] = DType(10, 4, "float", np.float32)
  float = float32
  float64: Final[DType] = DType(11, 8, "double", np.float64)


DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}
