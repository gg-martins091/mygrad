from typing import Optional, Final, Tuple, Any
from dataclasses import dataclass
import numpy as np


def all_int(x: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in x)
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
  def from_np(x) -> DType: return DTYPES_DICT[np.dtype(x).name]
  float32: Final[DType] = DType(10, 4, "float", np.float32)
  float = float32


DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}
