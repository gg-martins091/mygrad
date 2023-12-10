from typing import Iterator, Tuple, Union, Final, Optional, Any
from dataclasses import dataclass
import numpy as np
import os, functools, pathlib, hashlib, tempfile
from urllib import request
from tqdm import tqdm

@functools.lru_cache(maxsize=None)
def getenv(key, default=0): return type(default)(os.getenv(key, default))

DEBUG = getenv("DEBUG")
STV = getenv("STV") # show tensor values, is really lazydata values, but w/e

def all_int(x: Tuple[Any, ...]) -> bool: return all(isinstance(s, int) for s in x)
def argfix(*x): return tuple(x[0]) if x and x[0].__class__ in (list, tuple) else x
def argsort(x): return type(x)(sorted(range(len(x)), key=x.__getitem__))
def round_up(num, amt:int): return (num+amt-1)//amt * amt
def flatten(l:Iterator): return [item for sublist in l for item in sublist]
def dedup(x): return list(dict.fromkeys(x))   # retains list orderi
def make_pair(x:Union[int, Tuple[int, ...]], cnt=2) -> Tuple[int, ...]: return (x,)*cnt if isinstance(x, int) else x

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


  int8: Final[DType] = DType(1, 1, "char", np.int8)
  int16: Final[DType] = DType(3, 2, "short", np.int16)
  int32: Final[DType] = DType(5, 4, "int", np.int32)
  int64: Final[DType] = DType(7, 8, "long", np.int64)
  uint8: Final[DType] = DType(2, 1, "unsigned char", np.uint8)

DTYPES_DICT = {k: v for k, v in dtypes.__dict__.items() if not k.startswith('__') and not callable(v) and not v.__class__ == staticmethod}

_cache_dir: str = os.path.expanduser("~/.cache")

def fetch(url:str, name:Optional[Union[pathlib.Path, str]]=None, allow_caching=not getenv("DISABLE_HTTP_CACHE")) -> pathlib.Path:
  if url.startswith("/") or url.startswith("."): return pathlib.Path(url)
  fp = pathlib.Path(name) if name is not None and (isinstance(name, pathlib.Path) or '/' in name) else pathlib.Path(_cache_dir) / "tinygrad" / "downloads" / (name if name else hashlib.md5(url.encode('utf-8')).hexdigest())
  if not fp.is_file() or not allow_caching:
    with request.urlopen(url, timeout=10) as r:
      assert r.status == 200
      total_length = int(r.headers.get('content-length', 0))
      progress_bar = tqdm(total=total_length, unit='B', unit_scale=True, desc=url)
      (path := fp.parent).mkdir(parents=True, exist_ok=True)
      with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
        while chunk := r.read(16384): progress_bar.update(f.write(chunk))
        f.close()
        if (file_size:=os.stat(f.name).st_size) < total_length: raise RuntimeError(f"fetch size incomplete, {file_size} < {total_length}")
        pathlib.Path(f.name).rename(fp)
  return fp
