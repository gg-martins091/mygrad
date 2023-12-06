
from typing import Union, Optional
from .lazy import LazyBuffer
from .ops import LoadOps, Device
from .helpers import dtypes

class Tensor():
    __slots__= "lazydata", "grad"
    def __init__(self, data:Union[int, float, list, LazyBuffer], device:Optional[str]=None):
        device = Device.canonicalize(device)
        self.grad: Optional[Tensor] = None

        if isinstance(data, list):
            pass

        if isinstance(data, (int, float)):
            data = LazyBuffer.loadop(LoadOps.CONST, tuple(), dtypes.float32, device)

        self.lazydata = data

    def __repr__(self):
        return f"<Tensor {self.lazydata!r} on {self.device} with grad {(self.grad.lazydata if self.grad else None)!r}>"

    @property
    def device(self) -> str: return self.lazydata.device
