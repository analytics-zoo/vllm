from typing import Optional, Union

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup
from vllm.logger import init_logger

logger = init_logger(__name__)

class OneCCLCommunicator:

    def __init__(
            self,
            group: ProcessGroup,
            device: Union[int, str, torch.device],
    ):
        assert dist.is_initialized()
        self.rank = dist.get_rank(group)
        self.world_size = dist.get_world_size(group)

        self.group = group
        if isinstance(device, int):
            device = torch.device(f"xpu:{device}")
        elif isinstance(device, str):
            device = torch.device(device)
        # now `device` is a `torch.device` object
        assert isinstance(device, torch.device)
        self.device = device
    
    def send(self, tensor: torch.Tensor, dst: Optional[int]):
        # TODO: add more checks, refer to PyNcclCommunicator
        if dst is None:
            dst = (self.rank + 1) % self.world_size
        torch.distributed.send(tensor, dst, self.group)
    
    def recv(self,
             tensor: torch.Tensor,
             src: Optional[int]) :
        """Receives a tensor from the source rank."""
        """NOTE: `src` is the local rank of the source rank."""
        if src is None:
            src = (self.rank - 1) % self.world_size

        # tensor = torch.empty(size, dtype=dtype, device=self.device)
        torch.distributed.recv(tensor, src, self.group)
        return tensor

    

