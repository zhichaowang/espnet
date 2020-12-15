from abc import ABC
from abc import abstractmethod
from collections import OrderedDict
from typing import List
from typing import Tuple

import torch


class AbsEnhancement(torch.nn.Module, ABC):
    # @abstractmethod
    # def output_size(self) -> int:
    #     raise NotImplementedError

    @abstractmethod
    def forward(
        self,
        input: torch.Tensor,
        ilens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        raise NotImplementedError

    @abstractmethod
    def forward_rawwav(
        self, input: torch.Tensor, ilens: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, OrderedDict]:
        raise NotImplementedError

    @abstractmethod
    def process_targets(
        self, input: torch.Tensor, target: List[torch.Tensor], ilens: torch.Tensor
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        raise NotImplementedError
