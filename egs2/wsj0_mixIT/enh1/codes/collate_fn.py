from typing import Collection
from typing import Dict
from typing import List
from typing import Tuple
from typing import Union

import numpy as np
import torch
from typeguard import check_argument_types
from typeguard import check_return_type

from espnet.nets.pytorch_backend.nets_utils import pad_list


class CommonCollateFn:
    """Functor class of common_collate_fn()"""

    def __init__(
        self,
        float_pad_value: Union[float, int] = 0.0,
        int_pad_value: int = -32768,
        not_sequence: Collection[str] = (),
    ):
        assert check_argument_types()
        self.float_pad_value = float_pad_value
        self.int_pad_value = int_pad_value
        self.not_sequence = set(not_sequence)

    def __repr__(self):
        return (
            f"{self.__class__}(float_pad_value={self.float_pad_value}, "
            f"int_pad_value={self.float_pad_value})"
        )

    def __call__(
        self, data: Collection[Tuple[str, Dict[str, np.ndarray]]]
    ) -> Tuple[List[str], Dict[str, torch.Tensor]]:
        return common_collate_fn(
            data,
            float_pad_value=self.float_pad_value,
            int_pad_value=self.int_pad_value,
            not_sequence=self.not_sequence,
        )


def common_collate_fn(
    data: Collection[Tuple[str, Dict[str, np.ndarray]]],
    float_pad_value: Union[float, int] = 0.0,
    int_pad_value: int = -32768,
    not_sequence: Collection[str] = (),
) -> Tuple[List[str], Dict[str, torch.Tensor]]:
    """Concatenate ndarray-list to an array and convert to torch.Tensor.

    Examples:
        >>> from espnet2.samplers.constant_batch_sampler import ConstantBatchSampler,
        >>> import espnet2.tasks.abs_task
        >>> from espnet2.train.dataset import ESPnetDataset
        >>> sampler = ConstantBatchSampler(...)
        >>> dataset = ESPnetDataset(...)
        >>> keys = next(iter(sampler)
        >>> batch = [dataset[key] for key in keys]
        >>> batch = common_collate_fn(batch)
        >>> model(**batch)

        Note that the dict-keys of batch are propagated from
        that of the dataset as they are.

    """
    assert check_argument_types()
    uttids = [u for u, _ in data]
    data = [d for _, d in data]

    """
    Two examples in WSJ0-2mix:

    uttids['20l_20c_20la010v_1.7879_20cc010p_-1.7879']
    data[{'speech_mix': array([0.00036621, 0.00036621, 0.00024414, ..., 0.01504517,
                               0.01098633, 0.01089478], dtype=float32),
          'speech_ref1': array([1.5258789e-04, 9.1552734e-05, 1.8310547e-04, ...,
                                -1.5258789e-04, -2.1362305e-04, -1.2207031e-04],
                                dtype=float32),
          'speech_ref2': array([2.1362305e-04, 2.7465820e-04, 9.1552734e-05, ...,
                                1.5197754e-02, 1.1169434e-02, 1.1016846e-02],
                                dtype=float32)}]
    uttids['20d_40o_20dc010j_0.47296_40oo030c_-0.47296']
    data[{'speech_mix': array([0.00106812, 0.00183105, 0.00128174, ..., 0.00604248,
                               0.00238037, 0.00808716], dtype=float32),
          'speech_ref1': array([0.00106812, 0.00140381, 0.00125122, ..., 0.00280762,
                                -0.00131226, 0.00497437], dtype=float32),
          'speech_ref2': array([0.0000000e+00, 4.2724609e-04, 3.0517578e-05, ...,
                                3.2348633e-03, 3.6926270e-03, 3.1127930e-03],
                                dtype=float32)}]
    """

    assert all(set(data[0]) == set(d) for d in data), "dict-keys mismatching"
    assert all(
        not k.endswith("_lengths") for k in data[0]
    ), f"*_lengths is reserved: {list(data[0])}"

    output = {}
    for key in data[0]:
        # NOTE(kamo):
        # Each models, which accepts these values finally, are responsible
        # to repaint the pad_value to the desired value for each tasks.
        if data[0][key].dtype.kind == "i":
            pad_value = int_pad_value
        else:
            pad_value = float_pad_value

        array_list = [d[key] for d in data]

        # Assume the first axis is length:
        # tensor_list: Batch x (Length, ...)
        tensor_list = [torch.from_numpy(a) for a in array_list]
        # tensor: (Batch, Length, ...)
        tensor = pad_list(tensor_list, pad_value)
        output[key] = tensor

        assert all(len(d[key]) != 0 for d in data), [len(d[key]) for d in data]

        # lens: (Batch,)
        if key not in not_sequence:
            lens = torch.tensor([d[key].shape[0] for d in data], dtype=torch.long)
            output[key + "_lengths"] = lens

    output = (uttids, output)
    assert check_return_type(output)
    return output
